package com.example.eyebro

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.YuvImage
import android.media.AudioAttributes
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Build
import android.os.Bundle
import android.os.VibrationAttributes
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import com.google.ar.core.Pose
import com.google.ar.core.Session
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.NotYetAvailableException
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.label.ImageLabeling
import com.google.mlkit.vision.label.defaults.ImageLabelerOptions
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.channels.FileChannel
import java.util.Locale
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class MainActivity : AppCompatActivity(), GLSurfaceView.Renderer, TextToSpeech.OnInitListener {

    private lateinit var surfaceView: GLSurfaceView
    private lateinit var warningText: TextView
    private lateinit var statusText: TextView
    private lateinit var settingsFab: FloatingActionButton

    private var session: Session? = null
    private var isDepthSupported = false
    private val backgroundRenderer = SimpleBackgroundRenderer()

    // --- SETTINGS & PREFS ---
    private lateinit var prefs: SharedPreferences
    private var isObjDetectionEnabled = true
    private var isVibrationEnabled = true
    private var isTtsEnabled = true
    private var selectedModelType = "mlkit" // "mlkit" or "yolo"

    // --- ML Kit Labeler ---
    private val labeler = ImageLabeling.getClient(ImageLabelerOptions.DEFAULT_OPTIONS)

    // --- YOLO / TFLite ---
    private var tfliteInterpreter: Interpreter? = null
    private val YOLO_INPUT_SIZE = 640
    // Simplified COCO classes list (partial for demo)
    private val yoloClasses = listOf("person", "bicycle", "car", "motorcycle", "airplane", "bus", "bike", "vehical", "wall", "edge", "door", "laptop")

    private var lastLabelingTime = 0L
    private val LABELING_INTERVAL_MS = 1000L
    private var currentObjectLabel = ""

    // --- TTS ---
    private var tts: TextToSpeech? = null
    private var lastTtsTime = 0L
    private val TTS_INTERVAL_MS = 2500L

    // --- DISPLAY METRICS ---
    private var viewportWidth = 0
    private var viewportHeight = 0

    // --- ROI & SAFETY ---
    private val ROI_LEFT = 0.15f
    private val ROI_RIGHT = 0.85f
    private val ROI_TOP = 0.20f
    private val ROI_BOTTOM = 0.75f
    private val ROI_DROP_START = 0.60f
    private val OBSTACLE_LIMIT_MM = 1200
    private val DROP_OFF_LIMIT_MM = 4000

    private val VIBRATION_INTERVAL = 500L
    private var lastVibrationTime = 0L

    // Reusable array for matrix calculations to avoid GC
    private val poseMatrix = FloatArray(16)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        prefs = getSharedPreferences("EyebroPrefs", Context.MODE_PRIVATE)
        tts = TextToSpeech(this, this)

        surfaceView = findViewById(R.id.surfaceView)
        warningText = findViewById(R.id.warningText)
        statusText = findViewById(R.id.statusText)
        settingsFab = findViewById(R.id.settingsFab)

        settingsFab.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        setupSurfaceView()

        val overlayView = OverlayView(this)
        addContentView(overlayView, ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        ))

        checkCameraPermission()
    }

    private fun setupSurfaceView() {
        surfaceView.preserveEGLContextOnPause = true
        surfaceView.setEGLContextClientVersion(2)
        surfaceView.setEGLConfigChooser(8, 8, 8, 8, 16, 0)
        surfaceView.setRenderer(this)
        surfaceView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        surfaceView.setWillNotDraw(false)
    }

    override fun onResume() {
        super.onResume()
        refreshSettings()
        setupARSession()
    }

    private fun refreshSettings() {
        isObjDetectionEnabled = prefs.getBoolean("enable_obj", true)
        isVibrationEnabled = prefs.getBoolean("enable_vib", true)
        isTtsEnabled = prefs.getBoolean("enable_tts", true)
        val newModel = prefs.getString("model_type", "mlkit") ?: "mlkit"

        // Initialize/Switch Model logic
        if (newModel != selectedModelType) {
            selectedModelType = newModel
            if (selectedModelType == "yolo" && tfliteInterpreter == null) {
                initYoloModel()
            }
        }
        // Always try to init YOLO if selected, even if first run
        if (selectedModelType == "yolo" && tfliteInterpreter == null) {
            initYoloModel()
        }
    }

    private fun initYoloModel() {
        try {
            val assetManager = assets
            val descriptor = assetManager.openFd("yolov8n.tflite")
            val inputStream = FileInputStream(descriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = descriptor.startOffset
            val declaredLength = descriptor.declaredLength
            val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

            val options = Interpreter.Options()
            options.setNumThreads(4)
            tfliteInterpreter = Interpreter(modelBuffer, options)
            Toast.makeText(this, "YOLO Loaded", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e("Eyebro", "Error loading YOLO: ${e.message}")
            Toast.makeText(this, "Failed to load YOLO model", Toast.LENGTH_LONG).show()
            // Fallback
            selectedModelType = "mlkit"
            prefs.edit().putString("model_type", "mlkit").apply()
        }
    }

    private fun setupARSession() {
        if (session == null) {
            try {
                if (ArCoreApk.getInstance().requestInstall(this, true) == ArCoreApk.InstallStatus.INSTALL_REQUESTED) {
                    return
                }
                session = Session(this)
                val config = session!!.config
                if (session!!.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
                    config.depthMode = Config.DepthMode.AUTOMATIC
                    isDepthSupported = true
                    statusText.text = "Depth Active ($selectedModelType)"
                } else {
                    statusText.text = "Depth Not Supported"
                }
                session!!.configure(config)
            } catch (e: Exception) {
                Toast.makeText(this, "ARCore Failed: ${e.message}", Toast.LENGTH_LONG).show()
                return
            }
        }
        try {
            session?.resume()
            surfaceView.onResume()
        } catch (e: CameraNotAvailableException) {
            Toast.makeText(this, "Camera not available", Toast.LENGTH_LONG).show()
        }
    }

    override fun onPause() {
        super.onPause()
        surfaceView.onPause()
        session?.pause()
        tts?.stop()
    }

    override fun onDestroy() {
        tts?.shutdown()
        tfliteInterpreter?.close()
        super.onDestroy()
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts?.setLanguage(Locale.US)
        }
    }

    // --- RENDERER METHODS (onSurfaceCreated, onSurfaceChanged, onDrawFrame) ---
    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0.1f, 0.1f, 0.1f, 1.0f)
        backgroundRenderer.createOnGlThread()
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        viewportWidth = width
        viewportHeight = height
        GLES20.glViewport(0, 0, width, height)

        // FIX: Replaced deprecated windowManager.defaultDisplay with version-safe check
        val rotation = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            display?.rotation ?: 0
        } else {
            @Suppress("DEPRECATION")
            windowManager.defaultDisplay.rotation
        }
        session?.setDisplayGeometry(rotation, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        val currentSession = session ?: return

        try {
            currentSession.setCameraTextureName(backgroundRenderer.textureId)
            val frame = currentSession.update()
            backgroundRenderer.draw(frame)

            if (isDepthSupported) {
                processDepth(frame)
            }
        } catch (t: Throwable) {
            // Ignore frame errors
        }
    }

    // --- CORE LOGIC ---

    private fun processDepth(frame: Frame) {
        try {
            // --- 1. DEVICE ANGLE CHECK ---
            // Calculate looking direction (Forward Vector Y-component)
            frame.camera.displayOrientedPose.toMatrix(poseMatrix, 0)
            // Column 2 (index 8, 9, 10) is the "Backward" vector in OpenGL.
            // Forward is negative of that. We want Y component (index 9).
            val forwardY = -poseMatrix[9]

            // Safe range: -0.35 (distinctly down) to -0.85 (steeply down)
            // > -0.35 means looking too horizontal/up (Adjusted from -0.2 to prioritize angle correction)
            // < -0.85 means looking too vertical/down at feet
            val isAngleTooHigh = forwardY > -0.40
            val isAngleTooLow = forwardY < -0.75

            val depthImage = frame.acquireDepthImage16Bits() ?: return
            val buffer = depthImage.planes[0].buffer.order(ByteOrder.nativeOrder())
            val width = depthImage.width
            val height = depthImage.height
            val step = 8 // Slightly increased step for performance

            val startX = (width * ROI_LEFT).toInt()
            val endX = (width * ROI_RIGHT).toInt()
            val startY = (height * ROI_TOP).toInt()
            val endY = (height * ROI_BOTTOM).toInt()

            var closePixels = 0
            var deepPixels = 0

            // Tracking for Stair Detection
            var upperDepthSum = 0L
            var lowerDepthSum = 0L
            var upperCount = 0
            var lowerCount = 0
            val midY = startY + (endY - startY) / 2

            for (y in startY until endY step step) {
                for (x in startX until endX step step) {
                    val index = (y * depthImage.planes[0].rowStride) + (x * depthImage.planes[0].pixelStride)
                    val distanceMm = getDistanceAt(buffer, index)

                    if (distanceMm <= 0) continue

                    // 1. Obstacle Detection
                    if (distanceMm in 1 until OBSTACLE_LIMIT_MM) {
                        closePixels++
                    }

                    // 2. Depth Gradient (for Stairs/Drop-offs)
                    if (distanceMm > DROP_OFF_LIMIT_MM) {
                        deepPixels++
                    }

                    // 3. Collect Averages for Stair Signature
                    if (y < midY) {
                        upperDepthSum += distanceMm
                        upperCount++
                    } else {
                        lowerDepthSum += distanceMm
                        lowerCount++
                    }
                }
            }
            depthImage.close()

            // --- CALCULATIONS ---
            val totalPixels = ((endX - startX) / step) * ((endY - startY) / step)
            val obsThreshold = totalPixels * 0.15
            val dropThreshold = totalPixels * 0.40 // Lowered slightly to capture stairs

            // Stair Signature: Are the "far" pixels significantly deeper than "near" pixels
            // while both are beyond the floor range?
            val avgUpperDepth = if (upperCount > 0) upperDepthSum / upperCount else 0
            val avgLowerDepth = if (lowerCount > 0) lowerDepthSum / lowerCount else 0

            // If the top of the frame is significantly deeper than the bottom,
            // and we have many deep pixels, it's likely a staircase descending.
            val isStaircase = deepPixels > dropThreshold && (avgUpperDepth > avgLowerDepth + 500)

            // --- WARNING TRIGGERS ---
            runOnUiThread {
                when {
                    // Priority 1: High Obstacle
                    closePixels > obsThreshold -> {
                        identifyObstacleMlKit(frame)
                        val labelText = if (isObjDetectionEnabled && currentObjectLabel.isNotEmpty())
                            "STOP! ($currentObjectLabel)" else "STOP!"
                        triggerObstacleWarning(labelText, Color.parseColor("#FFA500"))
                    }

                    // Priority 2: Angle Correction
                    isAngleTooHigh -> triggerObstacleWarning("Tilt Phone Down ↘", Color.YELLOW)

                    // Priority 3: Stairs vs Drop-off
                    isStaircase -> triggerObstacleWarning("Descending Stairs Ahead ⬇️", Color.CYAN)

                    deepPixels > dropThreshold -> triggerObstacleWarning("Very Deep Surface Detected", Color.RED)

                    // Priority 4: Low Angle
                    isAngleTooLow -> triggerObstacleWarning("Tilt Phone Up ↗", Color.YELLOW)

                    else -> clearWarning()
                }
            }

        } catch (e: Exception) {
            // Log error
        }
    }

    // --- ML KIT IMPLEMENTATION ---
    private fun identifyObstacleMlKit(frame: Frame) {
        try {
            val cameraImage = frame.acquireCameraImage()
            // ML Kit handles YUV420 directly, very efficient
            val rotationDegrees = 90 // Should calculate dynamically based on display rotation
            val inputImage = InputImage.fromMediaImage(cameraImage, rotationDegrees)

            labeler.process(inputImage)
                .addOnSuccessListener { labels ->
                    if (labels.isNotEmpty() && labels[0].confidence > 0.75) {
                        currentObjectLabel = labels[0].text
                    }
                    cameraImage.close()
                }
                .addOnFailureListener { cameraImage.close() }
                .addOnCompleteListener {
                    try { cameraImage.close() } catch(e: Exception) {}
                }
        } catch (e: NotYetAvailableException) {
            // Skip
        } catch (e: Exception) {
            Log.e("Eyebro", "ML Kit Error: ${e.message}")
        }
    }

    // --- YOLO IMPLEMENTATION ---
    private fun identifyObstacleYolo(frame: Frame) {
        if (tfliteInterpreter == null) return

        try {
            val cameraImage = frame.acquireCameraImage()
            // 1. Convert YUV to Bitmap
            val yBuffer = cameraImage.planes[0].buffer
            val uBuffer = cameraImage.planes[1].buffer
            val vBuffer = cameraImage.planes[2].buffer

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)

            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)

            val yuvImage = YuvImage(nv21, ImageFormat.NV21, cameraImage.width, cameraImage.height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, cameraImage.width, cameraImage.height), 75, out)
            val imageBytes = out.toByteArray()
            val originalBitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
            cameraImage.close()

            // 2. Resize and Rotate for Model
            val matrix = Matrix()
            matrix.postRotate(90f) // ARCore default portrait
            val rotatedBitmap = Bitmap.createBitmap(originalBitmap, 0, 0, originalBitmap.width, originalBitmap.height, matrix, true)
            val scaledBitmap = Bitmap.createScaledBitmap(rotatedBitmap, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, true)

            // 3. Prepare Input Tensor (Float Buffer)
            val inputBuffer = ByteBuffer.allocateDirect(1 * YOLO_INPUT_SIZE * YOLO_INPUT_SIZE * 3 * 4)
            inputBuffer.order(ByteOrder.nativeOrder())
            val pixels = IntArray(YOLO_INPUT_SIZE * YOLO_INPUT_SIZE)
            scaledBitmap.getPixels(pixels, 0, YOLO_INPUT_SIZE, 0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE)

            for (pixel in pixels) {
                // Normalize to 0-1 range
                inputBuffer.putFloat(((pixel shr 16 and 0xFF) / 255.0f))
                inputBuffer.putFloat(((pixel shr 8 and 0xFF) / 255.0f))
                inputBuffer.putFloat(((pixel and 0xFF) / 255.0f))
            }

            // 4. Run Inference
            // YOLOv8 output: [1, 84, 8400] -> [Batch, (4 box coords + 80 classes), Anchors]
            val outputBuffer = Array(1) { Array(84) { FloatArray(8400) } }
            tfliteInterpreter?.run(inputBuffer, outputBuffer)

            // 5. Parse Output (Simplified for brevity: Find max confidence class)
            // In a real app, you would run Non-Maximum Suppression (NMS) here.
            var maxConf = 0.0f
            var maxClassIndex = -1

            val numAnchors = 8400
            // Loop through anchors
            for (i in 0 until numAnchors) {
                // Confidence scores start at index 4 (0-3 are x,y,w,h)
                // Find best class for this anchor
                var anchorMaxConf = 0.0f
                var anchorClass = -1

                // Assuming 80 classes
                for (c in 0 until 80) {
                    val conf = outputBuffer[0][4 + c][i]
                    if (conf > anchorMaxConf) {
                        anchorMaxConf = conf
                        anchorClass = c
                    }
                }

                if (anchorMaxConf > maxConf && anchorMaxConf > 0.5f) { // Threshold 0.5
                    maxConf = anchorMaxConf
                    maxClassIndex = anchorClass
                }
            }

            if (maxClassIndex != -1 && maxClassIndex < yoloClasses.size) {
                currentObjectLabel = yoloClasses[maxClassIndex]
            }

        } catch (e: Exception) {
            Log.e("Eyebro", "YOLO Error: ${e.message}")
        }
    }

    private fun getDistanceAt(buffer: ByteBuffer, index: Int): Int {
        val lowByte = buffer.get(index).toInt() and 0xFF
        val highByte = buffer.get(index + 1).toInt() and 0xFF
        return (highByte shl 8) or lowByte
    }

    private fun triggerObstacleWarning(text: String, color: Int) {
        warningText.visibility = View.VISIBLE
        warningText.text = text
        warningText.setTextColor(color)

        val currentTime = System.currentTimeMillis()

        if (isVibrationEnabled && (currentTime - lastVibrationTime > VIBRATION_INTERVAL)) {
            vibrateDevice()
            lastVibrationTime = currentTime
        }

        if (isTtsEnabled && (currentTime - lastTtsTime > TTS_INTERVAL_MS)) {
            val speakText = text.replace(Regex("[^a-zA-Z0-9 ]"), "")
            tts?.speak(speakText, TextToSpeech.QUEUE_FLUSH, null, "WarningId")
            lastTtsTime = currentTime
        }
    }

    private fun clearWarning() {
        warningText.visibility = View.GONE
    }

    private fun vibrateDevice() {
        val vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val vibratorManager = getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
            vibratorManager.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }

        // Safety check
        if (!vibrator.hasVibrator()) return

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            // API 33+ (Android 13+): Use VibrationAttributes
            val effect = VibrationEffect.createOneShot(200, 255)
            val vibrationAttributes = VibrationAttributes.Builder()
                .setUsage(VibrationAttributes.USAGE_ACCESSIBILITY)
                .build()
            vibrator.vibrate(effect, vibrationAttributes)
        } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            // API 26-32: Use AudioAttributes
            val effect = VibrationEffect.createOneShot(200, 255)
            val audioAttributes = AudioAttributes.Builder()
                .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                .build()
            @Suppress("DEPRECATION")
            vibrator.vibrate(effect, audioAttributes)
        } else {
            // Legacy
            @Suppress("DEPRECATION")
            vibrator.vibrate(200)
        }
    }

    private fun checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 0)
        }
    }

    // --- INNER CLASSES (Overlay & Renderer) ---
    inner class OverlayView(context: Context) : View(context) {
        private val paint = Paint().apply {
            color = Color.CYAN
            style = Paint.Style.STROKE
            strokeWidth = 10f
            isAntiAlias = true
        }
        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            val w = width.toFloat()
            val h = height.toFloat()
            val rect = RectF(w * ROI_LEFT, h * ROI_TOP, w * ROI_RIGHT, h * ROI_BOTTOM)
            canvas.drawRect(rect, paint)
        }
    }

    class SimpleBackgroundRenderer {
        private val VERTEX_SHADER = """
            attribute vec4 a_Position;
            attribute vec2 a_TexCoord;
            varying vec2 v_TexCoord;
            void main() {
               gl_Position = a_Position;
               v_TexCoord = a_TexCoord;
            }
        """
        private val FRAGMENT_SHADER = """
            #extension GL_OES_EGL_image_external : require
            precision mediump float;
            varying vec2 v_TexCoord;
            uniform samplerExternalOES sTexture;
            void main() {
                gl_FragColor = texture2D(sTexture, v_TexCoord);
            }
        """
        private val QUAD_COORDS = floatArrayOf(-1.0f, -1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, +1.0f)
        private val QUAD_TEX_COORDS = floatArrayOf(0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f)

        var textureId = -1
        private var programId = 0
        private var positionAttrib = 0
        private var texCoordAttrib = 0
        private var textureUniform = 0

        private var quadPositionBuffer: FloatBuffer = ByteBuffer.allocateDirect(32).order(ByteOrder.nativeOrder()).asFloatBuffer().apply { put(QUAD_COORDS); position(0) }
        private var quadTexCoordBuffer: FloatBuffer = ByteBuffer.allocateDirect(32).order(ByteOrder.nativeOrder()).asFloatBuffer().apply { put(QUAD_TEX_COORDS); position(0) }
        private var transformedTexCoordBuffer: FloatBuffer = ByteBuffer.allocateDirect(32).order(ByteOrder.nativeOrder()).asFloatBuffer()

        fun createOnGlThread() {
            val textures = IntArray(1)
            GLES20.glGenTextures(1, textures, 0)
            textureId = textures[0]
            GLES20.glBindTexture(36197, textureId)
            GLES20.glTexParameteri(36197, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(36197, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(36197, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(36197, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)

            val vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, VERTEX_SHADER)
            val fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, FRAGMENT_SHADER)
            programId = GLES20.glCreateProgram()
            GLES20.glAttachShader(programId, vertexShader)
            GLES20.glAttachShader(programId, fragmentShader)
            GLES20.glLinkProgram(programId)

            positionAttrib = GLES20.glGetAttribLocation(programId, "a_Position")
            texCoordAttrib = GLES20.glGetAttribLocation(programId, "a_TexCoord")
            textureUniform = GLES20.glGetUniformLocation(programId, "sTexture")
        }

        fun draw(frame: Frame) {
            GLES20.glUseProgram(programId)
            quadTexCoordBuffer.position(0)
            transformedTexCoordBuffer.position(0)

            // FIX: Replaced deprecated transformDisplayUvCoords with transformCoordinates2d
            // NOTE: Swapped VIEW_NORMALIZED and TEXTURE_NORMALIZED to fix orientation
            frame.transformCoordinates2d(
                Coordinates2d.VIEW_NORMALIZED,
                quadTexCoordBuffer,
                Coordinates2d.TEXTURE_NORMALIZED,
                transformedTexCoordBuffer
            )

            quadPositionBuffer.position(0)
            GLES20.glVertexAttribPointer(positionAttrib, 2, GLES20.GL_FLOAT, false, 0, quadPositionBuffer)
            transformedTexCoordBuffer.position(0)
            GLES20.glVertexAttribPointer(texCoordAttrib, 2, GLES20.GL_FLOAT, false, 0, transformedTexCoordBuffer)
            GLES20.glEnableVertexAttribArray(positionAttrib)
            GLES20.glEnableVertexAttribArray(texCoordAttrib)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            GLES20.glBindTexture(36197, textureId)
            GLES20.glUniform1i(textureUniform, 0)
            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
            GLES20.glDisableVertexAttribArray(positionAttrib)
            GLES20.glDisableVertexAttribArray(texCoordAttrib)
        }

        private fun loadShader(type: Int, shaderCode: String): Int {
            val shader = GLES20.glCreateShader(type)
            GLES20.glShaderSource(shader, shaderCode)
            GLES20.glCompileShader(shader)
            return shader
        }
    }
}