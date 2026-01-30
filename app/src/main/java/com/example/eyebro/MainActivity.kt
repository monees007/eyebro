package com.example.eyebro

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.media.Image
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Build
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Frame
import com.google.ar.core.Session
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.NotYetAvailableException
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.label.ImageLabeling
import com.google.mlkit.vision.label.defaults.ImageLabelerOptions
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class MainActivity : AppCompatActivity(), GLSurfaceView.Renderer {

    private lateinit var surfaceView: GLSurfaceView
    private lateinit var warningText: TextView
    private lateinit var statusText: TextView

    private var session: Session? = null
    private var isDepthSupported = false
    private val backgroundRenderer = SimpleBackgroundRenderer()

    // ML Kit Labeler
    private val labeler = ImageLabeling.getClient(ImageLabelerOptions.DEFAULT_OPTIONS)
    private var lastLabelingTime = 0L
    private val LABELING_INTERVAL_MS = 1000L // Run ML only once per second max
    private var currentObjectLabel = ""

    // Display metrics
    private var viewportWidth = 0
    private var viewportHeight = 0

    // --- DETECTION REGION (ROI) ---
    private val ROI_LEFT = 0.10f
    private val ROI_RIGHT = 0.90f
    private val ROI_TOP = 0.20f
    private val ROI_BOTTOM = 0.80f
    private val ROI_DROP_START = 0.60f

    // Safety Thresholds
    private val OBSTACLE_LIMIT_MM = 1200
    private val DROP_OFF_LIMIT_MM = 4000

    private val VIBRATION_INTERVAL = 500L
    private var lastVibrationTime = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        surfaceView = findViewById(R.id.surfaceView)
        warningText = findViewById(R.id.warningText)
        statusText = findViewById(R.id.statusText)

        surfaceView.preserveEGLContextOnPause = true
        surfaceView.setEGLContextClientVersion(2)
        surfaceView.setEGLConfigChooser(8, 8, 8, 8, 16, 0)
        surfaceView.setRenderer(this)
        surfaceView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        surfaceView.setWillNotDraw(false)

        val overlayView = OverlayView(this)
        addContentView(overlayView, ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        ))

        checkCameraPermission()
    }

    private fun checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 0)
        }
    }

    override fun onResume() {
        super.onResume()
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
                    statusText.text = "Depth Active"
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
    }

    // --- RENDERER CALLBACKS ---

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0.1f, 0.1f, 0.1f, 1.0f)
        backgroundRenderer.createOnGlThread()
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        viewportWidth = width
        viewportHeight = height
        GLES20.glViewport(0, 0, width, height)
        session?.setDisplayGeometry(windowManager.defaultDisplay.rotation, width, height)
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
            // Avoid crashing on session errors
        }
    }

    // --- CORE LOGIC ---

    private fun processDepth(frame: Frame) {
        try {
            val depthImage = frame.acquireDepthImage16Bits()
            if (depthImage.planes.isEmpty()) {
                depthImage.close()
                return
            }

            val buffer = depthImage.planes[0].buffer.order(ByteOrder.nativeOrder())
            val width = depthImage.width
            val height = depthImage.height
            val step = 6

            val startX = (width * ROI_LEFT).toInt()
            val endX = (width * ROI_RIGHT).toInt()
            val startY = (height * ROI_TOP).toInt()
            val endY = (height * ROI_BOTTOM).toInt()
            val dropStartY = (height * ROI_DROP_START).toInt()

            var closePixels = 0
            var deepPixels = 0

            for (y in startY until endY step step) {
                for (x in startX until endX step step) {
                    val index = (y * depthImage.planes[0].rowStride) + (x * depthImage.planes[0].pixelStride)
                    val distanceMm = getDistanceAt(buffer, index)

                    if (distanceMm in 1 until OBSTACLE_LIMIT_MM) {
                        closePixels++
                    }

                    if (y >= dropStartY) {
                        if (distanceMm > DROP_OFF_LIMIT_MM) {
                            deepPixels++
                        }
                    }
                }
            }

            depthImage.close()

            val roiWidthPixels = (endX - startX) / step
            val roiHeightPixels = (endY - startY) / step
            val dropHeightPixels = (endY - dropStartY) / step

            val totalPixels = roiWidthPixels * roiHeightPixels
            val totalDropPixels = roiWidthPixels * dropHeightPixels

            val obsThreshold = totalPixels * 0.15
            val dropThreshold = totalDropPixels * 0.25

            // --- OBSTACLE IDENTIFICATION TRIGGER ---
            // Only run ML if we actually see an obstacle, to save battery/perf
            if (closePixels > obsThreshold) {
                val currentTime = System.currentTimeMillis()
                if (currentTime - lastLabelingTime > LABELING_INTERVAL_MS) {
                    lastLabelingTime = currentTime
                    identifyObstacle(frame)
                }
            } else {
                // Reset label if path is clear
                if (System.currentTimeMillis() - lastLabelingTime > 2000) {
                    currentObjectLabel = ""
                }
            }

            runOnUiThread {
                if (deepPixels > dropThreshold) {
                    triggerObstacleWarning("EDGE DETECTED!", Color.RED)
                } else if (closePixels > obsThreshold) {
                    val msg = if (currentObjectLabel.isNotEmpty()) "STOP! ($currentObjectLabel)" else "STOP!"
                    triggerObstacleWarning(msg, Color.parseColor("#FFA500"))
                } else {
                    clearWarning()
                }
            }

        } catch (e: Exception) {
            // Depth not available
        }
    }

    /**
     * Uses ML Kit to identify what is in the camera frame.
     * Note: Requires 'com.google.mlkit:image-labeling' dependency.
     */
    private fun identifyObstacle(frame: Frame) {
        try {
            // Acquire the high-quality RGB camera image
            val cameraImage = frame.acquireCameraImage()

            // Note: rotationDegrees usually 90 for portrait on most phones
            val rotationDegrees = 90
            val inputImage = InputImage.fromMediaImage(cameraImage, rotationDegrees)

            labeler.process(inputImage)
                .addOnSuccessListener { labels ->
                    // Get the most confident label
                    if (labels.isNotEmpty()) {
                        val topLabel = labels[0]
                        // Only show if confidence is decent (> 70%)
                        if (topLabel.confidence > 0.8) {
                            currentObjectLabel = topLabel.text
                        }
                    }
                    cameraImage.close() // IMPORTANT: Must close or ARCore will freeze
                }
                .addOnFailureListener {
                    cameraImage.close()
                }
                .addOnCompleteListener {
                    // Ensure closed even if logic fails
                    try { cameraImage.close() } catch(e: Exception) {}
                }

        } catch (e: NotYetAvailableException) {
            // Image not ready, skip this frame
        } catch (e: Exception) {
            Log.e("Eyebro", "ML Error: ${e.message}")
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
        if (currentTime - lastVibrationTime > VIBRATION_INTERVAL) {
            vibrateDevice()
            lastVibrationTime = currentTime
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

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createOneShot(200, VibrationEffect.DEFAULT_AMPLITUDE))
        } else {
            @Suppress("DEPRECATION")
            vibrator.vibrate(200)
        }
    }

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

            val rect = RectF(
                w * ROI_LEFT,
                h * ROI_TOP,
                w * ROI_RIGHT,
                h * ROI_BOTTOM
            )
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
            frame.transformDisplayUvCoords(quadTexCoordBuffer, transformedTexCoordBuffer)

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