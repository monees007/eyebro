package com.example.eyebro

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Build
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
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
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import com.google.ar.core.Session
import com.google.ar.core.exceptions.CameraNotAvailableException
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

    // Display metrics for ARCore
    private var viewportWidth = 0
    private var viewportHeight = 0

    // --- FINE TUNING: DETECTION REGION (0.0 to 1.0) ---
    // Change these values to adjust both the visual box and the detection logic simultaneously.
    private val ROI_LEFT = 0.15f   // Left edge at 25% screen width
    private val ROI_RIGHT = 0.85f  // Right edge at 75% screen width
    private val ROI_TOP = 0.25f    // Top edge at 35% screen height
    private val ROI_BOTTOM = 0.80f // Bottom edge at 90% screen height (leaving 10% for feet)

    // Drop-off detection (ground too far) only makes sense for the floor,
    // so we only trigger it in the lower portion of the ROI.
    private val ROI_DROP_START = 0.60f

    // Safety Thresholds
    private val OBSTACLE_LIMIT_MM = 1200 // Alert if object is within 1.2 meters
    private val DROP_OFF_LIMIT_MM = 2500 // Alert if ground is deeper than 2.5 meters (Roof Edge/Stairs)

    private val VIBRATION_INTERVAL = 500L // Don't vibrate constantly
    private var lastVibrationTime = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        surfaceView = findViewById(R.id.surfaceView)
        warningText = findViewById(R.id.warningText)
        statusText = findViewById(R.id.statusText)

        // Setup OpenGL Renderer for ARCore
        surfaceView.preserveEGLContextOnPause = true
        surfaceView.setEGLContextClientVersion(2)
        surfaceView.setEGLConfigChooser(8, 8, 8, 8, 16, 0)
        surfaceView.setRenderer(this)
        surfaceView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        surfaceView.setWillNotDraw(false)

        // Add the visual overlay for the observed area
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

                // IMPORTANT: Enable Depth API
                if (session!!.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
                    config.depthMode = Config.DepthMode.AUTOMATIC
                    isDepthSupported = true
                    statusText.text = "eyebro: Depth Active"
                } else {
                    statusText.text = "eyebro: Depth Not Supported"
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

    // --- AR CORE LOGIC LOOP ---

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0.1f, 0.1f, 0.1f, 1.0f)

        // Prepare the background renderer (compile shaders, gen texture)
        backgroundRenderer.createOnGlThread()
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        viewportWidth = width
        viewportHeight = height
        GLES20.glViewport(0, 0, width, height)
        // Notify ARCore session of the display geometry so it can calculate UVs correctly
        session?.setDisplayGeometry(windowManager.defaultDisplay.rotation, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        // Clear screen
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)

        val currentSession = session ?: return

        try {
            // Bind the texture ID to the session so it knows where to push the camera feed
            currentSession.setCameraTextureName(backgroundRenderer.textureId)

            // Update session (this updates the texture with new camera frame)
            val frame = currentSession.update()

            // Draw the camera background
            backgroundRenderer.draw(frame)

            // Run existing Depth Logic
            if (isDepthSupported) {
                processDepth(frame)
            }

        } catch (t: Throwable) {
            // Handle errors
        }
    }

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
            val step = 6 // Higher step for better performance scanning

            // Calculate active pixel area using the ROI constants
            val startX = (width * ROI_LEFT).toInt()
            val endX = (width * ROI_RIGHT).toInt()
            val startY = (height * ROI_TOP).toInt()
            val endY = (height * ROI_BOTTOM).toInt()

            // Calculate where the "Drop-off" logic starts (lower part of screen)
            val dropStartY = (height * ROI_DROP_START).toInt()

            var closePixels = 0
            var deepPixels = 0

            // Single loop scanning the unified Region of Interest
            for (y in startY until endY step step) {
                for (x in startX until endX step step) {
                    val index = (y * depthImage.planes[0].rowStride) + (x * depthImage.planes[0].pixelStride)
                    val distanceMm = getDistanceAt(buffer, index)

                    // 1. Obstacle Check (Applies to the whole ROI)
                    // If distance is valid (>0) and too close (< LIMIT)
                    if (distanceMm in 1 until OBSTACLE_LIMIT_MM) {
                        closePixels++
                    }

                    // 2. Drop-off Check (Applies only to the lower portion of ROI)
                    // If we are looking at the floor (y > dropStartY) and it suddenly gets deep
                    if (y >= dropStartY) {
                        if (distanceMm > DROP_OFF_LIMIT_MM) {
                            deepPixels++
                        }
                    }
                }
            }

            depthImage.close()

            // Calculate pixel counts to determine thresholds
            val roiWidthPixels = (endX - startX) / step
            val roiHeightPixels = (endY - startY) / step
            val dropHeightPixels = (endY - dropStartY) / step

            val totalPixels = roiWidthPixels * roiHeightPixels
            val totalDropPixels = roiWidthPixels * dropHeightPixels

            // Trigger thresholds (percentages of the scanned area)
            val obsThreshold = totalPixels * 0.15 // 15% of box is blocked
            val dropThreshold = totalDropPixels * 0.25 // 25% of floor area is deep

            runOnUiThread {
                if (deepPixels > dropThreshold) {
                    triggerObstacleWarning("EDGE DETECTED!", Color.RED)
                } else if (closePixels > obsThreshold) {
                    triggerObstacleWarning("STOP!", Color.parseColor("#FFA500")) // Orange
                } else {
                    clearWarning()
                }
            }

        } catch (e: Exception) {
            // Depth image not available yet
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

    /**
     * Helper View to draw the observation box on screen.
     */
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

            // Use the SAME configuration values as the detector
            val left = w * ROI_LEFT
            val right = w * ROI_RIGHT
            val top = h * ROI_TOP
            val bottom = h * ROI_BOTTOM

            val rect = RectF(left, top, right, bottom)
            canvas.drawRect(rect, paint)
        }
    }

    /**
     * Simple renderer to handle the camera feed texture.
     */
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

        // 4 vertices, 2 floats per vertex (X, Y)
        private val QUAD_COORDS = floatArrayOf(
            -1.0f, -1.0f, // bottom left
            -1.0f, +1.0f, // top left
            +1.0f, -1.0f, // bottom right
            +1.0f, +1.0f  // top right
        )

        // 4 vertices, 2 floats per vertex (U, V) - Standard Quad
        private val QUAD_TEX_COORDS = floatArrayOf(
            0.0f, 1.0f,
            0.0f, 0.0f,
            1.0f, 1.0f,
            1.0f, 0.0f
        )

        var textureId = -1
        private var programId = 0
        private var positionAttrib = 0
        private var texCoordAttrib = 0
        private var textureUniform = 0

        private var quadPositionBuffer: FloatBuffer
        private var quadTexCoordBuffer: FloatBuffer
        private var transformedTexCoordBuffer: FloatBuffer

        init {
            // Initialize Position Buffer
            quadPositionBuffer = ByteBuffer.allocateDirect(QUAD_COORDS.size * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
            quadPositionBuffer.put(QUAD_COORDS)
            quadPositionBuffer.position(0)

            // Initialize Standard TexCoord Buffer (Input for ARCore)
            quadTexCoordBuffer = ByteBuffer.allocateDirect(QUAD_TEX_COORDS.size * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
            quadTexCoordBuffer.put(QUAD_TEX_COORDS)
            quadTexCoordBuffer.position(0)

            // Initialize Transformed TexCoord Buffer (Output from ARCore)
            transformedTexCoordBuffer = ByteBuffer.allocateDirect(QUAD_TEX_COORDS.size * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
        }

        fun createOnGlThread() {
            // Generate External Texture
            val textures = IntArray(1)
            GLES20.glGenTextures(1, textures, 0)
            textureId = textures[0]
            val textureTarget = 36197 // GL_TEXTURE_EXTERNAL_OES
            GLES20.glBindTexture(textureTarget, textureId)
            GLES20.glTexParameteri(textureTarget, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(textureTarget, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(textureTarget, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(textureTarget, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)

            // Compile Shaders
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

            // 1. Transform UVs using ARCore to correct for rotation/aspect ratio
            quadTexCoordBuffer.position(0)
            transformedTexCoordBuffer.position(0)
            frame.transformDisplayUvCoords(quadTexCoordBuffer, transformedTexCoordBuffer)

            // 2. Set Vertex Attributes
            // Positions (Static)
            quadPositionBuffer.position(0)
            GLES20.glVertexAttribPointer(positionAttrib, 2, GLES20.GL_FLOAT, false, 0, quadPositionBuffer)

            // Texture Coords (Dynamic)
            transformedTexCoordBuffer.position(0)
            GLES20.glVertexAttribPointer(texCoordAttrib, 2, GLES20.GL_FLOAT, false, 0, transformedTexCoordBuffer)

            GLES20.glEnableVertexAttribArray(positionAttrib)
            GLES20.glEnableVertexAttribArray(texCoordAttrib)

            // 3. Bind Texture
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            GLES20.glBindTexture(36197, textureId) // GL_TEXTURE_EXTERNAL_OES
            GLES20.glUniform1i(textureUniform, 0)

            // 4. Draw
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