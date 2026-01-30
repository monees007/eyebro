package com.example.eyebro

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.Image
import android.opengl.GLSurfaceView
import android.os.Build
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Frame
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.CameraNotAvailableException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class MainActivity : AppCompatActivity(), GLSurfaceView.Renderer {

    private lateinit var surfaceView: GLSurfaceView
    private lateinit var warningText: TextView
    private lateinit var statusText: TextView

    private var session: Session? = null
    private var isDepthSupported = false

    // Safety Thresholds
    private val OBSTACLE_LIMIT_MM = 1200 // Alert if object is within 1.2 meters
    private val VIBRATION_interval = 500L // Don't vibrate constantly
    private var lastVibrationTime = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        surfaceView = findViewById(R.id.surfaceView)
        warningText = findViewById(R.id.warningText)
        statusText = findViewById(R.id.statusText)

        // Setup OpenGL Renderer for ARCore (Required to keep session alive)
        surfaceView.preserveEGLContextOnPause = true
        surfaceView.setEGLContextClientVersion(2)
        surfaceView.setEGLConfigChooser(8, 8, 8, 8, 16, 0)
        surfaceView.setRenderer(this)
        surfaceView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY

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

    override fun onDrawFrame(gl: GL10?) {
        // Clear screen (standard GL stuff)
        gl?.glClear(GL10.GL_COLOR_BUFFER_BIT or GL10.GL_DEPTH_BUFFER_BIT)

        val session = session ?: return

        try {
            // Get the current frame
            val frame = session.update()

            // Just for the hackathon: We won't render the camera texture to GL
            // to save code lines, but in a real app you connect the texture here.
            // We focus purely on the DEPTH LOGIC.

            if (isDepthSupported) {
                processDepth(frame)
            }

        } catch (t: Throwable) {
            // Handle errors
        }
    }

    private fun processDepth(frame: Frame) {
        try {
            // Acquire the raw depth image (16-bit)
            // Pixel value = distance in millimeters
            val depthImage = frame.acquireDepthImage16Bits()

            if (depthImage.planes.isEmpty()) {
                depthImage.close()
                return
            }

            // Access the buffer
            val buffer = depthImage.planes[0].buffer.order(ByteOrder.nativeOrder())

            // Scan the center of the screen for obstacles
            val width = depthImage.width
            val height = depthImage.height
            val scanWidth = width / 4 // Check middle 25% width
            val scanHeight = height / 4 // Check middle 25% height
            val startX = (width - scanWidth) / 2
            val startY = (height - scanHeight) / 2

            var closePixels = 0
            val totalPixels = scanWidth * scanHeight

            // Stride allows skipping pixels for performance (check every 10th pixel)
            val step = 4

            for (y in startY until startY + scanHeight step step) {
                for (x in startX until startX + scanWidth step step) {
                    // Calculate buffer index
                    val index = (y * depthImage.planes[0].rowStride) + (x * depthImage.planes[0].pixelStride)

                    // Get 16-bit depth value (Little Endian)
                    val lowByte = buffer.get(index).toInt() and 0xFF
                    val highByte = buffer.get(index + 1).toInt() and 0xFF
                    val distanceMm = (highByte shl 8) or lowByte

                    // Valid depth is usually > 0. 0 means "unknown".
                    if (distanceMm in 1 until OBSTACLE_LIMIT_MM) {
                        closePixels++
                    }
                }
            }

            // Release image immediately to avoid memory leaks
            depthImage.close()

            // Heuristic: If > 20% of center pixels are close, trigger alert
            val scannedPixels = (scanWidth / step) * (scanHeight / step)
            val threshold = scannedPixels * 0.20

            runOnUiThread {
                if (closePixels > threshold) {
                    triggerObstacleWarning()
                } else {
                    clearWarning()
                }
            }

        } catch (e: Exception) {
            // Depth image not available yet
        }
    }

    private fun triggerObstacleWarning() {
        warningText.visibility = View.VISIBLE
        warningText.text = "STOP!"

        val currentTime = System.currentTimeMillis()
        if (currentTime - lastVibrationTime > VIBRATION_interval) {
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

    // Required GL Renderer stubs
    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {}
    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {}
}
