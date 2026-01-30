package com.example.eyebro

import android.content.Context
import android.content.SharedPreferences
import android.os.Bundle
import android.widget.Button
import android.widget.RadioButton
import android.widget.RadioGroup
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.materialswitch.MaterialSwitch

class SettingsActivity : AppCompatActivity() {

    private lateinit var prefs: SharedPreferences

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        prefs = getSharedPreferences("EyebroPrefs", Context.MODE_PRIVATE)

        val switchObj = findViewById<MaterialSwitch>(R.id.switchObjectDetection)
        val switchVib = findViewById<MaterialSwitch>(R.id.switchVibration)
        val switchTTS = findViewById<MaterialSwitch>(R.id.switchTTS)
        val rgModel = findViewById<RadioGroup>(R.id.radioGroupModel)
        val rbMlKit = findViewById<RadioButton>(R.id.rbMlKit)
        val rbYolo = findViewById<RadioButton>(R.id.rbYolo)
        val btnBack = findViewById<Button>(R.id.btnBack)

        // --- Load saved state ---
        switchObj.isChecked = prefs.getBoolean("enable_obj", true)
        switchVib.isChecked = prefs.getBoolean("enable_vib", true)
        switchTTS.isChecked = prefs.getBoolean("enable_tts", true)

        // Load Model Preference (Default to "mlkit")
        val currentModel = prefs.getString("model_type", "mlkit")
        if (currentModel == "yolo") {
            rbYolo.isChecked = true
        } else {
            rbMlKit.isChecked = true
        }

        // --- Listeners ---

        // Model Selection Logic
        rgModel.setOnCheckedChangeListener { _, checkedId ->
            val modelType = if (checkedId == R.id.rbYolo) "yolo" else "mlkit"
            prefs.edit().putString("model_type", modelType).apply()
        }

        switchObj.setOnCheckedChangeListener { _, isChecked ->
            prefs.edit().putBoolean("enable_obj", isChecked).apply()
        }
        switchVib.setOnCheckedChangeListener { _, isChecked ->
            prefs.edit().putBoolean("enable_vib", isChecked).apply()
        }
        switchTTS.setOnCheckedChangeListener { _, isChecked ->
            prefs.edit().putBoolean("enable_tts", isChecked).apply()
        }

        btnBack.setOnClickListener { finish() }
    }
}