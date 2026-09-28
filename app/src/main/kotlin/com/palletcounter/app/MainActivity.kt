package com.palletcounter.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import com.palletcounter.core.session.CountResult
import org.tensorflow.lite.Interpreter

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Build-configuration smoke test: touches core, CameraX and LiteRT symbols.
        @Suppress("UNUSED_VARIABLE") val types = listOf(ProcessCameraProvider::class.java, Interpreter::class.java)
        setContent { MaterialTheme { Text("Pallet counter: ${CountResult(16).formula}") } }
    }
}
