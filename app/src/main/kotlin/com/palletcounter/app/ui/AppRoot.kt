package com.palletcounter.app.ui

import android.widget.Toast
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext

@Composable
fun AppRoot(vm: AppViewModel) {
    val context = LocalContext.current
    LaunchedEffect(vm.toast) {
        vm.toast?.let {
            Toast.makeText(context, it, Toast.LENGTH_LONG).show()
            vm.toast = null
        }
    }
    PalletTheme {
        Surface(Modifier.fillMaxSize()) {
            when (val screen = vm.screen) {
                Screen.Setup -> SetupScreen(vm)
                Screen.Scan -> ScanScreen(vm)
                is Screen.Review -> ReviewScreen(vm, screen.result)
                Screen.Photo -> PhotoScreen(vm)
                Screen.VideoReplay -> VideoReplayScreen(vm)
                Screen.Settings -> SettingsScreen(vm)
            }
        }
    }
}
