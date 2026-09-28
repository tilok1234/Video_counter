package com.palletcounter.app.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

object Palette {
    val Background = Color(0xFF101214)
    val Surface = Color(0xFF1C1F23)
    val Accent = Color(0xFFFFB300) // wood/amber
    val Counted = Color(0xFF4CAF50)
    val Confirmed = Color(0xFF29B6F6)
    val Tentative = Color(0xFFFFEE58)
    val Lost = Color(0xFF9E9E9E)
    val Filtered = Color(0xFFFF7043)
    val Danger = Color(0xFFE53935)
}

@Composable
fun PalletTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(
            primary = Palette.Accent,
            onPrimary = Color.Black,
            secondary = Palette.Confirmed,
            background = Palette.Background,
            surface = Palette.Surface,
            error = Palette.Danger,
        ),
        content = content,
    )
}
