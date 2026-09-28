package com.palletcounter.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.palletcounter.app.data.DetectorMode
import com.palletcounter.core.session.LineMode
import com.palletcounter.core.session.StackSize
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale

@Composable
fun SetupScreen(vm: AppViewModel) {
    val settings by vm.settings.settings.collectAsStateWithLifecycle()
    val history by vm.history.entries.collectAsStateWithLifecycle()
    val simulation = settings.detectorMode == DetectorMode.SIMULATION
    val noModel = vm.modelDescription == null && !simulation

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp)) {
        Text("EUR-pallet counter", fontSize = 26.sp, fontWeight = FontWeight.Bold)
        Text(
            if (simulation) "Detector: SIMULATION (plumbing test, not real detection)" else "Model: ${vm.modelStatus}",
            color = if (simulation || noModel) Palette.Danger else Color.LightGray,
            style = MaterialTheme.typography.bodySmall,
        )
        if (noModel) {
            Spacer(Modifier.height(8.dp))
            WarningBanner("No trained model installed. Import one in Settings (see docs/MODEL_TRAINING.md) or switch the detector to SIMULATION to test the app.")
        }

        SectionTitle("Stack size")
        ChoiceRow(StackSize.entries, settings.stackSize, { "${it.halfPallets}" }, { v -> vm.settings.update { it.copy(stackSize = v) } })

        SectionTitle("Line mode")
        ChoiceRow(LineMode.entries, settings.lineMode, { it.label }, { v -> vm.settings.update { it.copy(lineMode = v) } })

        Spacer(Modifier.height(20.dp))
        Button(
            onClick = { vm.navigate(Screen.Scan) },
            modifier = Modifier.fillMaxWidth().height(72.dp),
            enabled = !noModel,
        ) { Text("START VIDEO SWEEP", fontSize = 22.sp, fontWeight = FontWeight.Bold) }
        Text(
            "The scan screen turns to ${settings.scanOrientation.label.substringBefore(" (").lowercase()}: hold the phone the same way. " +
                "Start before the first pallet, walk steadily along the line with the pallet bases inside the band, " +
                "and finish after the last pallet has crossed the yellow line.",
            style = MaterialTheme.typography.bodySmall,
            color = Color.Gray,
            modifier = Modifier.padding(top = 6.dp),
        )
        Spacer(Modifier.height(12.dp))
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = { vm.navigate(Screen.VideoReplay) }, Modifier.weight(1f)) { Text("Replay video") }
            OutlinedButton(onClick = { vm.navigate(Screen.Photo) }, Modifier.weight(1f)) { Text("Photo mode") }
            OutlinedButton(onClick = { vm.navigate(Screen.Settings) }, Modifier.weight(1f)) { Text("Settings") }
        }

        val today = history.filter { isToday(it.timestampMillis) && !it.simulated }
        SectionTitle("Today")
        Text("${today.size} line(s), ${today.sumOf { it.result.palletsInLine }} EUR-pallets, ${today.sumOf { it.result.halfPallets }} half-pallets")
        if (history.isNotEmpty()) {
            SectionTitle("Recent counts")
            val fmt = SimpleDateFormat("dd.MM HH:mm", Locale.getDefault())
            for (entry in history.take(15)) {
                Card(Modifier.fillMaxWidth().padding(vertical = 3.dp)) {
                    Row(Modifier.padding(10.dp)) {
                        Column(Modifier.weight(1f)) {
                            Text("${entry.result.halfPallets} half-pallets", fontWeight = FontWeight.Bold)
                            Text(
                                "${fmt.format(Date(entry.timestampMillis))} · ${entry.result.formula} · ${entry.result.settings.lineMode.label} · ${entry.source}" +
                                    if (entry.simulated) " · SIMULATED" else "",
                                style = MaterialTheme.typography.bodySmall,
                                color = if (entry.simulated) Palette.Danger else Color.LightGray,
                            )
                        }
                        TextButton(onClick = { vm.history.remove(entry) }) { Text("Delete") }
                    }
                }
            }
        }
    }
}

private fun isToday(millis: Long): Boolean {
    val a = Calendar.getInstance().apply { timeInMillis = millis }
    val b = Calendar.getInstance()
    return a.get(Calendar.YEAR) == b.get(Calendar.YEAR) && a.get(Calendar.DAY_OF_YEAR) == b.get(Calendar.DAY_OF_YEAR)
}
