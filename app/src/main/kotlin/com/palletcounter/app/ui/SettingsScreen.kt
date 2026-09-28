package com.palletcounter.app.ui

import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.palletcounter.app.data.Accelerator
import com.palletcounter.app.data.AnalysisResolution
import com.palletcounter.app.data.DetectorMode
import com.palletcounter.app.data.RoiPreset

@Composable
fun SettingsScreen(vm: AppViewModel) {
    val context = LocalContext.current
    val s by vm.settings.settings.collectAsStateWithLifecycle()
    var refresh by remember { mutableIntStateOf(0) }
    BackHandler { vm.navigate(Screen.Setup) }
    val importModel = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { it?.let(vm::importModel) }
    val importSidecar = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { it?.let(vm::importSidecar) }
    val exportLogs = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/zip")) { uri ->
        uri?.let { context.contentResolver.openOutputStream(it)?.use(vm.logs::exportZip); vm.toast = "Logs exported" }
    }
    val exportCaptures = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/zip")) { uri ->
        uri?.let { context.contentResolver.openOutputStream(it)?.use(vm.captures::exportZip); vm.toast = "Captures exported" }
    }
    fun set(transform: (com.palletcounter.app.data.AppSettings) -> com.palletcounter.app.data.AppSettings) = vm.settings.update(transform)

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp)) {
        Row(Modifier.fillMaxWidth()) {
            Text("Settings", fontSize = 22.sp, fontWeight = FontWeight.Bold, modifier = Modifier.weight(1f))
            OutlinedButton(onClick = { vm.navigate(Screen.Setup) }) { Text("Done") }
        }

        SectionTitle("Detector")
        ChoiceRow(DetectorMode.entries, s.detectorMode, { if (it == DetectorMode.MODEL) "Trained model" else "SIMULATION" }, { v -> set { it.copy(detectorMode = v) } })
        if (s.detectorMode == DetectorMode.SIMULATION) {
            WarningBanner("Simulation ignores the camera and produces synthetic boxes. Use it only to test the app's plumbing.")
        }
        val d = vm.modelDescription
        Text(
            if (d == null) "No model installed" else buildString {
                append("${d.sidecar.name} ${d.sidecar.version}\n")
                append("${d.source.displayName}, ${d.sizeBytes / 1024} KB, sha256 ${d.sha256Prefix}\n")
                append("format ${d.sidecar.outputFormat}, coordinates ${d.sidecar.coordinates}, classes ${d.sidecar.classes}\n")
                d.sidecar.quantization?.let { append("precision $it  ") }
                d.sidecar.framework?.let { append(it) }
            },
            style = MaterialTheme.typography.bodySmall,
            color = Color.LightGray,
        )
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { importModel.launch(arrayOf("*/*")) }, Modifier.weight(1f)) { Text("Import .tflite") }
            OutlinedButton(onClick = { importSidecar.launch(arrayOf("application/json", "*/*")) }, Modifier.weight(1f)) { Text("Import .json") }
        }
        OutlinedButton(onClick = vm::removeImportedModel) { Text("Remove imported model") }

        SectionTitle("Performance")
        ChoiceRow(Accelerator.entries, s.accelerator, { it.name }, { v -> set { it.copy(accelerator = v) } })
        ChoiceRow(listOf(2, 4, 6), s.cpuThreads, { "$it CPU threads" }, { v -> set { it.copy(cpuThreads = v) } }, Modifier.padding(top = 6.dp))
        SliderRow("Max detector rate", s.maxInferenceFps.toFloat(), 2f..30f, { "${it.toInt()} fps" }, { v -> set { it.copy(maxInferenceFps = v.toInt()) } })
        ChoiceRow(AnalysisResolution.entries, s.analysisResolution, { it.label }, { v -> set { it.copy(analysisResolution = v) } })

        SectionTitle("Guide band (region of interest)")
        ChoiceRow(RoiPreset.entries.take(2), s.roiPreset, { it.label }, { v -> set { it.copy(roiPreset = v) } })
        ChoiceRow(RoiPreset.entries.drop(2), s.roiPreset, { it.label }, { v -> set { it.copy(roiPreset = v) } }, Modifier.padding(top = 6.dp))
        if (s.roiPreset == RoiPreset.CUSTOM) {
            SliderRow("Band top", s.customRoiTop, 0f..0.9f, { "%.2f".format(it) }, { v -> set { it.copy(customRoiTop = v) } })
            SliderRow("Band bottom", s.customRoiBottom, 0.1f..1f, { "%.2f".format(it) }, { v -> set { it.copy(customRoiBottom = v) } })
        }
        Text(
            "Stacked in several tiers? Use 'Full height' so every visible wooden layer is counted.",
            style = MaterialTheme.typography.bodySmall,
            color = Color.Gray,
        )

        SectionTitle("Counting")
        SliderRow("Count line position", s.lineX, 0.25f..0.75f, { "%.2f".format(it) }, { v -> set { it.copy(lineX = v) } })
        SliderRow("Dead band (±)", s.hysteresis, 0.01f..0.1f, { "%.3f".format(it) }, { v -> set { it.copy(hysteresis = v) } })
        SliderRow("Detections before a track counts", s.minHits.toFloat(), 1f..6f, { "${it.toInt()}" }, { v -> set { it.copy(minHits = v.toInt()) } }, steps = 4)
        SliderRow("High score threshold", s.highThreshold, 0.2f..0.9f, { "%.2f".format(it) }, { v -> set { it.copy(highThreshold = v) } })
        SliderRow("New track threshold", s.newTrackThreshold, 0.2f..0.95f, { "%.2f".format(it) }, { v -> set { it.copy(newTrackThreshold = v) } })
        SliderRow("Low score threshold", s.lowThreshold, 0.05f..0.5f, { "%.2f".format(it) }, { v -> set { it.copy(lowThreshold = v) } })
        SliderRow("Ignore boxes smaller than (× typical)", s.sizeFilter, 0f..0.9f, { if (it < 0.05f) "off" else "%.2f".format(it) }, { v -> set { it.copy(sizeFilter = if (v < 0.05f) 0f else v) } })

        SectionTitle("Debug")
        SwitchRow("Debug overlay (IDs, scores, HUD)", s.showDebugOverlay, { v -> set { it.copy(showDebugOverlay = v) } })
        SwitchRow("Show raw detections", s.showRawDetections, { v -> set { it.copy(showRawDetections = v) } })
        SwitchRow("Show track trails", s.showTrails, { v -> set { it.copy(showTrails = v) } })
        SwitchRow("Sample capture buttons", s.showCaptureButtons, { v -> set { it.copy(showCaptureButtons = v) } }, "MISSED / FALSE + / BAD TRACK / GOOD save the current frame on the phone")
        SwitchRow("Record detection logs", s.recordDetectionLogs, { v -> set { it.copy(recordDetectionLogs = v) } }, "Boxes and scores only (no images), for replay on a PC")

        SectionTitle("Data on this phone")
        val counts = remember(refresh) { vm.captures.counts() }
        val logCount = remember(refresh) { vm.logs.list().size }
        Text(
            "Captures: " + counts.entries.joinToString { "${it.key.label} ${it.value}" } +
                " (${vm.captures.totalBytes() / 1024} KB)\nDetection logs: $logCount (${vm.logs.totalBytes() / 1024} KB)\n" +
                "Nothing is uploaded. Export only when you choose to.",
            style = MaterialTheme.typography.bodySmall,
            color = Color.LightGray,
        )
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = { exportCaptures.launch("pallet_captures.zip") }, Modifier.weight(1f)) { Text("Export captures") }
            OutlinedButton(onClick = { exportLogs.launch("pallet_scan_logs.zip") }, Modifier.weight(1f)) { Text("Export logs") }
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            DangerButton("Delete captures", { vm.captures.deleteAll(); refresh++ }, Modifier.weight(1f))
            DangerButton("Delete logs", { vm.logs.deleteAll(); refresh++ }, Modifier.weight(1f))
        }
        DangerButton("Clear history", { vm.history.clear() }, Modifier.padding(top = 8.dp))
        OutlinedButton(onClick = { vm.settings.resetToDefaults() }, Modifier.padding(top = 8.dp)) { Text("Reset settings to defaults") }
    }
}
