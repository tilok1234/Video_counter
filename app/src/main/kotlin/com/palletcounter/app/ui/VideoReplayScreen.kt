package com.palletcounter.app.ui

import android.graphics.Bitmap
import android.net.Uri
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.palletcounter.app.scan.ScanEngine
import com.palletcounter.app.scan.ScanUiState
import com.palletcounter.app.video.VideoFrameSource
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Runs a recorded video through exactly the same detector → tracker → counter pipeline as
 * the live camera. The main tool for reproducible testing without walking the yard.
 */
@Composable
fun VideoReplayScreen(vm: AppViewModel) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var uri by remember { mutableStateOf<Uri?>(null) }
    var fps by remember { mutableStateOf(10) }
    var rotation by remember { mutableStateOf(0) }
    var expected by remember { mutableStateOf("") }
    var progress by remember { mutableStateOf(0f) }
    var running by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }
    var job by remember { mutableStateOf<Job?>(null) }
    val preview = remember { MutableStateFlow<Bitmap?>(null) }
    val currentFrame by preview.collectAsStateWithLifecycle()
    var engine by remember { mutableStateOf<ScanEngine?>(null) }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri = it }
    val settings by vm.settings.settings.collectAsStateWithLifecycle()
    DisposableEffect(Unit) { onDispose { job?.cancel(); engine?.close() } }
    BackHandler { job?.cancel(); vm.navigate(Screen.Setup) }

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp)) {
        Text("Replay a recorded video", fontSize = 22.sp, fontWeight = FontWeight.Bold)
        Text(
            "Frames are decoded at the chosen rate and processed with the current model and settings. " +
                "Enter the true count to compare.",
            style = MaterialTheme.typography.bodySmall,
            color = Color.Gray,
        )
        OutlinedButton(onClick = { picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly)) }, enabled = !running) {
            Text(if (uri == null) "Choose video" else "Video selected — change")
        }
        SectionTitle("Detector frames per second")
        ChoiceRow(listOf(5, 10, 15), fps, { "$it" }, { fps = it })
        SectionTitle("Rotate footage (clockwise)")
        ChoiceRow(listOf(0, 90, 180, 270), rotation, { "$it°" }, { rotation = it })
        OutlinedTextField(
            value = expected,
            onValueChange = { expected = it.filter(Char::isDigit).take(4) },
            label = { Text("Expected pallet count (optional)") },
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
            modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
        )
        error?.let { WarningBanner(it) }
        Button(
            enabled = uri != null && !running,
            modifier = Modifier.fillMaxWidth().padding(top = 12.dp),
            onClick = {
                val videoUri = uri ?: return@Button
                error = null
                running = true
                val e = ScanEngine(
                    context = context.applicationContext,
                    models = vm.models,
                    settings = settings,
                    source = "video",
                    logStore = vm.logs,
                    captures = vm.captures,
                    expectedCount = expected.toIntOrNull(),
                    videoName = videoUri.lastPathSegment,
                )
                engine = e
                job = scope.launch {
                    val outcome = withContext(Dispatchers.Default) {
                        runCatching {
                            e.start()
                            check(e.awaitReady()) { e.state.value.error ?: "detector failed to load" }
                            VideoFrameSource(context, videoUri, fps.toDouble(), rotation).use { src ->
                                val total = src.frameCount
                                var i = 0
                                while (isActive) {
                                    val frame = src.frame(i) ?: break
                                    e.processBlocking(frame)
                                    if (i % 3 == 0) preview.value = frame.uprightCrop(maxSide = 720)
                                    i++
                                    progress = (i.toFloat() / total).coerceAtMost(1f)
                                }
                            }
                            e.finish()
                        }
                    }
                    running = false
                    e.close()
                    engine = null
                    outcome.onSuccess { vm.showReview(it) }.onFailure { error = it.message ?: it.toString() }
                }
            },
        ) { Text(if (running) "Processing…" else "Start replay") }
        if (running) {
            LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp))
            OutlinedButton(onClick = { job?.cancel(); running = false }) { Text("Cancel") }
        }
        val state: ScanUiState? = engine?.state?.collectAsStateWithLifecycle()?.value
        Text("Count: ${state?.snapshot?.count ?: 0}", fontSize = 28.sp, fontWeight = FontWeight.Bold, color = Palette.Accent)
        currentFrame?.let { bmp ->
            Box(Modifier.fillMaxWidth().aspectRatio(bmp.width.toFloat() / bmp.height)) {
                Image(bmp.asImageBitmap(), contentDescription = "video frame", Modifier.fillMaxSize(), contentScale = ContentScale.Fit)
                DetectionOverlay(
                    snapshot = state?.snapshot,
                    config = settings.pipelineConfig(),
                    imageAspect = bmp.width.toFloat() / bmp.height,
                    options = OverlayOptions(settings.showDebugOverlay, settings.showRawDetections, settings.showTrails),
                    modifier = Modifier.fillMaxSize(),
                )
            }
        }
    }
}
