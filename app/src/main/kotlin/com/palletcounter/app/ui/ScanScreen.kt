package com.palletcounter.app.ui

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.ContextWrapper
import android.content.Intent
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.net.Uri
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.palletcounter.app.camera.CameraBinder
import com.palletcounter.app.camera.FrameAnalyzer
import com.palletcounter.app.data.ScanOrientation
import com.palletcounter.app.debug.CaptureCategory
import com.palletcounter.app.scan.EngineStatus
import com.palletcounter.app.scan.ScanUiState
import com.palletcounter.core.session.CountResult
import com.palletcounter.core.tracking.TrackState
import kotlinx.coroutines.delay

@Composable
fun ScanScreen(vm: AppViewModel) {
    val context = LocalContext.current
    var hasPermission by remember { mutableStateOf(context.hasCameraPermission()) }
    var denied by remember { mutableStateOf(false) }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {
        hasPermission = it
        denied = !it
    }
    LaunchedEffect(Unit) { if (!hasPermission) launcher.launch(Manifest.permission.CAMERA) }
    RecheckPermissionOnResume { hasPermission = context.hasCameraPermission() }
    BackHandler { vm.navigate(Screen.Setup) }
    if (!hasPermission) {
        Column(Modifier.fillMaxSize().padding(24.dp), verticalArrangement = Arrangement.Center) {
            Text("Camera permission is needed to scan. Frames are processed on the phone and are not stored unless you tap a capture button.")
            Button(onClick = { launcher.launch(Manifest.permission.CAMERA) }, Modifier.padding(top = 16.dp)) { Text("Grant camera permission") }
            if (denied) {
                // After repeated denials Android no longer shows the dialog.
                Text("No dialog? Allow the camera in the app settings.", Modifier.padding(top = 12.dp), color = Color.Gray)
                OutlinedButton(onClick = { context.openAppSettings() }) { Text("Open app settings") }
            }
            OutlinedButton(onClick = { vm.navigate(Screen.Setup) }) { Text("Back") }
        }
        return
    }
    LiveScan(vm)
}

@Composable
private fun LiveScan(vm: AppViewModel) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val settings = remember { vm.settings.settings.value }
    val engine = remember { vm.engine ?: vm.startLiveScan() }
    val state by engine.state.collectAsStateWithLifecycle()
    val binder = remember { CameraBinder(context) }
    val analyzer = remember { FrameAnalyzer(engine) }
    var cameraError by remember { mutableStateOf<String?>(null) }
    ScanWindowEffects(settings.scanOrientation)
    DisposableEffect(Unit) { onDispose { binder.shutdown() } }

    // The camera takes its target rotation from the display when it is bound, so bind only
    // after the display has turned to the scan orientation (frames must be upright for the
    // model, and pallets must move horizontally across the vertical count line).
    val landscape = LocalConfiguration.current.orientation == Configuration.ORIENTATION_LANDSCAPE
    val orientationReached = landscape == (settings.scanOrientation == ScanOrientation.LANDSCAPE)
    var cameraReady by remember { mutableStateOf(false) }
    LaunchedEffect(orientationReached) {
        if (cameraReady) return@LaunchedEffect
        // Some devices (large screens) ignore orientation requests: then bind anyway.
        delay(if (orientationReached) 200L else 2_000L)
        cameraReady = true
    }

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        if (cameraReady) {
            AndroidView(
                factory = { ctx ->
                    PreviewView(ctx).apply {
                        scaleType = PreviewView.ScaleType.FIT_CENTER
                        post { binder.bind(lifecycleOwner, this, analyzer, settings.analysisResolution) { cameraError = it.message } }
                    }
                },
                modifier = Modifier.fillMaxSize(),
            )
        }
        val aspect = if (state.frameHeight > 0) state.frameWidth.toFloat() / state.frameHeight else 0f
        if (aspect > 0f) {
            DetectionOverlay(
                snapshot = state.snapshot,
                config = settings.pipelineConfig(),
                imageAspect = aspect,
                options = OverlayOptions(settings.showDebugOverlay, settings.showRawDetections, settings.showTrails),
                modifier = Modifier.fillMaxSize(),
            )
        }
        Column(Modifier.align(Alignment.TopStart).fillMaxWidth()) {
            if (state.detector?.isSimulation == true) {
                WarningBanner("SIMULATION — boxes are synthetic, the camera image is ignored. Not a real count.")
            }
            when (state.status) {
                EngineStatus.LOADING -> WarningBanner("Loading detector…", Color(0xFF455A64))
                EngineStatus.ERROR -> WarningBanner("Detector error: ${state.error}")
                else -> Unit
            }
            cameraError?.let { WarningBanner("Camera error: $it") }
            val motion = state.snapshot?.motion
            if (motion?.tooFast == true) WarningBanner("MOVING TOO FAST — slow down", Palette.Filtered)
            if (settings.showDebugOverlay) Hud(state)
            state.message?.let { Text(it, color = Palette.Accent, modifier = Modifier.padding(8.dp)) }
        }
        BottomControls(vm, state, settings.showCaptureButtons, Modifier.align(Alignment.BottomCenter)) { category ->
            engine.requestCapture(category)
        }
    }
}

@Composable
private fun Hud(state: ScanUiState) {
    val s = state.snapshot
    val tracks = s?.tracks.orEmpty()
    val lost = tracks.count { it.state == TrackState.LOST }
    val tentative = tracks.count { it.state == TrackState.TENTATIVE }
    val confirmed = tracks.size - lost - tentative
    val d = state.detector
    val p = state.perf
    val m = s?.motion
    val arrow = when (m?.direction) { 1 -> "→"; -1 -> "←"; else -> "·" }
    val c = s?.stats?.counter
    val lines = listOfNotNull(
        d?.let { "${it.name}  ${it.inputWidth}×${it.inputHeight}  ${it.accelerator}" },
        "infer %.0f ms  track %.1f ms  det %.1f fps  cam %.0f fps  thermal %s%s".format(
            p.inferenceMs, p.pipelineMs, p.detectorFps, p.cameraFps, p.thermal, if (p.throttled) " (throttled)" else "",
        ),
        "raw ${s?.rawDetections?.size ?: 0}  roi ${s?.acceptedDetections?.size ?: 0}  tracks $confirmed (+$tentative new, $lost lost)",
        m?.let { "motion $arrow %.2f fw/s  step %.0f%% of pallet%s".format(kotlin.math.abs(it.vx), it.displacementRatio * 100, if (it.stationary) "  STATIONARY" else "") },
        c?.let { "net ${s.netCount}  L→R ${it.leftToRight}  R→L ${it.rightToLeft}  stitched ${it.stitches}  late ${it.lateTracks}" },
    )
    Column(Modifier.padding(6.dp).background(Color.Black.copy(alpha = 0.55f)).padding(6.dp)) {
        for (line in lines) Text(line, color = Color.White, fontSize = 12.sp, fontFamily = FontFamily.Monospace)
    }
}

@Composable
private fun BottomControls(vm: AppViewModel, state: ScanUiState, showCaptures: Boolean, modifier: Modifier, onCapture: (CaptureCategory) -> Unit) {
    val settings = vm.settings.settings.value
    val count = state.snapshot?.count ?: 0
    val result = CountResult(count, settings = settings.scanSettings)
    Column(modifier.fillMaxWidth().background(Color.Black.copy(alpha = 0.6f)).padding(8.dp)) {
        if (showCaptures) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                for (cat in CaptureCategory.entries) {
                    FilledTonalButton(onClick = { onCapture(cat) }, Modifier.weight(1f)) { Text(cat.label, fontSize = 11.sp, maxLines = 1) }
                }
            }
        }
        Row(Modifier.fillMaxWidth().padding(top = 6.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("$count EUR", fontSize = 44.sp, fontWeight = FontWeight.Bold, color = Palette.Accent)
                Text("= ${result.formula} half-pallets", color = Color.White, style = MaterialTheme.typography.bodySmall)
            }
            val finishing = vm.finishing
            OutlinedButton(onClick = { vm.engine?.resetCount() }, enabled = !finishing) { Text("RESET") }
            Button(onClick = { vm.finishLiveScan() }, Modifier.padding(start = 8.dp).height(64.dp), enabled = !finishing) {
                Text("FINISH", fontSize = 20.sp, fontWeight = FontWeight.Bold)
            }
        }
    }
}

/**
 * Keeps the screen on and fixes the orientation while scanning. The orientation is requested
 * explicitly (not "locked" to whatever the display shows) so a phone held in landscape also
 * scans in landscape when system auto-rotate is off. A fixed orientation also means no
 * rotation mid-scan, which would swap the frame axes under the tracker.
 */
@Composable
fun ScanWindowEffects(orientation: ScanOrientation) {
    val context = LocalContext.current
    val view = LocalView.current
    DisposableEffect(orientation) {
        val activity = context.findActivity()
        val previous = activity?.requestedOrientation
        activity?.requestedOrientation = when (orientation) {
            ScanOrientation.LANDSCAPE -> ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
            ScanOrientation.PORTRAIT -> ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        }
        view.keepScreenOn = true
        onDispose {
            view.keepScreenOn = false
            activity?.requestedOrientation = previous ?: ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED
        }
    }
}

fun Context.hasCameraPermission(): Boolean =
    ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

/** Opens this app's page in the system settings (to allow a permanently denied permission). */
fun Context.openAppSettings() {
    val intent = Intent(android.provider.Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", packageName, null))
        .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
    runCatching { startActivity(intent) }
}

/** Runs [check] whenever the screen resumes, e.g. after returning from the system settings. */
@Composable
fun RecheckPermissionOnResume(check: () -> Unit) {
    val lifecycleOwner = LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event -> if (event == Lifecycle.Event.ON_RESUME) check() }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }
}

fun Context.findActivity(): Activity? {
    var c: Context? = this
    while (c is ContextWrapper) {
        if (c is Activity) return c
        c = c.baseContext
    }
    return null
}
