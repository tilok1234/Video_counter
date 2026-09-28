package com.palletcounter.app.ui

import android.Manifest
import android.content.ActivityNotFoundException
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
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
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.palletcounter.app.data.DetectorMode
import com.palletcounter.app.detection.FrameInput
import com.palletcounter.core.detection.Detection
import com.palletcounter.core.session.CountResult
import com.palletcounter.core.session.LineMode
import com.palletcounter.core.session.StackSize
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Static photo mode: detect visible pallet bases in one image, number them left to right,
 * let the user exclude wrong boxes (tap) and correct the count (+/-).
 * Less reliable than a sweep (occlusion, distance, perspective) — by design secondary.
 */
@Composable
fun PhotoScreen(vm: AppViewModel) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val settings by vm.settings.settings.collectAsStateWithLifecycle()
    var image by remember { mutableStateOf<Bitmap?>(null) }
    var detections by remember { mutableStateOf<List<Detection>>(emptyList()) }
    var excluded by remember { mutableStateOf(setOf<Int>()) }
    var threshold by remember { mutableStateOf(settings.photoConfidence) }
    var adjustment by remember { mutableStateOf(0) }
    var stack by remember { mutableStateOf(settings.stackSize) }
    var lineMode by remember { mutableStateOf(settings.lineMode) }
    var busy by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }
    var detectorName by remember { mutableStateOf("?") }
    BackHandler { vm.navigate(Screen.Setup) }

    val photoFile = remember { File(context.cacheDir, "photos").apply { mkdirs() }.resolve("capture.jpg") }
    val photoUri = remember { FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", photoFile) }

    fun analyze(uri: Uri) {
        busy = true
        error = null
        scope.launch {
            val outcome = withContext(Dispatchers.Default) {
                runCatching {
                    val bmp = try {
                        decodeUpright(context, uri, maxSide = 2048)
                    } finally {
                        // The camera photo is only needed in memory; do not keep it on disk.
                        if (uri == photoUri) photoFile.delete()
                    }
                    check(settings.detectorMode != DetectorMode.SIMULATION) { "Photo mode needs a trained model (the simulation detector has nothing to detect)." }
                    // Create, run and close on this thread (GPU delegates are thread-bound).
                    val detector = vm.models.createDetector(settings, decoderThreshold = 0.1f)
                    try {
                        val frame = detector.detect(FrameInput(bmp, bmp.width, bmp.height, 0, 0L), 0L)
                        Triple(bmp, frame.detections, detector.info.name)
                    } finally {
                        detector.close()
                    }
                }
            }
            busy = false
            outcome.onSuccess { (bmp, dets, name) ->
                image = bmp
                detections = dets.sortedBy { it.box.centerX }
                excluded = emptySet()
                adjustment = 0
                detectorName = name
            }.onFailure { error = it.message ?: it.toString() }
        }
    }

    val takePicture = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        if (ok) {
            analyze(photoUri)
        } else {
            photoFile.delete()
        }
    }
    fun launchCamera() {
        try {
            takePicture.launch(photoUri)
        } catch (e: ActivityNotFoundException) {
            error = "No camera app found. Use 'Choose image'."
        } catch (e: SecurityException) {
            error = "Camera permission is needed to take a photo."
        }
    }
    // The system camera app may only be started once this app holds the camera permission
    // (it declares CAMERA), otherwise Android throws a SecurityException.
    val cameraPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) {
            launchCamera()
        } else {
            error = "Camera permission denied. Allow it in the app settings or use 'Choose image'."
        }
    }
    DisposableEffect(Unit) { onDispose { photoFile.delete() } }
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { it?.let(::analyze) }

    val visible = detections.withIndex().filter { it.value.confidence >= threshold }
    val counted = visible.count { it.index !in excluded }
    val result = CountResult(counted, adjustment, settings.scanSettings.copy(stackSize = stack, lineMode = lineMode))

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp)) {
        Text("Photo mode", fontSize = 22.sp, fontWeight = FontWeight.Bold)
        Text("Static photos are less reliable than a video sweep. Tap a box to exclude it.", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
        Row(Modifier.fillMaxWidth().padding(vertical = 8.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = { if (context.hasCameraPermission()) launchCamera() else cameraPermission.launch(Manifest.permission.CAMERA) },
                enabled = !busy,
                modifier = Modifier.weight(1f),
            ) { Text("Take photo") }
            OutlinedButton(onClick = { pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) }, enabled = !busy, modifier = Modifier.weight(1f)) { Text("Choose image") }
        }
        if (busy) Text("Detecting…", color = Palette.Accent)
        error?.let { WarningBanner(it) }
        image?.let { bmp ->
            val aspect = bmp.width.toFloat() / bmp.height
            Box(Modifier.fillMaxWidth().aspectRatio(aspect)) {
                Image(bmp.asImageBitmap(), contentDescription = "photo", Modifier.fillMaxSize(), contentScale = ContentScale.Fit)
                val paint = remember { android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply { textSize = 42f; isFakeBoldText = true } }
                Canvas(
                    Modifier.fillMaxSize().pointerInput(visible, excluded) {
                        detectTapGestures { pos ->
                            val u = pos.x / size.width
                            val v = pos.y / size.height
                            visible.lastOrNull { it.value.box.contains(u, v) }?.let { hit ->
                                excluded = if (hit.index in excluded) excluded - hit.index else excluded + hit.index
                            }
                        }
                    },
                ) {
                    var number = 0
                    for ((index, d) in visible) {
                        val isExcluded = index in excluded
                        if (!isExcluded) number++
                        val color = if (isExcluded) Palette.Danger else Palette.Counted
                        val tl = Offset(d.box.left * size.width, d.box.top * size.height)
                        drawRect(color, tl, Size(d.box.width * size.width, d.box.height * size.height), style = Stroke(5f))
                        paint.color = android.graphics.Color.argb(255, (color.red * 255).toInt(), (color.green * 255).toInt(), (color.blue * 255).toInt())
                        val label = if (isExcluded) "✗" else "$number"
                        drawIntoCanvas { it.nativeCanvas.drawText("$label  %.2f".format(d.confidence), tl.x + 6f, tl.y - 8f, paint) }
                    }
                }
            }
            Text("Model: $detectorName", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
            SliderRow("Confidence threshold", threshold, 0.1f..0.9f, { "%.2f".format(it) }, { threshold = it })
        }
        Row(Modifier.fillMaxWidth().padding(top = 8.dp), horizontalArrangement = Arrangement.SpaceEvenly) {
            BigStat("EURO pallets", "${result.pallets}", valueColor = Palette.Accent)
            BigStat("Half-pallets", "${result.halfPallets}")
        }
        Text(result.formula, color = Color.LightGray, modifier = Modifier.padding(bottom = 8.dp))
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { adjustment = result.adjusted(-1).manualAdjustment }, Modifier.weight(1f)) { Text("−1 pallet") }
            Button(onClick = { adjustment = result.adjusted(+1).manualAdjustment }, Modifier.weight(1f)) { Text("+1 pallet") }
        }
        SectionTitle("Stack size")
        ChoiceRow(StackSize.entries, stack, { "${it.halfPallets}" }, { stack = it })
        SectionTitle("Line mode")
        ChoiceRow(LineMode.entries, lineMode, { it.label }, { lineMode = it })
        Button(
            onClick = { vm.accept(result, "photo", detectorName, simulated = false) },
            enabled = image != null,
            modifier = Modifier.fillMaxWidth().padding(top = 16.dp),
        ) { Text("Accept") }
    }
}

/** Decodes an image upright (EXIF orientation applied) into a software bitmap. */
fun decodeUpright(context: Context, uri: Uri, maxSide: Int): Bitmap {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
        val source = ImageDecoder.createSource(context.contentResolver, uri)
        return ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE // hardware bitmaps cannot be drawn to a software canvas
            val longest = maxOf(info.size.width, info.size.height)
            if (longest > maxSide) {
                val s = maxSide.toFloat() / longest
                decoder.setTargetSize((info.size.width * s).toInt(), (info.size.height * s).toInt())
            }
        }
    }
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    context.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
    var sample = 1
    while (maxOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxSide) sample *= 2
    val bmp = context.contentResolver.openInputStream(uri)?.use {
        BitmapFactory.decodeStream(it, null, BitmapFactory.Options().apply { inSampleSize = sample })
    } ?: error("cannot decode image")
    val orientation = context.contentResolver.openInputStream(uri)?.use { ExifInterface(it).rotationDegrees } ?: 0
    if (orientation == 0) return bmp
    val m = android.graphics.Matrix().apply { postRotate(orientation.toFloat()) }
    return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
}
