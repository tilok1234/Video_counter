package com.palletcounter.app.debug

import android.content.Context
import android.graphics.Bitmap
import com.palletcounter.core.detection.DetectorInfo
import com.palletcounter.core.pipeline.ScanSnapshot
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/** Why a frame was saved; becomes the sub-folder name. */
enum class CaptureCategory(val label: String, val folder: String) {
    MISSED_PALLET("MISSED", "missed_pallet"),
    FALSE_POSITIVE("FALSE +", "false_positive"),
    BAD_TRACK("BAD TRACK", "bad_track"),
    GOOD_EXAMPLE("GOOD", "good_example"),
}

@Serializable
private data class CaptureMeta(
    val category: String,
    val capturedAt: String,
    val frameWidth: Int,
    val frameHeight: Int,
    val detector: DetectorInfo?,
    val snapshot: ScanSnapshot,
)

/**
 * Saves developer samples (frame + detections) to app-private storage. Nothing leaves the
 * phone unless the user exports a ZIP through the system file picker.
 *
 * Each sample: `<category>/<time>.jpg`, a `.json` with detections/tracks/config, and a
 * YOLO `.txt` pre-label (current confident detections) to speed up annotation.
 */
class CaptureStore(context: Context) {
    val root = File(context.filesDir, "captures")
    private val json = Json { encodeDefaults = true }

    fun save(category: CaptureCategory, uprightFrame: Bitmap, snapshot: ScanSnapshot, detector: DetectorInfo?): File {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.ROOT).format(Date())
        val dir = File(root, category.folder).apply { mkdirs() }
        val base = "sample_$stamp"
        File(dir, "$base.jpg").outputStream().use { uprightFrame.compress(Bitmap.CompressFormat.JPEG, 92, it) }
        val meta = CaptureMeta(category.name, stamp, uprightFrame.width, uprightFrame.height, detector, snapshot.copy(tracks = snapshot.tracks.map { it.copy(trail = emptyList()) }))
        File(dir, "$base.json").writeText(json.encodeToString(CaptureMeta.serializer(), meta))
        val labels = snapshot.acceptedDetections.filter { it.confidence >= PRELABEL_MIN_SCORE }.joinToString("") { d ->
            val b = d.box
            String.format(Locale.ROOT, "0 %.6f %.6f %.6f %.6f\n", b.centerX, b.centerY, b.width, b.height)
        }
        File(dir, "$base.txt").writeText(labels)
        return File(dir, "$base.jpg")
    }

    fun counts(): Map<CaptureCategory, Int> = CaptureCategory.entries.associateWith { c ->
        File(root, c.folder).listFiles { f -> f.extension == "jpg" }?.size ?: 0
    }

    fun totalBytes(): Long = root.walkTopDown().filter { it.isFile }.sumOf { it.length() }

    fun exportZip(out: OutputStream) = zipDirectory(root, out)

    fun deleteAll() {
        root.deleteRecursively()
    }

    private companion object {
        const val PRELABEL_MIN_SCORE = 0.5f
    }
}

fun zipDirectory(dir: File, out: OutputStream) {
    ZipOutputStream(out.buffered()).use { zip ->
        if (!dir.exists()) return
        dir.walkTopDown().filter { it.isFile }.forEach { f ->
            zip.putNextEntry(ZipEntry(f.relativeTo(dir).path))
            f.inputStream().use { it.copyTo(zip) }
            zip.closeEntry()
        }
    }
}
