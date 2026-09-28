package com.palletcounter.app.debug

import android.content.Context
import com.palletcounter.core.detection.DetectorInfo
import com.palletcounter.core.pipeline.PipelineConfig
import com.palletcounter.core.replay.DetectionLogWriter
import com.palletcounter.core.replay.LogHeader
import com.palletcounter.core.session.ScanSettings
import java.io.File
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/**
 * Detection logs (JSONL: boxes and scores only, never images) for every scan, so a bad
 * count can be replayed on a PC with `pallet-core replay`. Kept in app-private storage;
 * the newest [MAX_LOGS] are retained.
 */
class LogStore(context: Context) {
    val dir = File(context.filesDir, "scan_logs").apply { mkdirs() }

    fun open(source: String, detector: DetectorInfo?, settings: ScanSettings, pipeline: PipelineConfig, expectedCount: Int?, video: String?): Pair<File, DetectionLogWriter> {
        prune()
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.ROOT).format(Date())
        val file = File(dir, "scan_${stamp}_$source.jsonl")
        val writer = DetectionLogWriter(file.bufferedWriter())
        val iso = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.ROOT).apply { timeZone = TimeZone.getTimeZone("UTC") }.format(Date())
        writer.header(
            LogHeader(
                createdAt = iso,
                source = "android-$source",
                detector = detector,
                settings = settings,
                pipeline = pipeline,
                expectedCount = expectedCount,
                video = video,
            ),
        )
        return file to writer
    }

    fun list(): List<File> = dir.listFiles { f -> f.extension == "jsonl" }?.sortedByDescending { it.lastModified() } ?: emptyList()

    fun totalBytes(): Long = list().sumOf { it.length() }

    fun exportZip(out: OutputStream) = zipDirectory(dir, out)

    fun deleteAll() {
        list().forEach { it.delete() }
    }

    private fun prune() {
        list().drop(MAX_LOGS - 1).forEach { it.delete() }
    }

    private companion object {
        const val MAX_LOGS = 200
    }
}
