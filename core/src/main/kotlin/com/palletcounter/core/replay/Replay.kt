package com.palletcounter.core.replay

import com.palletcounter.core.counting.CountEvent
import com.palletcounter.core.pipeline.PipelineConfig
import com.palletcounter.core.pipeline.PipelineStats
import com.palletcounter.core.pipeline.ScanPipeline
import com.palletcounter.core.pipeline.ScanSnapshot

data class ReplayResult(
    val count: Int,
    val netCount: Int,
    val frames: Int,
    val durationSeconds: Double,
    val detections: Int,
    val events: List<CountEvent>,
    val stats: PipelineStats,
    val config: PipelineConfig,
    val expectedCount: Int?,
    /** Count recorded by the app when the scan ended (footer), for reproducibility checks. */
    val recordedCount: Int?,
) {
    val detectorFps: Double get() = if (durationSeconds > 0) (frames - 1) / durationSeconds else 0.0
    val matchesExpected: Boolean? get() = expectedCount?.let { it == count }
}

object Replay {
    /**
     * Runs [log] through a fresh [ScanPipeline]. Uses the configuration recorded in the log
     * header unless [configOverride] is given, so an unmodified replay reproduces the count
     * the app showed.
     */
    fun run(
        log: ParsedLog,
        configOverride: PipelineConfig? = null,
        onFrame: ((ScanSnapshot) -> Unit)? = null,
    ): ReplayResult {
        val config = configOverride ?: log.header?.pipeline ?: PipelineConfig()
        val pipeline = ScanPipeline(config)
        for (frame in log.frames) {
            val snap = pipeline.process(frame)
            onFrame?.invoke(snap)
        }
        val frames = log.frames
        val duration = if (frames.size >= 2) (frames.last().timestampNanos - frames.first().timestampNanos) / 1e9 else 0.0
        return ReplayResult(
            count = pipeline.count,
            netCount = pipeline.netCount,
            frames = frames.size,
            durationSeconds = duration,
            detections = frames.sumOf { it.detections.size },
            events = pipeline.allEvents,
            stats = pipeline.latest?.stats ?: PipelineStats(),
            config = config,
            expectedCount = log.header?.expectedCount,
            recordedCount = log.footer?.count,
        )
    }
}
