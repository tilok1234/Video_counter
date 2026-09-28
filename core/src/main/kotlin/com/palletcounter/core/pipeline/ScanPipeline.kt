package com.palletcounter.core.pipeline

import com.palletcounter.core.counting.CountEvent
import com.palletcounter.core.counting.CounterConfig
import com.palletcounter.core.counting.CounterStats
import com.palletcounter.core.counting.LineCounter
import com.palletcounter.core.counting.LineSide
import com.palletcounter.core.counting.MotionStatus
import com.palletcounter.core.detection.Detection
import com.palletcounter.core.detection.DetectionFrame
import com.palletcounter.core.geometry.Box
import com.palletcounter.core.tracking.ByteTracker
import com.palletcounter.core.tracking.MatchStage
import com.palletcounter.core.tracking.TrackState
import com.palletcounter.core.tracking.TrackerConfig
import kotlinx.serialization.Serializable
import kotlin.math.abs

/**
 * Everything that turns a stream of [DetectionFrame]s into a pallet count.
 * Serializable so a scan log can record exactly which parameters produced its count.
 */
@Serializable
data class PipelineConfig(
    /**
     * Region of interest in normalized upright-frame coordinates. Only detections whose
     * centre lies inside (and that overlap it by [roiMinFractionInside]) are tracked, so
     * background stacks, machinery and upper tiers outside the guide band are ignored.
     */
    val roi: Box = DEFAULT_ROI,
    val roiMinFractionInside: Float = 0.5f,
    /** Detections smaller than this (normalized height) are ignored. */
    val minBoxHeight: Float = 0.015f,
    val tracker: TrackerConfig = TrackerConfig(),
    val counter: CounterConfig = CounterConfig(),
) {
    companion object {
        val DEFAULT_ROI = Box(0f, 0.25f, 1f, 0.85f)
    }
}

/** Immutable per-track view for overlays, logs and tests. */
@Serializable
data class TrackSnapshot(
    val id: Int,
    val box: Box,
    val state: TrackState,
    val confidence: Float,
    val maxConfidence: Float,
    val hits: Int,
    val ageMillis: Long,
    val millisSinceUpdate: Long,
    val updatedThisFrame: Boolean,
    val velocityX: Float,
    val matchStage: MatchStage,
    val recoveries: Int,
    /** Net contribution to the signed count (-1, 0, +1). */
    val contribution: Int,
    /** True if this track's contribution is part of the currently displayed count. */
    val counted: Boolean,
    val eligible: Boolean,
    val pendingCrossings: Int,
    val sizeFiltered: Boolean,
    val stitchedFrom: Int?,
    val side: LineSide?,
    val trail: List<Pair<Float, Float>> = emptyList(),
)

@Serializable
data class PipelineStats(
    val framesProcessed: Long = 0,
    val detectionsSeen: Long = 0,
    val detectionsInRoi: Long = 0,
    val tracksCreated: Int = 0,
    val maxConcurrentTracks: Int = 0,
    val counter: CounterStats = CounterStats(),
)

/** Result of processing one frame. */
@Serializable
data class ScanSnapshot(
    val timestampNanos: Long,
    val frameIndex: Long,
    val inferenceMillis: Float,
    val rawDetections: List<Detection>,
    val acceptedDetections: List<Detection>,
    val tracks: List<TrackSnapshot>,
    /** Signed net crossings (left→right minus right→left). */
    val netCount: Int,
    /** Pallets counted so far: |netCount|. */
    val count: Int,
    val newEvents: List<CountEvent>,
    val motion: MotionStatus,
    val stats: PipelineStats,
    val config: PipelineConfig,
)

/**
 * ROI filter → [ByteTracker] → [LineCounter].
 *
 * The same instance type is used by the live camera, video-file replay on the phone and the
 * desktop replay CLI, so a recorded detection log reproduces the app's count exactly.
 * Not thread-safe: feed it from a single worker.
 */
class ScanPipeline(config: PipelineConfig = PipelineConfig()) {
    var config: PipelineConfig = config
        private set

    private var tracker = ByteTracker(config.tracker)
    private val counter = LineCounter(config.counter)
    private var stats = PipelineStats()
    private val events = ArrayList<CountEvent>()
    private var lastSnapshot: ScanSnapshot? = null

    /** All count events since the last [reset], in order. */
    val allEvents: List<CountEvent> get() = events.toList()

    val count: Int get() = counter.count
    val netCount: Int get() = counter.signedCount
    val latest: ScanSnapshot? get() = lastSnapshot

    fun reset() {
        tracker = ByteTracker(config.tracker)
        counter.reset()
        stats = PipelineStats()
        events.clear()
        lastSnapshot = null
    }

    /**
     * Applies new parameters. ROI and counter changes take effect immediately without losing
     * the count; tracker changes restart tracking (the count is kept).
     */
    fun updateConfig(newConfig: PipelineConfig) {
        if (newConfig.tracker != config.tracker) tracker = ByteTracker(newConfig.tracker)
        if (newConfig.counter != config.counter) counter.updateConfig(newConfig.counter)
        config = newConfig
    }

    fun process(frame: DetectionFrame): ScanSnapshot {
        val accepted = frame.detections.filter(::acceptDetection)
        val trackerOut = tracker.update(frame.copy(detections = accepted))
        val counterUpdate = counter.update(trackerOut)
        for (id in counterUpdate.tracksToRemove) tracker.removeTrack(id)
        events += counterUpdate.events

        val removedIds = counterUpdate.tracksToRemove.toSet()
        val sign = if (counterUpdate.netCount >= 0) 1 else -1
        val now = frame.timestampNanos
        val tracks = trackerOut.tracks.filter { it.id !in removedIds }.map { t ->
            val info = counterUpdate.trackInfo[t.id]
            val contribution = info?.contribution ?: 0
            TrackSnapshot(
                id = t.id,
                box = t.box,
                state = t.state,
                confidence = t.lastDetection.confidence,
                maxConfidence = t.maxConfidence,
                hits = t.hits,
                ageMillis = t.ageMillis(now),
                millisSinceUpdate = t.millisSinceUpdate(now),
                updatedThisFrame = t.updatedThisFrame,
                velocityX = t.velocityX,
                matchStage = t.lastMatchStage,
                recoveries = t.recoveries,
                contribution = contribution,
                counted = contribution != 0 && contribution == sign && counterUpdate.netCount != 0,
                eligible = info?.eligible ?: false,
                pendingCrossings = info?.pendingCrossings ?: 0,
                sizeFiltered = info?.sizeFiltered ?: false,
                stitchedFrom = info?.stitchedFrom,
                side = info?.side,
                trail = t.trail(),
            )
        }
        stats = stats.copy(
            framesProcessed = stats.framesProcessed + 1,
            detectionsSeen = stats.detectionsSeen + frame.detections.size,
            detectionsInRoi = stats.detectionsInRoi + accepted.size,
            tracksCreated = stats.tracksCreated + trackerOut.born.size,
            maxConcurrentTracks = maxOf(stats.maxConcurrentTracks, tracks.count { it.state != TrackState.LOST }),
            counter = counterUpdate.stats,
        )
        val snapshot = ScanSnapshot(
            timestampNanos = now,
            frameIndex = frame.frameIndex,
            inferenceMillis = frame.inferenceMillis,
            rawDetections = frame.detections,
            acceptedDetections = accepted,
            tracks = tracks,
            netCount = counterUpdate.netCount,
            count = abs(counterUpdate.netCount),
            newEvents = counterUpdate.events,
            motion = counterUpdate.motion,
            stats = stats,
            config = config,
        )
        lastSnapshot = snapshot
        return snapshot
    }

    private fun acceptDetection(d: Detection): Boolean {
        val b = d.box
        if (!b.isValid || b.height < config.minBoxHeight) return false
        val roi = config.roi
        return roi.contains(b.centerX, b.centerY) && b.fractionInside(roi) >= config.roiMinFractionInside
    }
}
