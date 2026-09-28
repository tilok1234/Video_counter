package com.palletcounter.core.tracking

import com.palletcounter.core.detection.Detection
import com.palletcounter.core.geometry.Box
import kotlinx.serialization.Serializable

@Serializable
enum class TrackState {
    /** Newly born, not yet seen often enough to be trusted. Never counted in this state. */
    TENTATIVE,

    /** Seen in at least `minHitsToConfirm` frames. */
    CONFIRMED,

    /** Not matched recently; still predicted and eligible for re-association. */
    LOST,

    /** Dropped (lost too long, tentative miss, or merged by stitching). */
    REMOVED,
}

/** Which association stage last matched a track; shown in the debug overlay. */
@Serializable
enum class MatchStage { BIRTH, HIGH, LOW, TENTATIVE, RECOVERED }

/**
 * A tracked physical object. Mutable, owned by [ByteTracker]; the pipeline exposes
 * immutable snapshots of it to the UI.
 */
class Track internal constructor(
    val id: Int,
    detection: Detection,
    val startNanos: Long,
    velocityX: Float,
    velocityY: Float,
    hasVelocityPrior: Boolean,
    config: TrackerConfig,
) {
    var state: TrackState = TrackState.TENTATIVE
        internal set

    /** Number of detections matched to this track. */
    var hits: Int = 1
        private set

    /** Consecutive processed frames without a matched detection. */
    var missedFrames: Int = 0
        private set

    var lastUpdateNanos: Long = startNanos
        private set

    var lastDetection: Detection = detection
        private set

    var maxConfidence: Float = detection.confidence
        private set

    private var confidenceSum: Float = detection.confidence

    val meanConfidence: Float get() = confidenceSum / hits

    /** Times this track was recovered from [TrackState.LOST]. */
    var recoveries: Int = 0
        private set

    var lastMatchStage: MatchStage = MatchStage.BIRTH
        private set

    /** True if a detection was matched in the most recent tracker update. */
    var updatedThisFrame: Boolean = true
        internal set

    private val filter = BoxFilter(
        detection.box,
        velocityX,
        velocityY,
        if (hasVelocityPrior) config.velocityPriorStd else config.unknownVelocityStd,
        config,
    )

    private val trailX = FloatArray(TRAIL_LENGTH)
    private val trailY = FloatArray(TRAIL_LENGTH)
    private var trailSize = 0
    private var trailHead = 0

    init {
        pushTrail(detection.box)
    }

    /** Current filtered box (a prediction if the track was not updated this frame). */
    val box: Box get() = filter.box

    /** Velocity in normalized frame units per second. */
    val velocityX: Float get() = filter.velocityX
    val velocityY: Float get() = filter.velocityY

    fun ageMillis(nowNanos: Long): Long = (nowNanos - startNanos) / 1_000_000

    fun millisSinceUpdate(nowNanos: Long): Long = (nowNanos - lastUpdateNanos) / 1_000_000

    /** Recent detection centres, oldest first, as (x, y) pairs. */
    fun trail(): List<Pair<Float, Float>> = List(trailSize) { k ->
        val idx = (trailHead - trailSize + k + TRAIL_LENGTH) % TRAIL_LENGTH
        trailX[idx] to trailY[idx]
    }

    internal fun predict(dtSeconds: Double) {
        filter.predict(dtSeconds)
    }

    internal fun update(detection: Detection, timestampNanos: Long, stage: MatchStage) {
        filter.update(detection.box)
        if (state == TrackState.LOST) recoveries++
        hits++
        missedFrames = 0
        lastUpdateNanos = timestampNanos
        lastDetection = detection
        confidenceSum += detection.confidence
        if (detection.confidence > maxConfidence) maxConfidence = detection.confidence
        lastMatchStage = if (state == TrackState.LOST) MatchStage.RECOVERED else stage
        updatedThisFrame = true
        pushTrail(detection.box)
    }

    internal fun markMissed() {
        missedFrames++
        updatedThisFrame = false
    }

    internal fun compensateMotion(vx: Float, vy: Float) {
        filter.overrideVelocity(vx, vy)
    }

    private fun pushTrail(b: Box) {
        trailX[trailHead] = b.centerX
        trailY[trailHead] = b.centerY
        trailHead = (trailHead + 1) % TRAIL_LENGTH
        if (trailSize < TRAIL_LENGTH) trailSize++
    }

    override fun toString(): String = "Track#$id($state hits=$hits box=$box)"

    private companion object {
        const val TRAIL_LENGTH = 24
    }
}
