package com.palletcounter.core.tracking

import com.palletcounter.core.geometry.Box
import kotlin.math.max

/**
 * Constant-velocity Kalman filter for one coordinate: state = (position, velocity per
 * second), continuous white-noise-acceleration process noise, variable time step.
 *
 * Four of these (centre x, centre y, width, height) make up [BoxFilter]. Because the
 * SORT/ByteTrack 8-state filter keeps the coordinates uncorrelated, splitting it into
 * independent 2-state filters is equivalent, and it lets the tracker use real timestamps
 * (inference rate varies with device load) instead of assuming one step per frame.
 */
internal class Kalman1D(
    position: Double,
    velocity: Double,
    positionVariance: Double,
    velocityVariance: Double,
) {
    var position: Double = position
        private set
    var velocity: Double = velocity
        private set

    private var p00 = positionVariance
    private var p01 = 0.0
    private var p11 = velocityVariance

    fun predict(dt: Double, accelerationStd: Double) {
        if (dt <= 0.0) return
        position += velocity * dt
        val q = accelerationStd * accelerationStd
        val dt2 = dt * dt
        val n00 = p00 + 2.0 * dt * p01 + dt2 * p11 + q * dt2 * dt / 3.0
        val n01 = p01 + dt * p11 + q * dt2 / 2.0
        val n11 = p11 + q * dt
        p00 = n00
        p01 = n01
        p11 = n11
    }

    fun update(measurement: Double, measurementVariance: Double) {
        val s = p00 + measurementVariance
        if (s <= 0.0) return
        val k0 = p00 / s
        val k1 = p01 / s
        val residual = measurement - position
        position += k0 * residual
        velocity += k1 * residual
        val n00 = (1.0 - k0) * p00
        val n01 = (1.0 - k0) * p01
        val n11 = p11 - k1 * p01
        p00 = n00
        p01 = n01
        p11 = n11
    }

    /** Replaces the velocity estimate, e.g. with the camera-motion estimate for a lost track. */
    fun overrideVelocity(v: Double, variance: Double) {
        velocity = v
        p01 = 0.0
        p11 = variance
    }
}

/** Box filter over (cx, cy, w, h) in normalized frame coordinates. */
internal class BoxFilter(
    box: Box,
    velocityX: Float,
    velocityY: Float,
    velocityStdInBoxes: Float,
    private val config: TrackerConfig,
) {
    private val cx: Kalman1D
    private val cy: Kalman1D
    private val w: Kalman1D
    private val h: Kalman1D

    init {
        val sx = max(box.width, MIN_SCALE).toDouble()
        val sy = max(box.height, MIN_SCALE).toDouble()
        val m = config.measurementNoise.toDouble()
        val vs = velocityStdInBoxes.toDouble()
        cx = Kalman1D(box.centerX.toDouble(), velocityX.toDouble(), sq(m * sx), sq(vs * sx))
        cy = Kalman1D(box.centerY.toDouble(), velocityY.toDouble(), sq(m * sy), sq(vs * sy))
        w = Kalman1D(box.width.toDouble(), 0.0, sq(m * sx), sq(0.1 * sx))
        h = Kalman1D(box.height.toDouble(), 0.0, sq(m * sy), sq(0.1 * sy))
    }

    val box: Box
        get() = Box.fromCenter(
            cx.position.toFloat(),
            cy.position.toFloat(),
            max(w.position.toFloat(), MIN_SCALE),
            max(h.position.toFloat(), MIN_SCALE),
        )

    val velocityX: Float get() = cx.velocity.toFloat()
    val velocityY: Float get() = cy.velocity.toFloat()

    fun predict(dt: Double) {
        val sx = max(w.position, MIN_SCALE.toDouble())
        val sy = max(h.position, MIN_SCALE.toDouble())
        val a = config.accelerationNoise.toDouble()
        val sa = config.sizeAccelerationNoise.toDouble()
        cx.predict(dt, a * sx)
        cy.predict(dt, a * sy)
        w.predict(dt, sa * sx)
        h.predict(dt, sa * sy)
    }

    fun update(box: Box) {
        val m = config.measurementNoise.toDouble()
        val sx = max(box.width, MIN_SCALE).toDouble()
        val sy = max(box.height, MIN_SCALE).toDouble()
        cx.update(box.centerX.toDouble(), sq(m * sx))
        cy.update(box.centerY.toDouble(), sq(m * sy))
        w.update(box.width.toDouble(), sq(m * sx))
        h.update(box.height.toDouble(), sq(m * sy))
    }

    fun overrideVelocity(vx: Float, vy: Float) {
        val sx = max(w.position, MIN_SCALE.toDouble())
        val sy = max(h.position, MIN_SCALE.toDouble())
        cx.overrideVelocity(vx.toDouble(), sq(0.3 * sx))
        cy.overrideVelocity(vy.toDouble(), sq(0.3 * sy))
    }

    private companion object {
        const val MIN_SCALE = 0.01f
        fun sq(x: Double) = x * x
    }
}
