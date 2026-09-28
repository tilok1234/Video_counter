package com.palletcounter.app.detection

import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.Rect
import com.palletcounter.core.geometry.Box
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * One camera or video frame handed to a detector.
 *
 * [bitmap] is in sensor orientation; only its top-left [width] x [height] pixels are valid
 * (camera buffers can carry row padding). [rotationDegrees] is the clockwise rotation that
 * makes the frame upright. Detectors return boxes in normalized *upright* coordinates.
 */
class FrameInput(
    val bitmap: Bitmap,
    val width: Int,
    val height: Int,
    val rotationDegrees: Int,
    val timestampNanos: Long,
) {
    val uprightWidth: Int get() = if (rotationDegrees % 180 == 0) width else height
    val uprightHeight: Int get() = if (rotationDegrees % 180 == 0) height else width

    /** Pixel rectangle in [bitmap] covered by a normalized upright [box]. */
    fun sourceRect(box: Box): Rect {
        val corners = listOf(box.left to box.top, box.right to box.bottom).map { (u, v) -> uprightToSource(u, v) }
        val x1 = min(corners[0].first, corners[1].first).coerceIn(0, width - 1)
        val x2 = max(corners[0].first, corners[1].first).coerceIn(x1 + 1, width)
        val y1 = min(corners[0].second, corners[1].second).coerceIn(0, height - 1)
        val y2 = max(corners[0].second, corners[1].second).coerceIn(y1 + 1, height)
        return Rect(x1, y1, x2, y2)
    }

    private fun uprightToSource(u: Float, v: Float): Pair<Int, Int> {
        val (xs, ys) = when ((rotationDegrees % 360 + 360) % 360) {
            90 -> v * width to (1f - u) * height
            180 -> (1f - u) * width to (1f - v) * height
            270 -> (1f - v) * width to u * height
            else -> u * width to v * height
        }
        return xs.roundToInt() to ys.roundToInt()
    }

    /** Upright copy of the region [box] (or the whole frame), scaled so its longest side <= [maxSide]. */
    fun uprightCrop(box: Box? = null, maxSide: Int = 0, margin: Float = 0f): Bitmap {
        val region = box?.let {
            val mx = it.width * margin
            val my = it.height * margin
            Box(it.left - mx, it.top - my, it.right + mx, it.bottom + my).clippedToUnit()
        }
        val rect = region?.let(::sourceRect) ?: Rect(0, 0, width, height)
        val longest = max(rect.width(), rect.height()).toFloat()
        val scale = if (maxSide > 0 && longest > maxSide) maxSide / longest else 1f
        val m = Matrix().apply {
            postScale(scale, scale)
            postRotate(rotationDegrees.toFloat())
        }
        return Bitmap.createBitmap(bitmap, rect.left, rect.top, rect.width(), rect.height(), m, true)
    }
}
