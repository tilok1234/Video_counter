package com.palletcounter.app.ui

import android.graphics.Paint
import android.graphics.Typeface
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import com.palletcounter.core.geometry.Box
import com.palletcounter.core.pipeline.PipelineConfig
import com.palletcounter.core.pipeline.ScanSnapshot
import com.palletcounter.core.pipeline.TrackSnapshot
import com.palletcounter.core.tracking.TrackState

data class OverlayOptions(
    val debug: Boolean = true,
    val rawDetections: Boolean = false,
    val trails: Boolean = false,
    val showGuideText: Boolean = true,
)

/** Rectangle occupied by an image of [imageAspect] (w/h) fitted (centred) into [size]. */
fun fittedRect(size: Size, imageAspect: Float): Rect {
    if (imageAspect <= 0f || size.width <= 0f || size.height <= 0f) return Rect(Offset.Zero, size)
    val viewAspect = size.width / size.height
    return if (viewAspect > imageAspect) {
        val w = size.height * imageAspect
        Rect(Offset((size.width - w) / 2f, 0f), Size(w, size.height))
    } else {
        val h = size.width / imageAspect
        Rect(Offset(0f, (size.height - h) / 2f), Size(size.width, h))
    }
}

/**
 * Draws ROI, count line, tracks and (optionally) raw detections over an image of aspect
 * [imageAspect] that is displayed with FIT_CENTER in the same bounds (camera preview,
 * decoded video frame or saved frame).
 */
@Composable
fun DetectionOverlay(
    snapshot: ScanSnapshot?,
    config: PipelineConfig,
    imageAspect: Float,
    options: OverlayOptions,
    modifier: Modifier = Modifier,
) {
    val labelPaint = remember {
        Paint(Paint.ANTI_ALIAS_FLAG).apply {
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            setShadowLayer(4f, 0f, 0f, android.graphics.Color.BLACK)
        }
    }
    Canvas(modifier) {
        val area = fittedRect(size, imageAspect)
        drawGuides(area, config, options, labelPaint)
        if (snapshot == null) return@Canvas
        if (options.rawDetections) {
            for (d in snapshot.rawDetections) {
                val r = map(area, d.box)
                drawRect(Color.White.copy(alpha = 0.6f), r.topLeft, r.size, style = Stroke(1.5f))
                text("%.2f".format(d.confidence), r.left + 4f, r.bottom - 6f, Color.White, 24f, labelPaint)
            }
        }
        for (t in snapshot.tracks) drawTrack(area, t, options, labelPaint)
    }
}

private fun map(area: Rect, b: Box): Rect = Rect(
    area.left + b.left * area.width,
    area.top + b.top * area.height,
    area.left + b.right * area.width,
    area.top + b.bottom * area.height,
)

private fun DrawScope.drawGuides(area: Rect, config: PipelineConfig, options: OverlayOptions, paint: Paint) {
    val roi = map(area, config.roi)
    val shade = Color.Black.copy(alpha = 0.35f)
    // Dim everything outside the ROI.
    drawRect(shade, area.topLeft, Size(area.width, roi.top - area.top))
    drawRect(shade, Offset(area.left, roi.bottom), Size(area.width, area.bottom - roi.bottom))
    drawRect(Color.White.copy(alpha = 0.8f), roi.topLeft, roi.size, style = Stroke(3f, pathEffect = PathEffect.dashPathEffect(floatArrayOf(18f, 12f))))
    if (options.showGuideText) {
        text("KEEP PALLET BASES IN THIS AREA", roi.left + 12f, roi.top + 36f, Color.White, 30f, paint)
    }
    val c = config.counter
    val x = area.left + c.lineX * area.width
    val band = c.hysteresis * area.width
    drawRect(Palette.Accent.copy(alpha = 0.12f), Offset(x - band, roi.top), Size(2 * band, roi.height))
    drawLine(Palette.Accent, Offset(x, roi.top), Offset(x, roi.bottom), strokeWidth = 5f)
    if (options.debug) text("COUNT LINE", x + 8f, roi.bottom - 12f, Palette.Accent, 24f, paint)
}

private fun DrawScope.drawTrack(area: Rect, t: TrackSnapshot, options: OverlayOptions, paint: Paint) {
    val color = when {
        t.sizeFiltered -> Palette.Filtered
        t.counted -> Palette.Counted
        t.state == TrackState.LOST -> Palette.Lost
        t.state == TrackState.TENTATIVE -> Palette.Tentative
        else -> Palette.Confirmed
    }
    val r = map(area, t.box)
    val stroke = if (t.state == TrackState.LOST || t.state == TrackState.TENTATIVE) {
        Stroke(4f, pathEffect = PathEffect.dashPathEffect(floatArrayOf(14f, 10f)))
    } else {
        Stroke(6f)
    }
    drawRect(color, r.topLeft, r.size, style = stroke)
    text("[${t.id}]", r.left + 6f, r.top - 10f, color, 40f, paint)
    if (options.debug) {
        val status = when {
            t.sizeFiltered -> "SMALL (far row?)"
            t.counted -> "COUNTED"
            t.pendingCrossings > 0 -> "PENDING"
            t.state == TrackState.LOST -> "LOST ${t.millisSinceUpdate} ms"
            t.state == TrackState.TENTATIVE -> "TENTATIVE"
            else -> "TRACKED"
        }
        val lines = listOf(
            "ID ${t.id}  %.2f".format(t.confidence),
            "TRACKED ${t.hits} FRAMES",
            status + (t.stitchedFrom?.let { " ←#$it" } ?: ""),
        )
        lines.forEachIndexed { i, s -> text(s, r.left + 6f, r.top + 30f + i * 28f, color, 24f, paint) }
    }
    if (options.trails && t.trail.size > 1) {
        for (i in 1 until t.trail.size) {
            val (x0, y0) = t.trail[i - 1]
            val (x1, y1) = t.trail[i]
            drawLine(
                color.copy(alpha = 0.7f),
                Offset(area.left + x0 * area.width, area.top + y0 * area.height),
                Offset(area.left + x1 * area.width, area.top + y1 * area.height),
                strokeWidth = 3f,
            )
        }
    }
}

private fun DrawScope.text(s: String, x: Float, y: Float, color: Color, sizePx: Float, paint: Paint) {
    paint.color = color.toArgb()
    paint.textSize = sizePx
    drawIntoCanvas { it.nativeCanvas.drawText(s, x, y, paint) }
}
