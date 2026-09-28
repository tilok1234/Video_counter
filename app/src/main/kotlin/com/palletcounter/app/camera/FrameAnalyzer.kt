package com.palletcounter.app.camera

import android.graphics.Bitmap
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.palletcounter.app.detection.FrameInput

/**
 * Receives every camera frame (for the camera-FPS readout) but converts and forwards a
 * frame only when [gate] says the detector is ready for one. The RGBA buffer is copied into
 * one reused bitmap, so no per-frame allocations happen; the gate guarantees the previous
 * frame has been fully processed before the bitmap is overwritten.
 */
class FrameAnalyzer(
    private val gate: FrameGate,
) : ImageAnalysis.Analyzer {
    private var bitmap: Bitmap? = null

    override fun analyze(image: ImageProxy) {
        val timestamp = image.imageInfo.timestamp
        gate.onCameraFrame(timestamp)
        if (!gate.tryBegin(timestamp)) {
            image.close()
            return
        }
        try {
            val plane = image.planes[0]
            val rowPixels = plane.rowStride / plane.pixelStride
            val bmp = bitmap?.takeIf { it.width == rowPixels && it.height == image.height }
                ?: Bitmap.createBitmap(rowPixels, image.height, Bitmap.Config.ARGB_8888).also { bitmap = it }
            val buffer = plane.buffer
            buffer.rewind()
            bmp.copyPixelsFromBuffer(buffer)
            val frame = FrameInput(bmp, image.width, image.height, image.imageInfo.rotationDegrees, timestamp)
            image.close()
            gate.submit(frame)
        } catch (t: Throwable) {
            image.close()
            gate.abort(t)
        }
    }
}

/** Implemented by the scan engine: decides which frames are processed. */
interface FrameGate {
    fun onCameraFrame(timestampNanos: Long)

    /** Returns true (and reserves the worker) if a new frame should be converted now. */
    fun tryBegin(timestampNanos: Long): Boolean

    /** Hands over a frame after a successful [tryBegin]. */
    fun submit(frame: FrameInput)

    /** Releases a reservation made by [tryBegin] when conversion failed. */
    fun abort(error: Throwable)
}
