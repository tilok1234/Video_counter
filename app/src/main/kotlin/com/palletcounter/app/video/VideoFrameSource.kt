package com.palletcounter.app.video

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import com.palletcounter.app.detection.FrameInput
import kotlin.math.max
import kotlin.math.min

/**
 * Decodes frames of a recorded video at a fixed rate (e.g. 10 per second) for offline
 * replay through the scan pipeline. Timestamps come from the video timeline, so tracking
 * behaves as if the frames were live.
 *
 * Orientation: some decoders apply the video's rotation metadata and some do not; this
 * compares the decoded bitmap with the stored size to decide, and [extraRotation] fixes
 * footage that plays back sideways.
 */
class VideoFrameSource(
    private val context: Context,
    private val uri: Uri,
    private val framesPerSecond: Double,
    private val extraRotation: Int = 0,
    private val maxSide: Int = 1280,
) : AutoCloseable {
    private val retriever = MediaMetadataRetriever().apply { setDataSource(context, uri) }

    val durationMs: Long = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
    private val metaRotation = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull() ?: 0
    private val metaWidth = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
    private val metaHeight = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
    private val nativeFps: Double = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
        val count = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toDoubleOrNull()
        if (count != null && durationMs > 0) count * 1000.0 / durationMs else 30.0
    } else {
        30.0
    }

    val frameCount: Int get() = max(1, (durationMs / 1000.0 * framesPerSecond).toInt())

    /** Decodes the [index]-th sampled frame, or null past the end. */
    fun frame(index: Int): FrameInput? {
        val timeUs = (index / framesPerSecond * 1_000_000).toLong()
        if (timeUs > durationMs * 1000) return null
        val bitmap = decodeAt(timeUs) ?: return null
        val scaled = scaleDown(bitmap)
        val rotation = (residualRotation(bitmap) + extraRotation) % 360
        return FrameInput(scaled, scaled.width, scaled.height, rotation, timeUs * 1000)
    }

    private fun decodeAt(timeUs: Long): Bitmap? {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P && nativeFps > 0) {
            val count = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toIntOrNull()
            if (count != null && count > 0) {
                val idx = min(count - 1, (timeUs / 1_000_000.0 * nativeFps).toInt())
                runCatching { return retriever.getFrameAtIndex(idx) }
            }
        }
        return retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST)
    }

    /** Rotation still needed after decoding (0 if the decoder already applied metadata). */
    private fun residualRotation(bitmap: Bitmap): Int {
        if (metaRotation % 180 == 0 || metaWidth == 0 || metaHeight == 0) return metaRotation
        val decodedLandscape = bitmap.width >= bitmap.height
        val storedLandscape = metaWidth >= metaHeight
        return if (decodedLandscape == storedLandscape) metaRotation else 0
    }

    private fun scaleDown(bitmap: Bitmap): Bitmap {
        val longest = max(bitmap.width, bitmap.height)
        if (maxSide <= 0 || longest <= maxSide) return bitmap
        val s = maxSide.toFloat() / longest
        val out = Bitmap.createScaledBitmap(bitmap, (bitmap.width * s).toInt(), (bitmap.height * s).toInt(), true)
        if (out !== bitmap) bitmap.recycle()
        return out
    }

    override fun close() {
        retriever.release()
    }
}
