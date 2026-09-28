package com.palletcounter.app.scan

import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import com.palletcounter.app.camera.FrameGate
import com.palletcounter.app.data.AppSettings
import com.palletcounter.app.debug.CaptureCategory
import com.palletcounter.app.debug.CaptureStore
import com.palletcounter.app.debug.LogStore
import com.palletcounter.app.detection.FrameInput
import com.palletcounter.app.detection.ModelManager
import com.palletcounter.core.detection.DetectionFrame
import com.palletcounter.core.detection.DetectorInfo
import com.palletcounter.core.detection.PalletDetector
import com.palletcounter.core.pipeline.ScanPipeline
import com.palletcounter.core.pipeline.ScanSnapshot
import com.palletcounter.core.replay.DetectionLogWriter
import com.palletcounter.core.replay.LogFooter
import com.palletcounter.core.session.ScanSettings
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs

enum class EngineStatus { LOADING, RUNNING, ERROR, FINISHED }

data class PerfInfo(
    val inferenceMs: Float = 0f,
    val pipelineMs: Float = 0f,
    val detectorFps: Float = 0f,
    val cameraFps: Float = 0f,
    val thermal: String = "n/a",
    val throttled: Boolean = false,
)

data class ScanUiState(
    val status: EngineStatus = EngineStatus.LOADING,
    val error: String? = null,
    val detector: DetectorInfo? = null,
    val snapshot: ScanSnapshot? = null,
    val perf: PerfInfo = PerfInfo(),
    val message: String? = null,
    val frames: Long = 0,
    /** Size of the upright analysed frame (for overlay alignment). */
    val frameWidth: Int = 0,
    val frameHeight: Int = 0,
)

/** A counted pallet as shown on the review screen. */
class CountedPallet(val number: Int, val trackId: Int, val thumbnail: Bitmap?)

/** Everything the review screen needs after a scan. */
class ScanResult(
    val detectedCount: Int,
    val netCount: Int,
    val settings: ScanSettings,
    val detector: DetectorInfo?,
    val pallets: List<CountedPallet>,
    /** Frame (upright, downscaled) at the most recent count increase, and its snapshot. */
    val lastCountFrame: Bitmap?,
    val lastCountSnapshot: ScanSnapshot?,
    val lastSnapshot: ScanSnapshot?,
    val durationSeconds: Double,
    val frames: Long,
    val logFile: File?,
    val source: String,
    val expectedCount: Int?,
)

/**
 * Runs detection + tracking + counting on a dedicated worker thread.
 *
 * Live camera: the analyzer asks [tryBegin] before converting a frame, so frames that
 * arrive while the detector is busy (or faster than [AppSettings.maxInferenceFps]) are
 * dropped without being copied. Video replay: [processBlocking] processes every sampled
 * frame in order. Both paths use the same pipeline code as the desktop replay tool.
 */
class ScanEngine(
    private val context: Context,
    private val models: ModelManager,
    private val settings: AppSettings,
    private val source: String,
    private val logStore: LogStore?,
    private val captures: CaptureStore,
    private val expectedCount: Int? = null,
    private val videoName: String? = null,
) : FrameGate, AutoCloseable {
    private val worker: ExecutorService = Executors.newSingleThreadExecutor { Thread(it, "pallet-inference") }
    private val busy = AtomicBoolean(false)
    private val closed = AtomicBoolean(false)
    private val pipeline = ScanPipeline(settings.pipelineConfig())
    private var detector: PalletDetector<FrameInput>? = null
    private var logWriter: DetectionLogWriter? = null
    private var logFile: File? = null
    private val pallets = ArrayList<CountedPallet>()
    private var lastCountFrame: Bitmap? = null
    private var lastCountSnapshot: ScanSnapshot? = null
    private var firstTimestamp: Long? = null
    private var lastTimestamp: Long = 0
    private var frames = 0L

    @Volatile private var captureRequest: CaptureCategory? = null
    @Volatile private var lastAcceptedNanos = 0L
    private var minIntervalNanos = 1_000_000_000L / settings.maxInferenceFps.coerceAtLeast(1)

    private val cameraFps = RateMeter()
    private val detectorFps = RateMeter()
    private var inferenceEma = Ema()
    private var pipelineEma = Ema()
    private val powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager

    private val _state = MutableStateFlow(ScanUiState())
    val state: StateFlow<ScanUiState> = _state.asStateFlow()

    /** Loads the detector on the worker thread (GPU delegates are thread-bound). */
    fun start() {
        worker.execute {
            try {
                val det = models.createDetector(settings, decoderThreshold = settings.lowThreshold)
                detector = det
                if (logStore != null && settings.recordDetectionLogs) {
                    val (file, writer) = logStore.open(source, det.info, settings.scanSettings, pipeline.config, expectedCount, videoName)
                    logFile = file
                    logWriter = writer
                }
                _state.value = _state.value.copy(status = EngineStatus.RUNNING, detector = det.info)
            } catch (t: Throwable) {
                Log.e(TAG, "detector init failed", t)
                _state.value = _state.value.copy(status = EngineStatus.ERROR, error = t.message ?: t.toString())
            }
        }
    }

    // ---- FrameGate (camera analysis thread) ----

    override fun onCameraFrame(timestampNanos: Long) {
        cameraFps.tick(timestampNanos)
    }

    override fun tryBegin(timestampNanos: Long): Boolean {
        if (closed.get() || _state.value.status != EngineStatus.RUNNING) return false
        if (timestampNanos - lastAcceptedNanos < minIntervalNanos) return false
        if (!busy.compareAndSet(false, true)) return false
        lastAcceptedNanos = timestampNanos
        return true
    }

    override fun submit(frame: FrameInput) {
        try {
            worker.execute {
                try {
                    processFrame(frame)
                } catch (t: Throwable) {
                    Log.e(TAG, "frame processing failed", t)
                    _state.value = _state.value.copy(message = "Frame error: ${t.message}")
                } finally {
                    busy.set(false)
                }
            }
        } catch (e: RejectedExecutionException) {
            busy.set(false) // engine closed between tryBegin() and submit()
        }
    }

    override fun abort(error: Throwable) {
        Log.w(TAG, "frame conversion failed", error)
        busy.set(false)
    }

    // ---- Video replay ----

    /** Processes [frame] synchronously in order (offline video replay). */
    fun processBlocking(frame: FrameInput) {
        if (closed.get()) return
        worker.submit(Runnable { processFrame(frame) }).get()
    }

    fun awaitReady(timeoutMs: Long = 30_000): Boolean {
        val deadline = SystemClock.elapsedRealtime() + timeoutMs
        while (SystemClock.elapsedRealtime() < deadline) {
            when (_state.value.status) {
                EngineStatus.RUNNING -> return true
                EngineStatus.ERROR -> return false
                else -> Thread.sleep(20)
            }
        }
        return false
    }

    fun requestCapture(category: CaptureCategory) {
        captureRequest = category
    }

    /** Clears the count and tracks but keeps the detector loaded. */
    fun resetCount() {
        worker.execute {
            pipeline.reset()
            pallets.clear()
            lastCountFrame = null
            lastCountSnapshot = null
            _state.value = _state.value.copy(snapshot = null, message = "Count reset")
        }
    }

    // ---- Worker thread ----

    private fun processFrame(frame: FrameInput) {
        val det = detector ?: return
        val start = SystemClock.elapsedRealtimeNanos()
        val detections: DetectionFrame = det.detect(frame, frame.timestampNanos).copy(frameIndex = frames)
        val snapshot = pipeline.process(detections)
        val pipelineMs = (SystemClock.elapsedRealtimeNanos() - start) / 1e6f - detections.inferenceMillis.coerceAtLeast(0f)
        frames++
        if (firstTimestamp == null) firstTimestamp = frame.timestampNanos
        lastTimestamp = frame.timestampNanos
        logWriter?.frame(detections)
        handleCountEvents(snapshot, frame)
        val message = captureRequest?.let { category ->
            captureRequest = null
            runCatching { captures.save(category, frame.uprightCrop(), snapshot, det.info) }
                .fold({ "Saved ${category.label} sample" }, { "Capture failed: ${it.message}" })
        }
        detectorFps.tick(frame.timestampNanos)
        if (detections.inferenceMillis >= 0f) inferenceEma.add(detections.inferenceMillis)
        pipelineEma.add(pipelineMs)
        val thermal = thermalStatus()
        adaptToThermal(thermal)
        _state.value = _state.value.copy(
            snapshot = snapshot,
            frames = frames,
            frameWidth = frame.uprightWidth,
            frameHeight = frame.uprightHeight,
            message = message ?: _state.value.message.takeIf { frames % 30 != 0L },
            perf = PerfInfo(
                inferenceMs = inferenceEma.value,
                pipelineMs = pipelineEma.value,
                detectorFps = detectorFps.rate,
                cameraFps = cameraFps.rate,
                thermal = thermal,
                throttled = minIntervalNanos > 1_000_000_000L / settings.maxInferenceFps.coerceAtLeast(1),
            ),
        )
    }

    private fun handleCountEvents(snapshot: ScanSnapshot, frame: FrameInput) {
        var previous = snapshot.netCount - snapshot.newEvents.sumOf { it.direction }
        for (event in snapshot.newEvents) {
            val after = previous + event.direction
            if (abs(after) > abs(previous)) {
                val thumb = runCatching { frame.uprightCrop(event.box, maxSide = 240, margin = 0.15f) }.getOrNull()
                pallets += CountedPallet(pallets.size + 1, event.trackId, thumb)
                lastCountFrame = runCatching { frame.uprightCrop(maxSide = 960) }.getOrNull()
                lastCountSnapshot = snapshot
            } else if (pallets.isNotEmpty()) {
                // Walking back un-counts the most recently counted pallet first.
                pallets.removeAt(pallets.lastIndex)
            }
            previous = after
        }
    }

    private fun thermalStatus(): String {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) return "n/a"
        return when (powerManager.currentThermalStatus) {
            PowerManager.THERMAL_STATUS_NONE -> "ok"
            PowerManager.THERMAL_STATUS_LIGHT -> "light"
            PowerManager.THERMAL_STATUS_MODERATE -> "moderate"
            PowerManager.THERMAL_STATUS_SEVERE -> "SEVERE"
            PowerManager.THERMAL_STATUS_CRITICAL -> "CRITICAL"
            else -> "EMERGENCY"
        }
    }

    /** Halves the detector rate when the phone gets hot; restores it when it cools down. */
    private fun adaptToThermal(thermal: String) {
        val base = 1_000_000_000L / settings.maxInferenceFps.coerceAtLeast(1)
        minIntervalNanos = if (thermal in setOf("SEVERE", "CRITICAL", "EMERGENCY")) base * 2 else base
    }

    /** Stops the scan, closes the log and returns the result for the review screen. */
    fun finish(): ScanResult {
        closed.set(true)
        val future = worker.submit<ScanResult> {
            val snap = pipeline.latest
            logWriter?.let {
                runCatching {
                    it.footer(LogFooter(count = pipeline.count, netCount = pipeline.netCount))
                    it.close()
                }
            }
            logWriter = null
            ScanResult(
                detectedCount = pipeline.count,
                netCount = pipeline.netCount,
                settings = settings.scanSettings,
                detector = detector?.info,
                pallets = pallets.toList(),
                lastCountFrame = lastCountFrame,
                lastCountSnapshot = lastCountSnapshot,
                lastSnapshot = snap,
                durationSeconds = firstTimestamp?.let { (lastTimestamp - it) / 1e9 } ?: 0.0,
                frames = frames,
                logFile = logFile,
                source = source,
                expectedCount = expectedCount,
            )
        }
        val result = future.get(10, TimeUnit.SECONDS)
        _state.value = _state.value.copy(status = EngineStatus.FINISHED)
        return result
    }

    override fun close() {
        closed.set(true)
        worker.execute {
            runCatching { logWriter?.close() }
            logWriter = null
            runCatching { detector?.close() }
            detector = null
        }
        worker.shutdown()
    }

    private class RateMeter {
        private val times = ArrayDeque<Long>()
        var rate = 0f
            private set

        @Synchronized
        fun tick(t: Long) {
            times.addLast(t)
            while (times.size > 2 && t - times.first() > 2_000_000_000L) times.removeFirst()
            rate = if (times.size >= 2) (times.size - 1) * 1e9f / (times.last() - times.first()).coerceAtLeast(1) else 0f
        }
    }

    private class Ema(private val alpha: Float = 0.15f) {
        var value = 0f
            private set
        private var initialized = false

        fun add(v: Float) {
            value = if (!initialized) v.also { initialized = true } else value + alpha * (v - value)
        }
    }

    private companion object {
        const val TAG = "ScanEngine"
    }
}
