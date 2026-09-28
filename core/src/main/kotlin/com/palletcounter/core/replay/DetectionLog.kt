package com.palletcounter.core.replay

import com.palletcounter.core.detection.Detection
import com.palletcounter.core.detection.DetectionFrame
import com.palletcounter.core.detection.DetectorInfo
import com.palletcounter.core.geometry.Box
import com.palletcounter.core.pipeline.PipelineConfig
import com.palletcounter.core.session.ScanSettings
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.float
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long
import java.io.BufferedReader
import java.io.Closeable
import java.io.Writer
import java.util.Locale

/**
 * Detection log format (JSON Lines, one object per line). Contains boxes and scores only —
 * never image data — so logs are small and safe to keep. See docs/TESTING.md.
 *
 * ```
 * {"type":"header","version":1,"source":"android-live",...}
 * {"type":"frame","t":123456789,"i":0,"w":1280,"h":960,"ms":21.5,"d":[[l,t,r,b,score,cls],...]}
 * {"type":"footer","count":17,"adjustment":1,"accepted":true}
 * ```
 */
object DetectionLog {
    const val VERSION = 1

    internal val json = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
        explicitNulls = false
    }

    fun formatFrame(frame: DetectionFrame): String {
        val sb = StringBuilder(64 + frame.detections.size * 48)
        sb.append("{\"type\":\"frame\",\"t\":").append(frame.timestampNanos)
        sb.append(",\"i\":").append(frame.frameIndex)
        if (frame.frameWidth > 0) sb.append(",\"w\":").append(frame.frameWidth)
        if (frame.frameHeight > 0) sb.append(",\"h\":").append(frame.frameHeight)
        if (frame.inferenceMillis >= 0f) sb.append(",\"ms\":").append(fmt(frame.inferenceMillis, 2))
        sb.append(",\"d\":[")
        frame.detections.forEachIndexed { k, d ->
            if (k > 0) sb.append(',')
            sb.append('[')
                .append(fmt(d.box.left)).append(',')
                .append(fmt(d.box.top)).append(',')
                .append(fmt(d.box.right)).append(',')
                .append(fmt(d.box.bottom)).append(',')
                .append(fmt(d.confidence, 3)).append(',')
                .append(d.classId)
                .append(']')
        }
        sb.append("]}")
        return sb.toString()
    }

    fun formatHeader(header: LogHeader): String = json.encodeToString(LogHeader.serializer(), header)

    fun formatFooter(footer: LogFooter): String = json.encodeToString(LogFooter.serializer(), footer)

    /** Parses a whole log. Malformed lines are reported in [ParsedLog.errors], not thrown. */
    fun parse(lines: Sequence<String>): ParsedLog {
        var header: LogHeader? = null
        var footer: LogFooter? = null
        val frames = ArrayList<DetectionFrame>()
        val errors = ArrayList<String>()
        lines.forEachIndexed { lineNo, raw ->
            val line = raw.trim()
            if (line.isEmpty()) return@forEachIndexed
            try {
                val obj = json.parseToJsonElement(line).jsonObject
                when (obj["type"]?.jsonPrimitive?.content) {
                    "header" -> header = json.decodeFromJsonElement(LogHeader.serializer(), obj)
                    "footer" -> footer = json.decodeFromJsonElement(LogFooter.serializer(), obj)
                    "frame" -> frames += parseFrame(obj)
                    else -> errors += "line ${lineNo + 1}: unknown type"
                }
            } catch (e: Exception) {
                errors += "line ${lineNo + 1}: ${e.message}"
            }
        }
        return ParsedLog(header, frames, footer, errors)
    }

    fun parse(reader: BufferedReader): ParsedLog = reader.useLines { parse(it) }

    private fun parseFrame(obj: JsonObject): DetectionFrame {
        val dets = (obj["d"] as? JsonArray ?: JsonArray(emptyList())).map { el ->
            val a = el.jsonArray
            Detection(
                box = Box(a[0].jsonPrimitive.float, a[1].jsonPrimitive.float, a[2].jsonPrimitive.float, a[3].jsonPrimitive.float),
                confidence = a[4].jsonPrimitive.float,
                classId = if (a.size > 5) a[5].jsonPrimitive.int else 0,
            )
        }
        return DetectionFrame(
            timestampNanos = obj.getValue("t").jsonPrimitive.long,
            detections = dets,
            frameWidth = obj["w"]?.jsonPrimitive?.int ?: 0,
            frameHeight = obj["h"]?.jsonPrimitive?.int ?: 0,
            inferenceMillis = obj["ms"]?.jsonPrimitive?.float ?: -1f,
            frameIndex = obj["i"]?.jsonPrimitive?.long ?: -1,
        )
    }

    /** Locale-independent compact number formatting (a comma decimal would break JSON). */
    internal fun fmt(v: Float, decimals: Int = 4): String {
        if (!v.isFinite()) return "0"
        val s = String.format(Locale.ROOT, "%.${decimals}f", v)
        return if (s.contains('.')) s.trimEnd('0').trimEnd('.').let { if (it == "-0") "0" else it } else s
    }
}

@Serializable
data class LogHeader(
    val type: String = "header",
    val version: Int = DetectionLog.VERSION,
    @SerialName("created_at") val createdAt: String? = null,
    /** "android-live", "android-video", "python-predict", "simulator", … */
    val source: String = "unknown",
    val detector: DetectorInfo? = null,
    val settings: ScanSettings? = null,
    /** Pipeline parameters the recording app used; replay uses them unless overridden. */
    val pipeline: PipelineConfig? = null,
    /** Ground truth if known (e.g. counted by hand), used by `eval`. */
    @SerialName("expected_count") val expectedCount: Int? = null,
    val video: String? = null,
    val notes: String? = null,
)

@Serializable
data class LogFooter(
    val type: String = "footer",
    /** Count shown when the scan was finished. */
    val count: Int,
    @SerialName("net_count") val netCount: Int? = null,
    /** Manual +/- applied on the review screen. */
    val adjustment: Int = 0,
    val accepted: Boolean = false,
    @SerialName("half_pallets") val halfPallets: Int? = null,
)

data class ParsedLog(
    val header: LogHeader?,
    val frames: List<DetectionFrame>,
    val footer: LogFooter?,
    val errors: List<String>,
)

/** Streaming writer; the app appends one line per processed frame. */
class DetectionLogWriter(private val out: Writer) : Closeable {
    fun header(header: LogHeader) = line(DetectionLog.formatHeader(header))
    fun frame(frame: DetectionFrame) = line(DetectionLog.formatFrame(frame))
    fun footer(footer: LogFooter) = line(DetectionLog.formatFooter(footer))

    private fun line(s: String) {
        out.write(s)
        out.write("\n")
    }

    fun flush() = out.flush()

    override fun close() = out.close()
}
