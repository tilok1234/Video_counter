package com.palletcounter.core.cli

import com.palletcounter.core.geometry.Box
import com.palletcounter.core.pipeline.PipelineConfig
import com.palletcounter.core.replay.DetectionLog
import com.palletcounter.core.replay.DetectionLogWriter
import com.palletcounter.core.replay.LogHeader
import com.palletcounter.core.replay.ParsedLog
import com.palletcounter.core.replay.Replay
import com.palletcounter.core.replay.ReplayResult
import com.palletcounter.core.sim.Blackout
import com.palletcounter.core.sim.Segment
import com.palletcounter.core.sim.SimConfig
import com.palletcounter.core.sim.SyntheticLineScene
import kotlinx.serialization.json.Json
import java.io.File
import kotlin.math.abs
import kotlin.system.exitProcess

private const val USAGE = """
pallet-core — replay/evaluate detection logs with the app's tracking + counting code.

Usage:
  replay <log.jsonl> [options]        Replay one log and print the pallet count.
  eval <dir | manifest.csv> [options] Replay many logs and compare with expected counts.
                                      Directory: every *.jsonl, expected count from its header.
                                      CSV: lines "path,expected" (relative to the CSV file).
  simulate [--scenario NAME] [--pallets N] [--seed S] --out <file.jsonl>
                                      Write a synthetic log (scenarios: steady, noisy,
                                      reverse, jerks, blackout). Plumbing tests only.

Options (override the parameters recorded in the log):
  --config <pipeline.json>   Full PipelineConfig JSON.
  --line <x>                 Count line position (0..1).
  --hysteresis <h>           Half-width of the dead band around the line.
  --roi <l,t,r,b>            Region of interest (normalized).
  --min-hits <n>             Detections before a track may count.
  --high <s> --low <s>       Tracker high / low score thresholds.
  --new <s>                  Minimum score to start a track.
  --lost-ms <ms>             How long lost tracks are kept.
  --size-filter <ratio>      Relative size filter (0 = off).
  --expected <n>             Ground truth for `replay`.
  --events                   Print every count event.
  --trace                    Print one line per frame.

Exit status of `eval` is 1 if any log's count differs from its expected count.
"""

fun main(args: Array<String>) {
    if (args.isEmpty() || args[0] in setOf("-h", "--help", "help")) {
        println(USAGE.trimIndent())
        return
    }
    val opts = Options.parse(args.drop(1))
    val code = when (args[0]) {
        "replay" -> replay(opts)
        "eval" -> evaluate(opts)
        "simulate" -> simulate(opts)
        else -> {
            System.err.println("Unknown command '${args[0]}'.\n${USAGE.trimIndent()}")
            2
        }
    }
    if (code != 0) exitProcess(code)
}

private class Options(val positional: List<String>, val values: Map<String, String>, val flags: Set<String>) {
    fun has(flag: String) = flag in flags
    operator fun get(key: String): String? = values[key]

    companion object {
        private val FLAGS = setOf("--events", "--trace")

        fun parse(args: List<String>): Options {
            val pos = ArrayList<String>()
            val values = HashMap<String, String>()
            val flags = HashSet<String>()
            var i = 0
            while (i < args.size) {
                val a = args[i]
                when {
                    a in FLAGS -> flags += a
                    a.startsWith("--") -> {
                        require(i + 1 < args.size) { "Missing value for $a" }
                        values[a] = args[++i]
                    }
                    else -> pos += a
                }
                i++
            }
            return Options(pos, values, flags)
        }
    }
}

private val json = Json { ignoreUnknownKeys = true; prettyPrint = true; encodeDefaults = true }

/** Applies command-line overrides on top of [base] (null if nothing is overridden). */
private fun overrides(opts: Options, base: PipelineConfig): PipelineConfig? {
    var cfg = opts["--config"]?.let { json.decodeFromString(PipelineConfig.serializer(), File(it).readText()) } ?: base
    var changed = opts["--config"] != null
    fun <T> set(key: String, parse: (String) -> T, apply: (PipelineConfig, T) -> PipelineConfig) {
        opts[key]?.let {
            cfg = apply(cfg, parse(it))
            changed = true
        }
    }
    set("--line", String::toFloat) { c, v -> c.copy(counter = c.counter.copy(lineX = v)) }
    set("--hysteresis", String::toFloat) { c, v -> c.copy(counter = c.counter.copy(hysteresis = v)) }
    set("--min-hits", String::toInt) { c, v ->
        c.copy(counter = c.counter.copy(minHits = v), tracker = c.tracker.copy(minHitsToConfirm = v))
    }
    set("--high", String::toFloat) { c, v -> c.copy(tracker = c.tracker.copy(highThreshold = v)) }
    set("--low", String::toFloat) { c, v -> c.copy(tracker = c.tracker.copy(lowThreshold = v)) }
    set("--new", String::toFloat) { c, v -> c.copy(tracker = c.tracker.copy(newTrackThreshold = v)) }
    set("--lost-ms", String::toLong) { c, v -> c.copy(tracker = c.tracker.copy(lostTimeoutMillis = v)) }
    set("--size-filter", String::toFloat) { c, v -> c.copy(counter = c.counter.copy(relativeSizeFilter = v)) }
    set("--roi", { s ->
        val p = s.split(',').map { it.trim().toFloat() }
        require(p.size == 4) { "--roi needs l,t,r,b" }
        Box(p[0], p[1], p[2], p[3])
    }) { c, v -> c.copy(roi = v) }
    return if (changed) cfg else null
}

private fun load(path: String): ParsedLog {
    val log = File(path).bufferedReader().use { DetectionLog.parse(it) }
    log.errors.take(5).forEach { System.err.println("warning: $path $it") }
    return log
}

private fun replay(opts: Options): Int {
    val path = opts.positional.firstOrNull() ?: run {
        System.err.println("replay needs a log file")
        return 2
    }
    val log = load(path)
    val override = overrides(opts, log.header?.pipeline ?: PipelineConfig())
    val result = Replay.run(log, override) { snap ->
        if (opts.has("--trace")) {
            val tracks = snap.tracks.joinToString(" ") { t ->
                "#${t.id}:${t.state.name.take(1)}%.2f%s".format(t.box.centerX, if (t.counted) "*" else "")
            }
            println("t=%.2fs n=%d dets=%d %s".format(snap.timestampNanos / 1e9, snap.netCount, snap.acceptedDetections.size, tracks))
        }
    }
    val expected = opts["--expected"]?.toInt() ?: result.expectedCount
    println("log:        $path")
    log.header?.let { h -> println("source:     ${h.source}  detector: ${h.detector?.name ?: "?"}  created: ${h.createdAt ?: "?"}") }
    println("frames:     ${result.frames} (%.1f s, %.1f detector fps), detections: ${result.detections}".format(result.durationSeconds, result.detectorFps))
    println("count:      ${result.count} (net ${result.netCount})" + (expected?.let { e -> "   expected: $e  ${if (e == result.count) "OK" else "MISMATCH (${result.count - e})"}" } ?: ""))
    result.recordedCount?.let { rc ->
        val note = if (override == null) (if (rc == result.count) "reproduced" else "NOT reproduced") else "parameters overridden"
        println("app count:  $rc ($note)")
    }
    val c = result.stats.counter
    println("crossings:  L->R ${c.leftToRight}, R->L ${c.rightToLeft}; stitches ${c.stitches}; delayed ${c.delayedCrossings}; " +
        "dropped pending ${c.discardedPendingCrossings}; late tracks ${c.lateTracks}; tracks created ${result.stats.tracksCreated}")
    if (opts.has("--events")) {
        for (e in result.events) {
            println("  t=%.2fs track #%d %s net=%d%s".format(
                e.timestampNanos / 1e9, e.trackId, if (e.direction > 0) "L->R" else "R->L", e.netCountAfter,
                e.stitchedFrom?.let { " (stitched from #$it)" } ?: "",
            ))
        }
    }
    return if (expected != null && expected != result.count) 1 else 0
}

private fun evaluate(opts: Options): Int {
    val target = opts.positional.firstOrNull()?.let(::File) ?: run {
        System.err.println("eval needs a directory or manifest.csv")
        return 2
    }
    val entries: List<Pair<File, Int?>> = if (target.isDirectory) {
        target.listFiles { f -> f.extension == "jsonl" }.orEmpty().sortedBy { it.name }.map { it to null }
    } else {
        target.readLines().filter { it.isNotBlank() && !it.startsWith("#") && !it.startsWith("path") }.map { line ->
            val parts = line.split(',')
            val f = File(parts[0].trim()).let { if (it.isAbsolute) it else File(target.parentFile, parts[0].trim()) }
            f to parts.getOrNull(1)?.trim()?.toIntOrNull()
        }
    }
    if (entries.isEmpty()) {
        System.err.println("no logs found")
        return 2
    }
    val rows = ArrayList<Pair<String, ReplayResult>>()
    println("%-40s %8s %8s %6s".format("log", "expected", "counted", "diff"))
    var mismatches = 0
    var absErr = 0
    var withTruth = 0
    for ((file, csvExpected) in entries) {
        val log = load(file.path)
        val result = Replay.run(log, overrides(opts, log.header?.pipeline ?: PipelineConfig()))
        val expected = csvExpected ?: result.expectedCount
        rows += file.name to result
        val diff = expected?.let { result.count - it }
        if (diff != null) {
            withTruth++
            absErr += abs(diff)
            if (diff != 0) mismatches++
        }
        println("%-40s %8s %8d %6s".format(file.name.take(40), expected?.toString() ?: "-", result.count, diff?.let { if (it > 0) "+$it" else "$it" } ?: "-"))
    }
    if (withTruth > 0) {
        println("\n$withTruth logs with ground truth: ${withTruth - mismatches} exact, $mismatches wrong, " +
            "mean abs error %.2f pallets".format(absErr.toDouble() / withTruth))
    }
    return if (mismatches > 0) 1 else 0
}

private fun simulate(opts: Options): Int {
    val out = opts["--out"] ?: run {
        System.err.println("simulate needs --out <file.jsonl>")
        return 2
    }
    val pallets = opts["--pallets"]?.toInt() ?: 16
    val seed = opts["--seed"]?.toLong() ?: 1L
    val base = SimConfig(palletCount = pallets, seed = seed)
    val scenario = opts["--scenario"] ?: "steady"
    val length = base.firstPalletM + (pallets - 1) * base.palletPitchM + 2.5
    val scene = when (scenario) {
        "steady" -> SyntheticLineScene(base)
        "noisy" -> SyntheticLineScene(base.copy(missProbability = 0.2, lowConfidenceProbability = 0.2, falsePositivesPerFrame = 0.2, jitter = 0.01f))
        "reverse" -> SyntheticLineScene(base, listOf(Segment(length / 0.8 / 2, 0.8), Segment(4.0, -0.6), Segment(length / 0.8, 0.8)))
        "jerks" -> SyntheticLineScene(base, listOf(Segment(6.0, 0.8), Segment(0.1, 3.5), Segment(length / 0.8, 0.8)))
        "blackout" -> SyntheticLineScene(base.copy(blackouts = listOf(Blackout(5.0, 5.7), Blackout(11.0, 11.6))))
        else -> {
            System.err.println("unknown scenario $scenario")
            return 2
        }
    }
    File(out).bufferedWriter().use { w ->
        val writer = DetectionLogWriter(w)
        writer.header(
            LogHeader(
                source = "simulator",
                expectedCount = scene.expectedCount(),
                notes = "synthetic scenario '$scenario' (not real detections)",
            ),
        )
        scene.frames().forEach(writer::frame)
    }
    println("wrote $out (${scene.frames().size} frames, expected count ${scene.expectedCount()})")
    return 0
}
