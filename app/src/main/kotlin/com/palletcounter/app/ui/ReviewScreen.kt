package com.palletcounter.app.ui

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.palletcounter.app.scan.ScanResult
import com.palletcounter.core.session.CountResult
import com.palletcounter.core.session.LineMode
import com.palletcounter.core.session.StackSize

@Composable
fun ReviewScreen(vm: AppViewModel, scan: ScanResult) {
    var result by remember { mutableStateOf(CountResult(scan.detectedCount, 0, scan.settings)) }
    val simulated = scan.detector?.isSimulation == true
    BackHandler { vm.navigate(Screen.Setup) }

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp)) {
        if (simulated) WarningBanner("SIMULATED DETECTIONS — this is not a real count")
        Text("Review", fontSize = 24.sp, fontWeight = FontWeight.Bold)
        scan.expectedCount?.let { expected ->
            val ok = expected == scan.detectedCount
            Text(
                "Expected $expected, app counted ${scan.detectedCount} ${if (ok) "✓" else "(${scan.detectedCount - expected})"}",
                color = if (ok) Palette.Counted else Palette.Danger,
                fontWeight = FontWeight.Bold,
            )
        }
        Row(Modifier.fillMaxWidth().padding(top = 12.dp), horizontalArrangement = Arrangement.SpaceEvenly) {
            BigStat("EURO pallets", "${result.pallets}", valueColor = Palette.Accent)
            BigStat("Half-pallets", "${result.halfPallets}")
        }
        Text(
            "${result.formula}" + if (result.manualAdjustment != 0) "   (detected ${scan.detectedCount}, manual ${"%+d".format(result.manualAdjustment)})" else "",
            modifier = Modifier.align(Alignment.CenterHorizontally),
            color = Color.LightGray,
        )
        Row(Modifier.fillMaxWidth().padding(vertical = 12.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { result = result.adjusted(-1) }, Modifier.weight(1f).height(64.dp)) { Text("−1 pallet", fontSize = 18.sp) }
            Button(onClick = { result = result.adjusted(+1) }, Modifier.weight(1f).height(64.dp)) { Text("+1 pallet", fontSize = 18.sp) }
        }
        SectionTitle("Stack size")
        ChoiceRow(StackSize.entries, result.settings.stackSize, { "${it.halfPallets}" }, { result = result.withSettings(result.settings.copy(stackSize = it)) })
        SectionTitle("Line mode")
        ChoiceRow(LineMode.entries, result.settings.lineMode, { it.label }, { result = result.withSettings(result.settings.copy(lineMode = it)) })

        Row(Modifier.fillMaxWidth().padding(top = 20.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(
                onClick = { vm.navigate(if (scan.source == "video") Screen.VideoReplay else Screen.Scan) },
                Modifier.weight(1f).height(56.dp),
            ) { Text("Rescan") }
            Button(
                onClick = { vm.accept(result, scan.source, scan.detector?.name ?: "?", simulated) },
                Modifier.weight(1f).height(56.dp),
            ) { Text("Accept", fontWeight = FontWeight.Bold) }
        }

        if (scan.pallets.isNotEmpty()) {
            SectionTitle("Counted pallets (${scan.pallets.size}) — check for duplicates / misses")
            LazyRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                items(scan.pallets) { p ->
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        p.thumbnail?.let {
                            Image(it.asImageBitmap(), contentDescription = "pallet ${p.number}", Modifier.size(110.dp), contentScale = ContentScale.Fit)
                        }
                        Text("#${p.number} (track ${p.trackId})", fontSize = 12.sp)
                    }
                }
            }
        }
        scan.lastCountFrame?.let { frame ->
            SectionTitle("Frame at last count")
            Box(Modifier.fillMaxWidth().aspectRatio(frame.width.toFloat() / frame.height)) {
                Image(frame.asImageBitmap(), contentDescription = "last counted frame", Modifier.fillMaxSize(), contentScale = ContentScale.Fit)
                scan.lastCountSnapshot?.let { snap ->
                    DetectionOverlay(snap, snap.config, frame.width.toFloat() / frame.height, OverlayOptions(debug = false, showGuideText = false), Modifier.fillMaxSize())
                }
            }
        }
        SectionTitle("Scan details")
        val c = scan.lastSnapshot?.stats?.counter
        Text(
            buildString {
                append("Detector: ${scan.detector?.name ?: "?"} (${scan.detector?.accelerator ?: "?"})\n")
                append("Duration %.1f s, %d detector frames (%.1f fps)\n".format(scan.durationSeconds, scan.frames, if (scan.durationSeconds > 0) scan.frames / scan.durationSeconds else 0.0))
                if (c != null) {
                    append("Crossings L→R ${c.leftToRight}, R→L ${c.rightToLeft} (net ${scan.netCount})\n")
                    append("Stitched tracks ${c.stitches}, delayed crossings ${c.delayedCrossings}, dropped ${c.discardedPendingCrossings}\n")
                    append("Tracks first seen past the line (possible misses): ${c.lateTracks}\n")
                }
                scan.logFile?.let { append("Detection log: ${it.name}") }
            },
            style = MaterialTheme.typography.bodySmall,
            color = Color.LightGray,
        )
        Spacer(Modifier.height(24.dp))
    }
}
