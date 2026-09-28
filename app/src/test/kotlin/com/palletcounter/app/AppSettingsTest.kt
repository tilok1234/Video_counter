package com.palletcounter.app

import com.palletcounter.app.data.AppSettings
import com.palletcounter.app.data.RoiPreset
import com.palletcounter.core.session.LineMode
import com.palletcounter.core.session.StackSize
import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AppSettingsTest {
    @Test
    fun presetsMapToRoi() {
        assertEquals(0.25f, AppSettings(roiPreset = RoiPreset.CENTER_BAND).roi().top, 1e-6f)
        val full = AppSettings(roiPreset = RoiPreset.FULL_HEIGHT).roi()
        assertEquals(0f, full.top, 1e-6f)
        assertEquals(1f, full.bottom, 1e-6f)
        val custom = AppSettings(roiPreset = RoiPreset.CUSTOM, customRoiTop = 0.4f, customRoiBottom = 0.9f).roi()
        assertEquals(0.4f, custom.top, 1e-6f)
        assertEquals(0.9f, custom.bottom, 1e-6f)
    }

    @Test
    fun invertedCustomBandIsRepaired() {
        val roi = AppSettings(roiPreset = RoiPreset.CUSTOM, customRoiTop = 0.8f, customRoiBottom = 0.3f).roi()
        assertTrue(roi.bottom > roi.top)
    }

    @Test
    fun pipelineConfigUsesSettings() {
        val s = AppSettings(lineX = 0.4f, hysteresis = 0.05f, minHits = 2, highThreshold = 0.6f, newTrackThreshold = 0.5f, sizeFilter = 0.8f)
        val p = s.pipelineConfig()
        assertEquals(0.4f, p.counter.lineX, 1e-6f)
        assertEquals(0.05f, p.counter.hysteresis, 1e-6f)
        assertEquals(2, p.counter.minHits)
        assertEquals(2, p.tracker.minHitsToConfirm)
        assertEquals(0.8f, p.counter.relativeSizeFilter, 1e-6f)
        // A new track can never need less confidence than the high threshold.
        assertEquals(0.6f, p.tracker.newTrackThreshold, 1e-6f)
    }

    @Test
    fun settingsSurviveJsonRoundTrip() {
        val json = Json { encodeDefaults = true; ignoreUnknownKeys = true }
        val s = AppSettings(stackSize = StackSize.TWENTY, lineMode = LineMode.WHOLE_LINE_X1, roiPreset = RoiPreset.LOW_BAND)
        val back = json.decodeFromString(AppSettings.serializer(), json.encodeToString(AppSettings.serializer(), s))
        assertEquals(s, back)
        // Unknown keys from a future version are ignored.
        assertEquals(AppSettings(), json.decodeFromString(AppSettings.serializer(), "{\"futureOption\":1}"))
    }
}
