package com.palletcounter.app

import android.Manifest
import android.content.res.Configuration
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.v2.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.rule.GrantPermissionRule
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Whole app on an emulator: switch to the SIMULATION detector, start a live sweep with the
 * emulated camera (the scan screen turns to landscape first), wait until pallets are
 * counted, finish, check the review screen, accept.
 * Exercises CameraX → analyzer → scan engine → pipeline → UI. (The simulated detector ignores
 * image content; real detection quality needs a trained model and real footage.)
 */
@RunWith(AndroidJUnit4::class)
class AppFlowInstrumentedTest {
    @get:Rule(order = 0)
    val permission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    // v2 rule: coroutines are queued on the test dispatcher and run on the UI thread. The v1
    // rule runs them eagerly on whichever thread resumes them (e.g. the scan engine's worker
    // via state-flow updates); one CI run with v1 crashed with a layout re-entrance error,
    // most likely from that. In the app itself composition coroutines run on the main thread.
    @get:Rule(order = 1)
    val compose = createAndroidComposeRule<MainActivity>()

    private fun countOnScreen(): Int? = compose.onAllNodes(hasText(" EUR", substring = true))
        .fetchSemanticsNodes()
        .flatMap { it.config.getOrElse(SemanticsProperties.Text) { emptyList() } }
        .firstNotNullOfOrNull { Regex("^(\\d+) EUR$").find(it.text)?.groupValues?.get(1)?.toInt() }

    private fun orientation() = compose.activity.resources.configuration.orientation

    @Test
    fun simulatedSweepEndToEnd() {
        val initialOrientation = orientation()
        compose.onNodeWithText("Settings").performClick()
        compose.onNodeWithText("SIMULATION").performClick()
        compose.onNodeWithText("Done").performClick()
        compose.onNodeWithText("START VIDEO SWEEP").assertExists().performClick()

        // Camera frames drive the simulated walk; the first pallet crosses after ~4 s.
        compose.waitUntil(timeoutMillis = 45_000) { (countOnScreen() ?: 0) >= 2 }
        // Default scan orientation; the camera was bound after the display turned.
        assertEquals(Configuration.ORIENTATION_LANDSCAPE, orientation())
        // Banner and HUD both mention the simulation detector.
        assertTrue(compose.onAllNodes(hasText("SIMULATION", substring = true)).fetchSemanticsNodes().isNotEmpty())

        compose.onNodeWithText("FINISH").performClick()
        compose.waitUntil(timeoutMillis = 15_000) {
            compose.onAllNodes(hasText("Review")).fetchSemanticsNodes().isNotEmpty()
        }
        // Leaving the scan screen restores the previous orientation; let it settle before tapping.
        compose.waitUntil(timeoutMillis = 10_000) { orientation() == initialOrientation }
        compose.waitForIdle()
        compose.onNodeWithText("SIMULATED DETECTIONS", substring = true).assertExists()
        compose.onNodeWithText("+1 pallet").performClick()
        compose.onNodeWithText("Accept").performScrollTo().performClick()
        compose.waitUntil(timeoutMillis = 10_000) {
            compose.onAllNodes(hasText("START VIDEO SWEEP")).fetchSemanticsNodes().isNotEmpty()
        }
        // The accepted count is in the history, flagged as simulated.
        assertTrue(compose.onAllNodes(hasText("SIMULATED", substring = true)).fetchSemanticsNodes().isNotEmpty())
    }
}
