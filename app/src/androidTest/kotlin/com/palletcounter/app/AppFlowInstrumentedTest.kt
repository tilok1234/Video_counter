package com.palletcounter.app

import android.Manifest
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.rule.GrantPermissionRule
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Whole app on an emulator: switch to the SIMULATION detector, start a live sweep with the
 * emulated camera, wait until pallets are counted, finish, check the review screen, accept.
 * Exercises CameraX → analyzer → scan engine → pipeline → UI. (The simulated detector ignores
 * image content; real detection quality needs a trained model and real footage.)
 */
@RunWith(AndroidJUnit4::class)
class AppFlowInstrumentedTest {
    @get:Rule(order = 0)
    val permission: GrantPermissionRule = GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @get:Rule(order = 1)
    val compose = createAndroidComposeRule<MainActivity>()

    private fun countOnScreen(): Int? = compose.onAllNodes(androidx.compose.ui.test.hasText(" EUR", substring = true))
        .fetchSemanticsNodes()
        .flatMap { it.config.getOrElse(SemanticsProperties.Text) { emptyList() } }
        .firstNotNullOfOrNull { Regex("^(\\d+) EUR$").find(it.text)?.groupValues?.get(1)?.toInt() }

    @Test
    fun simulatedSweepEndToEnd() {
        compose.onNodeWithText("Settings").performClick()
        compose.onNodeWithText("SIMULATION").performClick()
        compose.onNodeWithText("Done").performClick()
        compose.onNodeWithText("START VIDEO SWEEP").assertExists().performClick()

        // Camera frames drive the simulated walk; the first pallet crosses after ~4 s.
        compose.waitUntil(timeoutMillis = 45_000) { (countOnScreen() ?: 0) >= 2 }
        compose.onNodeWithText("SIMULATION", substring = true).assertExists()

        compose.onNodeWithText("FINISH").performClick()
        compose.waitUntil(timeoutMillis = 15_000) {
            compose.onAllNodes(androidx.compose.ui.test.hasText("Review")).fetchSemanticsNodes().isNotEmpty()
        }
        compose.onNodeWithText("SIMULATED DETECTIONS", substring = true).assertExists()
        compose.onNodeWithText("+1 pallet").performClick()
        compose.onNodeWithText("Accept").performScrollTo().performClick()
        compose.waitUntil(timeoutMillis = 10_000) {
            compose.onAllNodes(androidx.compose.ui.test.hasText("START VIDEO SWEEP")).fetchSemanticsNodes().isNotEmpty()
        }
        compose.onNodeWithText("SIMULATED", substring = true).assertExists()
    }
}
