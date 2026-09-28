package com.palletcounter.app.ui

import android.app.Application
import android.net.Uri
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.viewModelScope
import com.palletcounter.app.data.DetectorMode
import com.palletcounter.app.data.HistoryEntry
import com.palletcounter.app.data.HistoryRepository
import com.palletcounter.app.data.SettingsRepository
import com.palletcounter.app.debug.CaptureStore
import com.palletcounter.app.debug.LogStore
import com.palletcounter.app.detection.ModelDescription
import com.palletcounter.app.detection.ModelManager
import com.palletcounter.app.scan.ScanEngine
import com.palletcounter.app.scan.ScanResult
import com.palletcounter.core.session.CountResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.OutputStream

sealed interface Screen {
    data object Setup : Screen
    data object Scan : Screen
    data class Review(val result: ScanResult) : Screen
    data object Photo : Screen
    data object VideoReplay : Screen
    data object Settings : Screen
}

/**
 * Screens that are restored after the process was killed in the background (e.g. while the
 * system camera or file picker was open, so its result still reaches the screen). A scan or
 * a review cannot be restored and falls back to the setup screen.
 */
private fun Screen.saveKey(): String = when (this) {
    Screen.Photo -> "photo"
    Screen.VideoReplay -> "video"
    Screen.Settings -> "settings"
    else -> "setup"
}

private fun screenFromKey(key: String?): Screen = when (key) {
    "photo" -> Screen.Photo
    "video" -> Screen.VideoReplay
    "settings" -> Screen.Settings
    else -> Screen.Setup
}

class AppViewModel(app: Application, private val savedState: SavedStateHandle) : AndroidViewModel(app) {
    val settings = SettingsRepository(app)
    val history = HistoryRepository(app)
    val models = ModelManager(app)
    val captures = CaptureStore(app)
    val logs = LogStore(app)

    private var currentScreen by mutableStateOf(screenFromKey(savedState[SCREEN_KEY]))
    var screen: Screen
        get() = currentScreen
        private set(value) {
            currentScreen = value
            savedState[SCREEN_KEY] = value.saveKey()
        }

    var modelDescription by mutableStateOf<ModelDescription?>(null)
        private set
    var modelStatus by mutableStateOf("Checking model…")
        private set

    /** Live camera engine, created when the scan screen opens. */
    var engine: ScanEngine? = null
        private set

    private var finishingEngine by mutableStateOf<ScanEngine?>(null)

    /** True between FINISH and the review screen (the last frame is still being processed). */
    val finishing: Boolean get() = finishingEngine.let { it != null && it === engine }

    var toast by mutableStateOf<String?>(null)

    init {
        refreshModel()
    }

    fun navigate(target: Screen) {
        if (screen == Screen.Scan && target != Screen.Scan) discardEngine()
        screen = target
    }

    fun refreshModel() {
        viewModelScope.launch {
            val (desc, status) = withContext(Dispatchers.IO) {
                runCatching {
                    val source = models.activeSource()
                    if (source == null) {
                        null to "No model installed"
                    } else {
                        val d = models.describe(source)
                        d to "${d.sidecar.name} (${source.displayName})"
                    }
                }.getOrElse { null to "Model error: ${it.message}" }
            }
            modelDescription = desc
            modelStatus = status
        }
    }

    fun startLiveScan(): ScanEngine {
        discardEngine()
        val e = ScanEngine(
            context = getApplication(),
            models = models,
            settings = settings.settings.value,
            source = "live",
            logStore = logs,
            captures = captures,
        )
        e.start()
        engine = e
        return e
    }

    fun finishLiveScan() {
        val e = engine ?: return
        if (finishingEngine === e) return // FINISH tapped twice
        finishingEngine = e
        viewModelScope.launch {
            val result = withContext(Dispatchers.Default) { runCatching { e.finish() } }
            e.close()
            if (finishingEngine === e) finishingEngine = null
            // The user may have left the scan screen meanwhile (back): then the scan is discarded.
            if (engine !== e) return@launch
            engine = null
            result.onSuccess { screen = Screen.Review(it) }
                .onFailure {
                    toast = "Could not finish scan: ${it.message}"
                    screen = Screen.Setup
                }
        }
    }

    fun showReview(result: ScanResult) {
        screen = Screen.Review(result)
    }

    private fun discardEngine() {
        engine?.close()
        engine = null
    }

    fun accept(result: CountResult, source: String, modelName: String, simulated: Boolean) {
        history.add(HistoryEntry(System.currentTimeMillis(), result, source, modelName, simulated))
        toast = "Saved: ${result.formula} half-pallets"
        screen = Screen.Setup
    }

    fun importModel(uri: Uri) {
        viewModelScope.launch {
            val msg = withContext(Dispatchers.IO) {
                runCatching { models.importModel(uri) }.fold(
                    { "Imported model (${it.sizeBytes / 1024} KB). Import its .json sidecar too if you have one." },
                    { "Import failed: ${it.message}" },
                )
            }
            if (!msg.startsWith("Import failed")) settings.update { it.copy(detectorMode = DetectorMode.MODEL) }
            toast = msg
            refreshModel()
        }
    }

    fun importSidecar(uri: Uri) {
        viewModelScope.launch {
            toast = withContext(Dispatchers.IO) {
                runCatching { models.importSidecar(uri) }.fold({ "Sidecar for '${it.name}' imported" }, { "Invalid sidecar: ${it.message}" })
            }
            refreshModel()
        }
    }

    /** Writes a ZIP of captures or logs to a user-chosen document, off the main thread. */
    fun export(uri: Uri, what: String, write: (OutputStream) -> Unit) {
        viewModelScope.launch {
            toast = withContext(Dispatchers.IO) {
                runCatching {
                    val out = checkNotNull(getApplication<Application>().contentResolver.openOutputStream(uri)) { "cannot open the file" }
                    out.use(write)
                }.fold({ "$what exported" }, { "Export failed: ${it.message}" })
            }
        }
    }

    fun removeImportedModel() {
        models.removeImported()
        toast = "Imported model removed"
        refreshModel()
    }

    override fun onCleared() {
        discardEngine()
    }

    private companion object {
        const val SCREEN_KEY = "screen"
    }
}
