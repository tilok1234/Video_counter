package com.palletcounter.app.ui

import android.app.Application
import android.net.Uri
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
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

sealed interface Screen {
    data object Setup : Screen
    data object Scan : Screen
    data class Review(val result: ScanResult) : Screen
    data object Photo : Screen
    data object VideoReplay : Screen
    data object Settings : Screen
}

class AppViewModel(app: Application) : AndroidViewModel(app) {
    val settings = SettingsRepository(app)
    val history = HistoryRepository(app)
    val models = ModelManager(app)
    val captures = CaptureStore(app)
    val logs = LogStore(app)

    var screen by mutableStateOf<Screen>(Screen.Setup)
        private set

    var modelDescription by mutableStateOf<ModelDescription?>(null)
        private set
    var modelStatus by mutableStateOf("Checking model…")
        private set

    /** Live camera engine, created when the scan screen opens. */
    var engine: ScanEngine? = null
        private set

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
        viewModelScope.launch {
            val result = withContext(Dispatchers.Default) { runCatching { e.finish() } }
            e.close()
            if (engine === e) engine = null
            result.onSuccess { screen = Screen.Review(it) }
                .onFailure { toast = "Could not finish scan: ${it.message}" }
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

    fun removeImportedModel() {
        models.removeImported()
        toast = "Imported model removed"
        refreshModel()
    }

    override fun onCleared() {
        discardEngine()
    }
}
