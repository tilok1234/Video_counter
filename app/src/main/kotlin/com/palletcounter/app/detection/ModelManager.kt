package com.palletcounter.app.detection

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import com.palletcounter.app.data.AppSettings
import com.palletcounter.app.data.DetectorMode
import com.palletcounter.core.detection.ModelSidecar
import com.palletcounter.core.detection.PalletDetector
import java.io.File
import java.io.FileInputStream
import java.io.RandomAccessFile
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.security.MessageDigest

/** Where the active model comes from. */
sealed interface ModelSource {
    val displayName: String

    /** Built into the APK: app/src/main/assets/models/eur_pallet.tflite (+ .json). */
    data class Bundled(val assetPath: String, val sidecarAssetPath: String) : ModelSource {
        override val displayName get() = "bundled: $assetPath"
    }

    /** Imported on the phone through Settings → Import model. */
    data class Imported(val file: File, val sidecarFile: File, val originalName: String) : ModelSource {
        override val displayName get() = "imported: $originalName"
    }
}

data class ModelDescription(
    val source: ModelSource,
    val sidecar: ModelSidecar,
    val sizeBytes: Long,
    val sha256Prefix: String,
)

/**
 * Finds, imports and loads detector models. Imported models take priority over the bundled
 * one so a newly trained model can be tried without rebuilding the app.
 */
class ModelManager(private val context: Context) {
    private val dir = File(context.filesDir, "models").apply { mkdirs() }
    private val importedModel = File(dir, "active.tflite")
    private val importedSidecar = File(dir, "active.json")
    private val importedName = File(dir, "active.name")

    fun activeSource(): ModelSource? = importedSource() ?: bundledSource()

    private fun importedSource(): ModelSource.Imported? =
        if (importedModel.exists()) {
            ModelSource.Imported(importedModel, importedSidecar, importedName.takeIf { it.exists() }?.readText() ?: "model.tflite")
        } else {
            null
        }

    private fun bundledSource(): ModelSource.Bundled? {
        val files = context.assets.list(ASSET_DIR)?.toSet() ?: return null
        return if (BUNDLED_MODEL in files) {
            ModelSource.Bundled("$ASSET_DIR/$BUNDLED_MODEL", "$ASSET_DIR/$BUNDLED_SIDECAR")
        } else {
            null
        }
    }

    fun describe(source: ModelSource): ModelDescription {
        val bytes = when (source) {
            is ModelSource.Bundled -> context.assets.open(source.assetPath).use { it.readBytes() }
            is ModelSource.Imported -> source.file.readBytes()
        }
        val digest = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
        return ModelDescription(source, sidecar(source), bytes.size.toLong(), digest.take(12))
    }

    fun sidecar(source: ModelSource): ModelSidecar {
        val text = when (source) {
            is ModelSource.Bundled -> runCatching { context.assets.open(source.sidecarAssetPath).use { it.readBytes().decodeToString() } }.getOrNull()
            is ModelSource.Imported -> source.sidecarFile.takeIf { it.exists() }?.readText()
        }
        val fallbackName = when (source) {
            is ModelSource.Bundled -> source.assetPath.substringAfterLast('/')
            is ModelSource.Imported -> source.originalName
        }
        return text?.let { runCatching { ModelSidecar.parse(it) }.getOrNull() } ?: ModelSidecar(name = fallbackName)
    }

    fun map(source: ModelSource): MappedByteBuffer = when (source) {
        is ModelSource.Bundled -> context.assets.openFd(source.assetPath).use { fd ->
            FileInputStream(fd.fileDescriptor).channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
        is ModelSource.Imported -> RandomAccessFile(source.file, "r").use { raf ->
            raf.channel.map(FileChannel.MapMode.READ_ONLY, 0, raf.length())
        }
    }

    /**
     * Creates the detector selected in [settings]. Call on the worker thread that will run
     * inference (GPU delegates are thread-bound).
     *
     * @throws NoModelException if the model mode is selected but no model is installed.
     */
    fun createDetector(settings: AppSettings, decoderThreshold: Float): PalletDetector<FrameInput> {
        if (settings.detectorMode == DetectorMode.SIMULATION) return SimulatedDetector()
        val source = activeSource() ?: throw NoModelException()
        return TfliteYoloDetector(map(source), sidecar(source), settings.accelerator, settings.cpuThreads, decoderThreshold)
    }

    /** Copies a user-picked .tflite into app storage and validates it by loading it. */
    fun importModel(uri: Uri): ModelDescription {
        val name = displayName(uri) ?: "model.tflite"
        val tmp = File(dir, "incoming.tflite")
        context.contentResolver.openInputStream(uri)?.use { input -> tmp.outputStream().use { input.copyTo(it) } }
            ?: throw IllegalArgumentException("Cannot read $uri")
        // Validate: the interpreter must accept it on the CPU.
        RandomAccessFile(tmp, "r").use { raf ->
            val buf = raf.channel.map(FileChannel.MapMode.READ_ONLY, 0, raf.length())
            org.tensorflow.lite.Interpreter(buf).close()
        }
        tmp.copyTo(importedModel, overwrite = true)
        tmp.delete()
        importedName.writeText(name)
        // A sidecar from a previous model would describe the wrong tensors; import the new
        // model's .json afterwards (without one, the layout is detected automatically).
        importedSidecar.delete()
        return describe(activeSource()!!)
    }

    fun importSidecar(uri: Uri): ModelSidecar {
        val text = context.contentResolver.openInputStream(uri)?.use { it.readBytes().decodeToString() }
            ?: throw IllegalArgumentException("Cannot read $uri")
        val parsed = ModelSidecar.parse(text) // throws on invalid JSON
        importedSidecar.writeText(text)
        return parsed
    }

    fun removeImported() {
        importedModel.delete()
        importedSidecar.delete()
        importedName.delete()
    }

    private fun displayName(uri: Uri): String? =
        context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) c.getString(0) else null
        }

    companion object {
        const val ASSET_DIR = "models"
        const val BUNDLED_MODEL = "eur_pallet.tflite"
        const val BUNDLED_SIDECAR = "eur_pallet.json"
    }
}

class NoModelException : Exception(
    "No detector model installed. Train one (docs/MODEL_TRAINING.md), then import it in Settings, " +
        "or switch the detector to SIMULATION to test the plumbing.",
)
