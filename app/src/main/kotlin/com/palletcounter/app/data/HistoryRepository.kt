package com.palletcounter.app.data

import android.content.Context
import com.palletcounter.core.session.CountResult
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.serialization.Serializable
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json
import java.io.File

@Serializable
data class HistoryEntry(
    val timestampMillis: Long,
    val result: CountResult,
    /** "live", "video" or "photo". */
    val source: String,
    val modelName: String,
    val simulated: Boolean,
    val note: String = "",
)

/** Accepted counts, stored locally as JSON (nothing is uploaded). */
class HistoryRepository(context: Context) {
    private val file = File(context.filesDir, "history.json")
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }
    private val serializer = ListSerializer(HistoryEntry.serializer())
    private val _entries = MutableStateFlow(load())
    val entries: StateFlow<List<HistoryEntry>> = _entries.asStateFlow()

    private fun load(): List<HistoryEntry> =
        if (file.exists()) runCatching { json.decodeFromString(serializer, file.readText()) }.getOrDefault(emptyList()) else emptyList()

    @Synchronized
    fun add(entry: HistoryEntry) = save(listOf(entry) + _entries.value)

    @Synchronized
    fun remove(entry: HistoryEntry) = save(_entries.value - entry)

    @Synchronized
    fun clear() = save(emptyList())

    private fun save(list: List<HistoryEntry>) {
        _entries.value = list
        val tmp = File(file.parentFile, file.name + ".tmp")
        tmp.writeText(json.encodeToString(serializer, list))
        tmp.renameTo(file)
    }
}
