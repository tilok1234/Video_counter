package com.palletcounter.app.data

import android.content.Context
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.serialization.json.Json

/** Settings persisted in SharedPreferences as a single JSON document. */
class SettingsRepository(context: Context) {
    private val prefs = context.getSharedPreferences("pallet_counter", Context.MODE_PRIVATE)
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }
    private val _settings = MutableStateFlow(load())
    val settings: StateFlow<AppSettings> = _settings.asStateFlow()

    private fun load(): AppSettings = prefs.getString(KEY, null)?.let {
        runCatching { json.decodeFromString(AppSettings.serializer(), it) }.getOrNull()
    } ?: AppSettings()

    fun update(transform: (AppSettings) -> AppSettings) {
        val next = transform(_settings.value)
        _settings.value = next
        prefs.edit().putString(KEY, json.encodeToString(AppSettings.serializer(), next)).apply()
    }

    fun resetToDefaults() = update { AppSettings(stackSize = it.stackSize, lineMode = it.lineMode) }

    private companion object {
        const val KEY = "settings_v1"
    }
}
