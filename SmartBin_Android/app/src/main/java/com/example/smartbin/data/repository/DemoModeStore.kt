package com.example.smartbin.data.repository

import android.content.Context
import com.example.smartbin.BuildConfig
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

enum class AppMode(val label: String) {
    MOCK("Mock"),
    LIVE("Live Server"),
}

@Singleton
class DemoModeStore @Inject constructor(
    @ApplicationContext context: Context,
) {
    private val preferences = context.getSharedPreferences("smartbin_demo_prefs", Context.MODE_PRIVATE)
    private val _mode = MutableStateFlow(
        preferences.getString(KEY_MODE, null)
            ?.let { saved -> AppMode.entries.find { it.name == saved } }
            ?: if (BuildConfig.DEFAULT_DEMO_MODE) AppMode.MOCK else AppMode.LIVE,
    )
    private val _watchedBinId = MutableStateFlow(preferences.getString(KEY_WATCHED_BIN_ID, null))
    val mode: StateFlow<AppMode> = _mode.asStateFlow()
    val watchedBinId: StateFlow<String?> = _watchedBinId.asStateFlow()

    fun setMode(mode: AppMode) {
        _mode.value = mode
        preferences.edit().putString(KEY_MODE, mode.name).apply()
    }

    fun setWatchedBinId(binId: String?) {
        _watchedBinId.value = binId
        preferences.edit().putString(KEY_WATCHED_BIN_ID, binId).apply()
    }

    fun toggleMode() {
        _mode.update { current ->
            val next = if (current == AppMode.MOCK) AppMode.LIVE else AppMode.MOCK
            preferences.edit().putString(KEY_MODE, next.name).apply()
            next
        }
    }

    companion object {
        private const val KEY_MODE = "app_mode"
        private const val KEY_WATCHED_BIN_ID = "watched_bin_id"
    }
}
