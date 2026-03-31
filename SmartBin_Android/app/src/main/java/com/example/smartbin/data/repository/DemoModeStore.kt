package com.example.smartbin.data.repository

import com.example.smartbin.BuildConfig
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
class DemoModeStore @Inject constructor() {
    private val _mode = MutableStateFlow(if (BuildConfig.DEFAULT_DEMO_MODE) AppMode.MOCK else AppMode.LIVE)
    val mode: StateFlow<AppMode> = _mode.asStateFlow()

    fun setMode(mode: AppMode) {
        _mode.value = mode
    }

    fun toggleMode() {
        _mode.update { current ->
            if (current == AppMode.MOCK) AppMode.LIVE else AppMode.MOCK
        }
    }
}
