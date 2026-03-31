package com.example.smartbin.presentation.map

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.smartbin.data.repository.AppMode
import com.example.smartbin.data.repository.DemoModeStore
import com.example.smartbin.domain.model.Bin
import com.example.smartbin.domain.repository.BinRepository
import com.example.smartbin.domain.usecase.StreamWasteEventsUseCase
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.delay
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class MapState(
    val bins: List<Bin> = emptyList(),
    val isLoading: Boolean = false,
    val selectedBinIds: Set<String> = emptySet(),
    val selectedLocality: String? = null,
    val detailBinId: String? = null,
    val recentlyActiveBinId: String? = null,
    val streamConnected: Boolean = true,
    val errorMessage: String? = null,
    val appMode: AppMode = AppMode.MOCK,
) {
    val localities: List<String>
        get() = bins.map { it.locality }.distinct().sorted()

    val visibleBins: List<Bin>
        get() = bins.filter { selectedLocality == null || it.locality == selectedLocality }

    val selectedBins: List<Bin>
        get() = bins.filter { it.id in selectedBinIds }

    val detailBin: Bin?
        get() = bins.find { it.id == detailBinId }
}

@HiltViewModel
class MapViewModel @Inject constructor(
    private val binRepository: BinRepository,
    private val demoModeStore: DemoModeStore,
    private val streamWasteEventsUseCase: StreamWasteEventsUseCase,
) : ViewModel() {

    private val _state = MutableStateFlow(MapState())
    val state: StateFlow<MapState> = _state.asStateFlow()
    private var recentActivityResetJob: Job? = null

    init {
        loadBins()
        observeAppMode()
        observeStreamStatus()
        observeLiveEvents()
    }

    private fun loadBins() {
        _state.update { it.copy(isLoading = true) }
        viewModelScope.launch {
            binRepository.observeBins()
                .catch { error ->
                    _state.update {
                        it.copy(
                            isLoading = false,
                            errorMessage = error.message ?: "Unable to load bins",
                        )
                    }
                }
                .collect { bins ->
                _state.update { current ->
                    val retainedSelection = current.selectedBinIds.filterTo(linkedSetOf()) { selectedId ->
                        bins.any { it.id == selectedId }
                    }
                    current.copy(
                        bins = bins,
                        isLoading = false,
                        selectedBinIds = retainedSelection,
                        errorMessage = null,
                    )
                }
            }
        }
    }

    private fun observeLiveEvents() {
        viewModelScope.launch {
            streamWasteEventsUseCase()
                .catch { error ->
                    _state.update {
                        it.copy(
                            streamConnected = false,
                            errorMessage = error.message ?: "Realtime updates failed",
                        )
                    }
                }
                .collect { event ->
                _state.update {
                    it.copy(
                        recentlyActiveBinId = event.binId,
                        detailBinId = it.detailBinId ?: event.binId,
                        errorMessage = null,
                    )
                }
                recentActivityResetJob?.cancel()
                recentActivityResetJob = viewModelScope.launch {
                    delay(3500)
                    _state.update { state ->
                        if (state.recentlyActiveBinId == event.binId) {
                            state.copy(recentlyActiveBinId = null)
                        } else {
                            state
                        }
                    }
                }
            }
        }
    }

    private fun observeAppMode() {
        viewModelScope.launch {
            demoModeStore.mode.collect { mode ->
                _state.update { it.copy(appMode = mode) }
            }
        }
    }

    private fun observeStreamStatus() {
        viewModelScope.launch {
            binRepository.observeStreamStatus()
                .catch { error ->
                    _state.update {
                        it.copy(
                            streamConnected = false,
                            errorMessage = error.message ?: "Stream status unavailable",
                        )
                    }
                }
                .collect { isConnected ->
                _state.update { it.copy(streamConnected = isConnected) }
            }
        }
    }

    fun onMarkerTapped(binId: String) {
        _state.update { it.copy(detailBinId = binId) }
    }

    fun dismissBinDetails() {
        _state.update { it.copy(detailBinId = null) }
    }

    fun toggleBinSelection(binId: String) {
        _state.update { state ->
            val updated = state.selectedBinIds.toMutableSet().apply {
                if (!add(binId)) remove(binId)
            }
            state.copy(selectedBinIds = updated)
        }
    }

    fun selectLocality(locality: String?) {
        _state.update { state ->
            val visibleIds = state.bins
                .filter { locality == null || it.locality == locality }
                .mapTo(linkedSetOf()) { it.id }
            state.copy(
                selectedLocality = locality,
                selectedBinIds = if (locality == null) state.selectedBinIds else visibleIds,
            )
        }
    }

    fun selectVisibleBins() {
        _state.update { state ->
            state.copy(selectedBinIds = state.visibleBins.mapTo(linkedSetOf()) { it.id })
        }
    }

    fun clearSelection() {
        _state.update { it.copy(selectedBinIds = emptySet(), selectedLocality = null, detailBinId = null) }
    }

    fun triggerDemoEvent(binId: String? = null) {
        viewModelScope.launch {
            val targetBinId = binId
                ?: state.value.detailBinId
                ?: state.value.selectedBins.firstOrNull()?.id
                ?: state.value.visibleBins.firstOrNull()?.id
            binRepository.triggerDemoEvent(targetBinId)
        }
    }

    fun toggleAppMode() {
        demoModeStore.toggleMode()
    }
}
