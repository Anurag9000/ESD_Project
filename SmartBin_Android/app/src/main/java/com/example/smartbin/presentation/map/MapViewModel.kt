package com.example.smartbin.presentation.map

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.smartbin.data.repository.AppMode
import com.example.smartbin.data.repository.DemoModeStore
import com.example.smartbin.domain.model.Bin
import com.example.smartbin.domain.model.WasteType
import com.example.smartbin.domain.repository.BinRepository
import com.example.smartbin.domain.usecase.StreamWasteEventsUseCase
import com.example.smartbin.notifications.BinAlertNotifier
import dagger.hilt.android.lifecycle.HiltViewModel
import java.time.Instant
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

data class WatchedBinAlert(
    val binId: String,
    val binName: String,
    val wasteType: WasteType,
    val confidence: Float,
    val timestamp: Instant,
)

data class MapState(
    val bins: List<Bin> = emptyList(),
    val isLoading: Boolean = true,
    val selectedBinIds: Set<String> = emptySet(),
    val selectedLocalities: Set<String> = emptySet(),
    val detailBinId: String? = null,
    val watchedBinId: String? = null,
    val recentlyActiveBinIds: Set<String> = emptySet(),
    val latestWatchedAlert: WatchedBinAlert? = null,
    val streamConnected: Boolean = false,
    val errorMessage: String? = null,
    val appMode: AppMode = AppMode.MOCK,
) {
    val localities: List<String>
        get() = bins.map { it.locality }.distinct().sorted()

    val visibleBins: List<Bin>
        get() = bins.filter { selectedLocalities.isEmpty() || it.locality in selectedLocalities }

    val selectedBins: List<Bin>
        get() = bins.filter { it.id in selectedBinIds }

    val detailBin: Bin?
        get() = bins.find { it.id == detailBinId }

    val watchedBin: Bin?
        get() = bins.find { it.id == watchedBinId }

    val explicitTriggerTargetBinId: String?
        get() = detailBinId ?: watchedBinId ?: selectedBins.singleOrNull()?.id
}

@HiltViewModel
class MapViewModel @Inject constructor(
    private val binRepository: BinRepository,
    private val demoModeStore: DemoModeStore,
    private val streamWasteEventsUseCase: StreamWasteEventsUseCase,
    private val binAlertNotifier: BinAlertNotifier,
) : ViewModel() {

    private val _state = MutableStateFlow(MapState())
    val state: StateFlow<MapState> = _state.asStateFlow()
    private val recentActivityResetJobs = linkedMapOf<String, Job>()

    init {
        observeAppMode()
        observeWatchedBin()
        observeLoadingState()
        observeRepositoryErrors()
        loadBins()
        observeStreamStatus()
        observeLiveEvents()
    }

    private fun observeLoadingState() {
        viewModelScope.launch {
            binRepository.observeBinsLoading().collect { loading ->
                _state.update { it.copy(isLoading = loading) }
            }
        }
    }

    private fun observeRepositoryErrors() {
        viewModelScope.launch {
            binRepository.observeRepositoryErrors().collect { message ->
                _state.update { it.copy(errorMessage = message) }
            }
        }
    }

    private fun observeWatchedBin() {
        viewModelScope.launch {
            demoModeStore.watchedBinId.collect { watchedBinId ->
                _state.update {
                    it.copy(
                        watchedBinId = watchedBinId,
                        latestWatchedAlert = if (watchedBinId == null) null else it.latestWatchedAlert,
                    )
                }
            }
        }
    }

    private fun loadBins() {
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
                val current = state.value
                val retainedLocalities = current.selectedLocalities.filterTo(linkedSetOf()) { selectedLocality ->
                    bins.any { it.locality == selectedLocality }
                }
                val retainedSelection = if (current.selectedLocalities.isNotEmpty()) {
                    bins.filter { it.locality in retainedLocalities }.mapTo(linkedSetOf()) { it.id }
                } else {
                    current.selectedBinIds.filterTo(linkedSetOf()) { selectedId ->
                        bins.any { it.id == selectedId }
                    }
                }
                val retainedWatchedBinId = current.watchedBinId?.takeIf { watchedBinId ->
                    bins.any { it.id == watchedBinId }
                }
                val retainedDetailBinId = current.detailBinId?.takeIf { detailBinId ->
                    bins.any { bin ->
                        bin.id == detailBinId && (retainedLocalities.isEmpty() || bin.locality in retainedLocalities)
                    }
                }
                if (current.watchedBinId != null && retainedWatchedBinId == null) {
                    demoModeStore.setWatchedBinId(null)
                }
                _state.update { current ->
                    current.copy(
                        bins = bins,
                        selectedBinIds = retainedSelection,
                        selectedLocalities = retainedLocalities,
                        detailBinId = retainedDetailBinId,
                        watchedBinId = retainedWatchedBinId,
                        latestWatchedAlert = current.latestWatchedAlert?.takeIf { alert -> alert.binId == retainedWatchedBinId },
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
                val currentState = state.value
                trackRecentActivity(event.binId)
                if (currentState.watchedBinId == event.binId) {
                    val watchedBin = currentState.bins.find { it.id == event.binId }
                    val watchedBinName = watchedBin?.name ?: "Watched bin ${event.binId}"
                    val alert = WatchedBinAlert(
                        binId = event.binId,
                        binName = watchedBinName,
                        wasteType = event.wasteType,
                        confidence = event.confidence,
                        timestamp = event.timestamp,
                    )
                    _state.update { it.copy(latestWatchedAlert = alert) }
                    binAlertNotifier.showWasteDetectedNotification(event.binId, watchedBinName, event)
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

    private fun trackRecentActivity(binId: String) {
        _state.update { it.copy(recentlyActiveBinIds = it.recentlyActiveBinIds + binId) }
        recentActivityResetJobs.remove(binId)?.cancel()
        recentActivityResetJobs[binId] = viewModelScope.launch {
            delay(3500)
            _state.update { state ->
                state.copy(recentlyActiveBinIds = state.recentlyActiveBinIds - binId)
            }
            recentActivityResetJobs.remove(binId)
        }
    }

    fun dismissWatchedAlert() {
        _state.update { it.copy(latestWatchedAlert = null) }
    }

    fun toggleWatchedBin(binId: String) {
        val current = state.value
        val nextWatchedBinId = if (current.watchedBinId == binId) null else binId
        demoModeStore.setWatchedBinId(nextWatchedBinId)
        _state.update { state ->
            state.copy(
                watchedBinId = nextWatchedBinId,
                latestWatchedAlert = if (nextWatchedBinId == null || nextWatchedBinId != state.latestWatchedAlert?.binId) null else state.latestWatchedAlert,
            )
        }
    }

    private fun observedLocalityBinIds(state: MapState, localities: Set<String>): Set<String> {
        return state.bins
            .filter { it.locality in localities }
            .mapTo(linkedSetOf()) { it.id }
    }

    fun onMarkerTapped(binId: String) {
        _state.update { it.copy(detailBinId = binId) }
    }

    fun dismissBinDetails() {
        _state.update { it.copy(detailBinId = null) }
    }

    fun toggleBinSelection(binId: String) {
        _state.update { state ->
            val baseSelection = state.selectedBinIds.toMutableSet()
            val updated = baseSelection.apply {
                if (!add(binId)) remove(binId)
            }
            state.copy(
                selectedBinIds = updated,
                selectedLocalities = emptySet(),
            )
        }
    }

    fun selectLocality(locality: String?) {
        _state.update { state ->
            if (locality == null) {
                return@update state.copy(selectedLocalities = emptySet(), selectedBinIds = emptySet())
            }
            val updatedLocalities = state.selectedLocalities.toMutableSet().apply {
                if (!add(locality)) remove(locality)
            }
            val localityBinIds = observedLocalityBinIds(state, updatedLocalities)
            val retainedDetailBinId = state.detailBinId?.takeIf { detailBinId ->
                state.bins.any { bin ->
                    bin.id == detailBinId && (updatedLocalities.isEmpty() || bin.locality in updatedLocalities)
                }
            }
            state.copy(
                selectedLocalities = updatedLocalities,
                selectedBinIds = localityBinIds,
                detailBinId = retainedDetailBinId,
            )
        }
    }

    fun selectVisibleBins() {
        _state.update { state ->
            state.copy(
                selectedBinIds = state.visibleBins.mapTo(linkedSetOf()) { it.id },
                selectedLocalities = emptySet(),
            )
        }
    }

    fun clearSelection() {
        _state.update {
            it.copy(
                selectedBinIds = emptySet(),
                selectedLocalities = emptySet(),
                detailBinId = null,
            )
        }
    }

    fun triggerDemoEvent(binId: String? = null) {
        viewModelScope.launch {
            val targetBinId = binId
                ?: state.value.detailBinId
                ?: state.value.watchedBinId
                ?: state.value.selectedBins.singleOrNull()?.id
            binRepository.triggerDemoEvent(targetBinId)
        }
    }

    fun toggleAppMode() {
        demoModeStore.toggleMode()
    }

    fun openBinFromNotification(binId: String) {
        _state.update { state ->
            state.copy(
                detailBinId = binId,
                selectedBinIds = linkedSetOf(binId),
                selectedLocalities = emptySet(),
            )
        }
    }
}
