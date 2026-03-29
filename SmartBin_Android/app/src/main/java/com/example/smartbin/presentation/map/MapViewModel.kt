package com.example.smartbin.presentation.map

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.smartbin.domain.model.Bin
import com.example.smartbin.domain.repository.BinRepository
import com.example.smartbin.domain.usecase.StreamWasteEventsUseCase
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class MapState(
    val bins: List<Bin> = emptyList(),
    val isLoading: Boolean = false,
    val selectedBinId: String? = null,
    val lastEventBinId: String? = null
)

@HiltViewModel
class MapViewModel @Inject constructor(
    private val binRepository: BinRepository,
    private val streamWasteEventsUseCase: StreamWasteEventsUseCase
) : ViewModel() {

    private val _state = MutableStateFlow(MapState())
    val state: StateFlow<MapState> = _state.asStateFlow()

    init {
        loadBins()
        observeLiveEvents()
    }

    private fun loadBins() {
        _state.update { it.copy(isLoading = true) }
        viewModelScope.launch {
            binRepository.getBins().collect { bins ->
                _state.update { it.copy(bins = bins, isLoading = false) }
            }
        }
    }

    private fun observeLiveEvents() {
        viewModelScope.launch {
            streamWasteEventsUseCase().collect { event ->
                // Visual Pulse: Set the active bin for 3 seconds
                _state.update { it.copy(lastEventBinId = event.binId) }
                delay(3000)
                _state.update { it.copy(lastEventBinId = null) }
            }
        }
    }

    fun onBinSelected(binId: String) {
        _state.update { it.copy(selectedBinId = binId) }
    }

    fun triggerDemoEvent() {
        viewModelScope.launch {
            (binRepository as? com.example.smartbin.data.repository.MockBinRepository)?.let { mockRepo ->
                mockRepo.triggerManualEvent(
                    binId = _state.value.bins.randomOrNull()?.id ?: "BIN-001",
                    type = com.example.smartbin.domain.model.WasteType.entries.random()
                )
            }
        }
    }
}
