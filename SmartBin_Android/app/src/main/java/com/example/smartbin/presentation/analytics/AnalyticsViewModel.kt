package com.example.smartbin.presentation.analytics

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.smartbin.domain.model.TimeRange
import com.example.smartbin.domain.repository.BinRepository
import com.example.smartbin.domain.usecase.AnalyticsBucket
import com.example.smartbin.domain.usecase.AnalyticsResult
import com.example.smartbin.domain.usecase.GetAggregatedAnalyticsUseCase
import com.example.smartbin.domain.usecase.StreamWasteEventsUseCase
import dagger.hilt.android.lifecycle.HiltViewModel
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import kotlinx.coroutines.delay
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

enum class TimeFilter(val label: String) {
    TODAY("Today"),
    WEEK("Week"),
    MONTH("Month"),
    SEASON("Season"),
    YEAR("Year"),
    CUSTOM("Custom"),
}

data class AnalyticsState(
    val selectedBinIds: Set<String> = emptySet(),
    val selectedLocalities: Set<String> = emptySet(),
    val timeFilter: TimeFilter = TimeFilter.WEEK,
    val customStartDate: LocalDate = LocalDate.now().minusDays(30),
    val customEndDate: LocalDate = LocalDate.now(),
    val analyticsResult: AnalyticsResult? = null,
    val isLoading: Boolean = false,
    val errorMessage: String? = null,
    val isCustomDateDialogVisible: Boolean = false,
)

@HiltViewModel
class AnalyticsViewModel @Inject constructor(
    private val binRepository: BinRepository,
    private val getAggregatedAnalyticsUseCase: GetAggregatedAnalyticsUseCase,
    private val streamWasteEventsUseCase: StreamWasteEventsUseCase,
) : ViewModel() {

    private val _state = MutableStateFlow(AnalyticsState())
    val state: StateFlow<AnalyticsState> = _state.asStateFlow()

    private var analyticsJob: Job? = null
    private var liveRefreshDebounceJob: Job? = null
    private var binLocalitiesById: Map<String, String> = emptyMap()

    init {
        observeBins()
        observeLiveUpdates()
    }

    fun onSelectionChanged(binIds: Set<String>, localities: Set<String>) {
        _state.update {
            it.copy(
                selectedBinIds = binIds,
                selectedLocalities = localities,
            )
        }
        refreshAnalytics()
    }

    fun onTimeFilterChanged(filter: TimeFilter) {
        _state.update { it.copy(timeFilter = filter) }
        if (filter == TimeFilter.CUSTOM) {
            _state.update { it.copy(isCustomDateDialogVisible = true) }
        } else {
            refreshAnalytics()
        }
    }

    fun showCustomDateDialog() {
        _state.update { it.copy(isCustomDateDialogVisible = true) }
    }

    fun dismissCustomDateDialog() {
        _state.update { it.copy(isCustomDateDialogVisible = false) }
    }

    fun onCustomDateRangeChanged(start: LocalDate, end: LocalDate) {
        val normalizedStart = minOf(start, end)
        val normalizedEnd = maxOf(start, end)
        _state.update {
            it.copy(
                timeFilter = TimeFilter.CUSTOM,
                customStartDate = normalizedStart,
                customEndDate = normalizedEnd,
                isCustomDateDialogVisible = false,
            )
        }
        refreshAnalytics()
    }

    fun refreshAnalytics() {
        val current = _state.value
        if (current.selectedBinIds.isEmpty() && current.selectedLocalities.isEmpty()) {
            analyticsJob?.cancel()
            analyticsJob = null
            liveRefreshDebounceJob?.cancel()
            liveRefreshDebounceJob = null
            _state.update { it.copy(analyticsResult = null, isLoading = false, errorMessage = null) }
            return
        }

        val timeRange = calculateTimeRange(current)
        analyticsJob?.cancel()
        _state.update { it.copy(isLoading = true, errorMessage = null) }
        analyticsJob = viewModelScope.launch {
            getAggregatedAnalyticsUseCase(
                binIds = current.selectedBinIds,
                localities = current.selectedLocalities,
                startTime = timeRange.start,
                endTime = timeRange.end,
                bucket = bucketFor(current.timeFilter),
            ).catch { error ->
                _state.update { it.copy(isLoading = false, errorMessage = error.message) }
            }.collect { result ->
                _state.update {
                    it.copy(
                        analyticsResult = result,
                        isLoading = false,
                        errorMessage = null,
                    )
                }
            }
        }
    }

    private fun observeLiveUpdates() {
        viewModelScope.launch {
            streamWasteEventsUseCase()
                .catch { error ->
                    _state.update { it.copy(errorMessage = error.message ?: "Realtime analytics updates failed") }
                }
                .collect { event ->
                    val current = _state.value
                    val matchesSelection = when {
                        current.selectedBinIds.isNotEmpty() -> event.binId in current.selectedBinIds
                        current.selectedLocalities.isNotEmpty() -> {
                            val eventLocality = binLocalitiesById[event.binId]
                            eventLocality != null && eventLocality in current.selectedLocalities
                        }
                        else -> false
                    }
                    if (matchesSelection) {
                        liveRefreshDebounceJob?.cancel()
                        liveRefreshDebounceJob = viewModelScope.launch {
                            delay(450)
                            refreshAnalytics()
                        }
                    }
                }
        }
    }

    private fun observeBins() {
        viewModelScope.launch {
            binRepository.observeBins()
                .catch { }
                .collect { bins ->
                    binLocalitiesById = bins.associate { it.id to it.locality }
                }
        }
    }

    private fun calculateTimeRange(state: AnalyticsState): TimeRange {
        val zoneId = ZoneId.systemDefault()
        val now = LocalDate.now(zoneId)
        val startDate = when (state.timeFilter) {
            TimeFilter.TODAY -> now
            TimeFilter.WEEK -> now.minusDays((now.dayOfWeek.value - 1).toLong())
            TimeFilter.MONTH -> now.withDayOfMonth(1)
            TimeFilter.SEASON -> seasonStart(now)
            TimeFilter.YEAR -> now.withDayOfYear(1)
            TimeFilter.CUSTOM -> state.customStartDate
        }
        val endDate = when (state.timeFilter) {
            TimeFilter.CUSTOM -> state.customEndDate
            else -> now
        }
        return TimeRange(
            start = startDate.atStartOfDay(zoneId).toInstant(),
            end = endDate.plusDays(1).atStartOfDay(zoneId).minusNanos(1).toInstant(),
        )
    }

    private fun bucketFor(filter: TimeFilter): AnalyticsBucket = when (filter) {
        TimeFilter.TODAY -> AnalyticsBucket.DAY
        TimeFilter.WEEK -> AnalyticsBucket.DAY
        TimeFilter.MONTH -> AnalyticsBucket.WEEK
        TimeFilter.SEASON -> AnalyticsBucket.WEEK
        TimeFilter.YEAR -> AnalyticsBucket.MONTH
        TimeFilter.CUSTOM -> AnalyticsBucket.DAY
    }

    private fun seasonStart(date: LocalDate): LocalDate {
        return when (date.monthValue) {
            2, 3, 4 -> date.withMonth(2).withDayOfMonth(1)
            5, 6, 7 -> date.withMonth(5).withDayOfMonth(1)
            8, 9, 10 -> date.withMonth(8).withDayOfMonth(1)
            11, 12 -> date.withMonth(11).withDayOfMonth(1)
            else -> date.minusYears(1).withMonth(11).withDayOfMonth(1)
        }
    }
}
