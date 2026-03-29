package com.example.smartbin.presentation.analytics

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.smartbin.domain.model.WasteType
import com.example.smartbin.domain.usecase.AnalyticsResult
import com.example.smartbin.domain.usecase.GetAggregatedAnalyticsUseCase
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import javax.inject.Inject

enum class TimeFilter {
    WEEK, MONTH, SEASON, YEAR, CUSTOM
}

data class AnalyticsState(
    val selectedBinIds: List<String> = emptyList(),
    val timeFilter: TimeFilter = TimeFilter.WEEK,
    val analyticsResult: AnalyticsResult? = null,
    val isLoading: Boolean = false,
    val errorMessage: String? = null
)

@HiltViewModel
class AnalyticsViewModel @Inject constructor(
    private val getAggregatedAnalyticsUseCase: GetAggregatedAnalyticsUseCase
) : ViewModel() {

    private val _state = MutableStateFlow(AnalyticsState())
    val state: StateFlow<AnalyticsState> = _state.asStateFlow()

    fun onBinsSelected(binIds: List<String>) {
        _state.update { it.copy(selectedBinIds = binIds) }
        refreshAnalytics()
    }

    fun onTimeFilterChanged(filter: TimeFilter) {
        _state.update { it.copy(timeFilter = filter) }
        refreshAnalytics()
    }

    private fun refreshAnalytics() {
        val currentState = _state.value
        if (currentState.selectedBinIds.isEmpty()) return

        val (startTime, endTime) = calculateTimeRange(currentState.timeFilter)

        _state.update { it.copy(isLoading = true, errorMessage = null) }

        viewModelScope.launch {
            getAggregatedAnalyticsUseCase(
                binIds = currentState.selectedBinIds,
                startTime = startTime,
                endTime = endTime
            ).catch { e ->
                _state.update { it.copy(isLoading = false, errorMessage = e.message) }
            }.collect { result ->
                _state.update { it.copy(analyticsResult = result, isLoading = false) }
            }
        }
    }

    private fun calculateTimeRange(filter: TimeFilter): Pair<String, String> {
        val now = LocalDate.now()
        val formatter = DateTimeFormatter.ISO_DATE
        val start = when (filter) {
            TimeFilter.WEEK -> now.minusWeeks(1)
            TimeFilter.MONTH -> now.minusMonths(1)
            TimeFilter.SEASON -> getStartOfCurrentSeason(now)
            TimeFilter.YEAR -> now.withDayOfYear(1)
            TimeFilter.CUSTOM -> now.minusDays(30) // Default for custom
        }
        return Pair(start.format(formatter), now.format(formatter))
    }

    /**
     * Requested Seasonal Logic:
     * Q1: Feb-Apr | Q2: May-Jul | Q3: Aug-Oct | Q4: Nov-Jan
     */
    private fun getStartOfCurrentSeason(date: LocalDate): LocalDate {
        val month = date.monthValue
        return when (month) {
            2, 3, 4 -> date.withMonth(2).withDayOfMonth(1)
            5, 6, 7 -> date.withMonth(5).withDayOfMonth(1)
            8, 9, 10 -> date.withMonth(8).withDayOfMonth(1)
            else -> { // Nov, Dec, Jan
                if (month == 1) date.minusYears(1).withMonth(11).withDayOfMonth(1)
                else date.withMonth(11).withDayOfMonth(1)
            }
        }
    }
}
