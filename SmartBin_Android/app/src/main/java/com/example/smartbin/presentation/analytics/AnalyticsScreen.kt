@file:OptIn(ExperimentalLayoutApi::class)

package com.example.smartbin.presentation.analytics

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DatePickerDialog
import androidx.compose.material3.DateRangePicker
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberDateRangePickerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.smartbin.domain.model.Bin
import com.example.smartbin.domain.model.WasteType
import com.example.smartbin.domain.usecase.AnalyticsResult
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId

@Composable
fun AnalyticsScreen(
    state: AnalyticsState,
    selectedBins: List<Bin>,
    selectedLocalities: Set<String>,
    onTimeFilterChanged: (TimeFilter) -> Unit,
    onShowCustomDateDialog: () -> Unit,
    onDismissCustomDateDialog: () -> Unit,
    onCustomDateRangeSelected: (LocalDate, LocalDate) -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        AnalyticsHeader(selectedBins = selectedBins, selectedLocalities = selectedLocalities)
        TimeFilterRow(
            selectedFilter = state.timeFilter,
            onFilterSelected = { filter ->
                if (filter == TimeFilter.CUSTOM) onShowCustomDateDialog() else onTimeFilterChanged(filter)
            },
        )

        when {
            state.isLoading -> {
                Box(modifier = Modifier.fillMaxWidth().padding(top = 48.dp), contentAlignment = Alignment.Center) {
                    CircularProgressIndicator()
                }
            }

            state.errorMessage != null -> {
                Text(
                    text = state.errorMessage,
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.error,
                )
            }

            state.analyticsResult != null -> {
                AnalyticsDashboard(result = state.analyticsResult!!)
            }

            else -> {
                EmptyAnalyticsState()
            }
        }
    }

    if (state.isCustomDateDialogVisible) {
        CustomDateRangeDialog(
            start = state.customStartDate,
            end = state.customEndDate,
            onDismiss = onDismissCustomDateDialog,
            onConfirm = onCustomDateRangeSelected,
        )
    }
}

@Composable
private fun AnalyticsHeader(selectedBins: List<Bin>, selectedLocalities: Set<String>) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(
            text = "Waste Analytics",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
        )
        Text(
            text = when {
                selectedBins.isNotEmpty() && selectedLocalities.isNotEmpty() ->
                    "${selectedBins.size} bins selected in ${selectedLocalities.joinToString()}"
                selectedBins.isNotEmpty() ->
                    "${selectedBins.size} bins selected for aggregate analysis"
                selectedLocalities.isNotEmpty() ->
                    "Viewing locality analytics for ${selectedLocalities.joinToString()}"
                else ->
                    "Select bins on the map to build an analytics group"
            },
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.secondary,
        )
    }
}

@Composable
private fun TimeFilterRow(
    selectedFilter: TimeFilter,
    onFilterSelected: (TimeFilter) -> Unit,
) {
    FlowRow(
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        TimeFilter.entries.forEach { filter ->
            FilterChip(
                selected = selectedFilter == filter,
                onClick = { onFilterSelected(filter) },
                label = { Text(filter.label) },
                colors = FilterChipDefaults.filterChipColors(
                    selectedContainerColor = MaterialTheme.colorScheme.primaryContainer,
                ),
            )
        }
    }
}

@Composable
private fun AnalyticsDashboard(result: AnalyticsResult) {
    SummaryCards(result)
    WasteCompositionSection(result)
    TrendSection(result)
    ConfidenceSection(result)
}

@Composable
private fun SummaryCards(result: AnalyticsResult) {
    FlowRow(
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        AnalyticsMetricCard("Total events", result.totalEvents.toString())
        AnalyticsMetricCard("Average confidence", "${(result.averageConfidence * 100).toInt()}%")
        AnalyticsMetricCard("Selected bins", result.selectedBinCount.toString())
        AnalyticsMetricCard("Localities", result.selectedLocalityCount.toString())
    }
}

@Composable
private fun AnalyticsMetricCard(label: String, value: String) {
    Card(
        modifier = Modifier.width(160.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerHighest),
        shape = RoundedCornerShape(24.dp),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(text = label, style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text(text = value, style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
        }
    }
}

@Composable
private fun WasteCompositionSection(result: AnalyticsResult) {
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        shape = RoundedCornerShape(28.dp),
    ) {
        BoxWithConstraints(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
        ) {
            val useStackedLayout = maxWidth < 520.dp
            Column(verticalArrangement = Arrangement.spacedBy(20.dp)) {
                Text("Waste composition", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
                if (useStackedLayout) {
                    Column(
                        modifier = Modifier.fillMaxWidth(),
                        verticalArrangement = Arrangement.spacedBy(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                    ) {
                        DonutChart(
                            values = WasteType.entries.map { result.percentagesByType[it] ?: 0f },
                            colors = WasteType.entries.map(::wasteColor),
                            modifier = Modifier.size(220.dp),
                        )
                        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                            WasteType.entries.forEach { type ->
                                val count = result.countsByType[type] ?: 0
                                val percentage = (result.percentagesByType[type] ?: 0f) * 100f
                                LegendRow(
                                    type = type,
                                    label = "${type.label} · $count items",
                                    value = "${percentage.toInt()}%",
                                )
                            }
                        }
                    }
                } else {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(16.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        DonutChart(
                            values = WasteType.entries.map { result.percentagesByType[it] ?: 0f },
                            colors = WasteType.entries.map(::wasteColor),
                            modifier = Modifier.size(220.dp),
                        )
                        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                            WasteType.entries.forEach { type ->
                                val count = result.countsByType[type] ?: 0
                                val percentage = (result.percentagesByType[type] ?: 0f) * 100f
                                LegendRow(
                                    type = type,
                                    label = "${type.label} · $count items",
                                    value = "${percentage.toInt()}%",
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ConfidenceSection(result: AnalyticsResult) {
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        shape = RoundedCornerShape(28.dp),
    ) {
        Column(
            modifier = Modifier.fillMaxWidth().padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Text("Average confidence by waste type", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
            WasteType.entries.forEach { type ->
                val confidence = result.averageConfidenceByType[type] ?: 0f
                val percentage = (confidence * 100f).toInt()
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                    ) {
                        Text(type.label, style = MaterialTheme.typography.bodyLarge)
                        Text("$percentage%", style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.SemiBold)
                    }
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(12.dp)
                            .background(MaterialTheme.colorScheme.surfaceContainerHighest, RoundedCornerShape(999.dp)),
                    ) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth(confidence.coerceIn(0f, 1f))
                                .height(12.dp)
                                .background(wasteColor(type), RoundedCornerShape(999.dp)),
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun TrendSection(result: AnalyticsResult) {
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        shape = RoundedCornerShape(28.dp),
    ) {
        Column(
            modifier = Modifier.fillMaxWidth().padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(18.dp),
        ) {
            Text("Waste trend over time", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
            if (result.trend.isEmpty()) {
                Text("No trend data for the selected period.", color = MaterialTheme.colorScheme.onSurfaceVariant)
            } else {
                val trendPoints = result.trend.takeLast(8)
                val maxEvents = remember(trendPoints) { trendPoints.maxOf { it.totalEvents }.coerceAtLeast(1) }
                if (result.trend.size > trendPoints.size) {
                    Text(
                        "Showing the most recent ${trendPoints.size} time buckets from the selected period.",
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
                trendPoints.forEach { point ->
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                        ) {
                            Text(point.label, style = MaterialTheme.typography.bodyMedium)
                            Text("${point.totalEvents} events", style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold)
                        }
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(6.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            WasteType.entries.forEach { type ->
                                val widthFraction = (point.countsByType[type] ?: 0).toFloat() / maxEvents.toFloat()
                                Box(
                                    modifier = Modifier
                                        .weight(1f)
                                        .height(14.dp)
                                        .background(MaterialTheme.colorScheme.surfaceContainerHighest, RoundedCornerShape(999.dp)),
                                ) {
                                    Box(
                                        modifier = Modifier
                                            .fillMaxWidth(widthFraction.coerceIn(0f, 1f))
                                            .height(14.dp)
                                            .background(wasteColor(type), RoundedCornerShape(999.dp)),
                                    )
                                }
                            }
                        }
                    }
                }
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    WasteType.entries.forEach { type ->
                        LegendRow(type = type, label = type.label, value = "")
                    }
                }
            }
        }
    }
}

@Composable
private fun LegendRow(type: WasteType, label: String, value: String) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
        Box(
            modifier = Modifier
                .size(12.dp)
                .background(wasteColor(type), CircleShape),
        )
        Text(label, style = MaterialTheme.typography.bodyMedium)
        if (value.isNotBlank()) {
            Text(value, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold)
        }
    }
}

@Composable
private fun EmptyAnalyticsState() {
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow),
        shape = RoundedCornerShape(28.dp),
    ) {
        Column(
            modifier = Modifier.fillMaxWidth().padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text("No bins selected", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
            Text(
                "Select one bin, multiple bins, or a locality on the fleet map. The app will aggregate all matching waste events automatically.",
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun DonutChart(values: List<Float>, colors: List<Color>, modifier: Modifier = Modifier) {
    val holeColor = MaterialTheme.colorScheme.surface
    Canvas(modifier = modifier) {
        val strokeWidth = size.minDimension * 0.18f
        val diameter = size.minDimension - strokeWidth
        val topLeft = Offset((size.width - diameter) / 2f, (size.height - diameter) / 2f)
        val arcSize = Size(diameter, diameter)
        var startAngle = -90f
        values.zip(colors).forEach { (value, color) ->
            val sweep = value.coerceAtLeast(0f) * 360f
            if (sweep > 0f) {
                drawArc(
                    color = color,
                    startAngle = startAngle,
                    sweepAngle = sweep,
                    useCenter = false,
                    topLeft = topLeft,
                    size = arcSize,
                    style = Stroke(width = strokeWidth, cap = StrokeCap.Round),
                )
            }
            startAngle += sweep
        }
        drawCircle(color = holeColor, radius = (diameter - strokeWidth) / 2f, center = center)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun CustomDateRangeDialog(
    start: LocalDate,
    end: LocalDate,
    onDismiss: () -> Unit,
    onConfirm: (LocalDate, LocalDate) -> Unit,
) {
    val zoneId = remember { ZoneId.systemDefault() }
    val state = rememberDateRangePickerState(
        initialSelectedStartDateMillis = start.atStartOfDay(zoneId).toInstant().toEpochMilli(),
        initialSelectedEndDateMillis = end.atStartOfDay(zoneId).toInstant().toEpochMilli(),
    )
    val selectedStart = state.selectedStartDateMillis?.let { millis ->
        Instant.ofEpochMilli(millis).atZone(zoneId).toLocalDate()
    }
    val selectedEnd = state.selectedEndDateMillis?.let { millis ->
        Instant.ofEpochMilli(millis).atZone(zoneId).toLocalDate()
    }
    val canApply = selectedStart != null && selectedEnd != null && !selectedEnd.isBefore(selectedStart)
    DatePickerDialog(
        onDismissRequest = onDismiss,
        confirmButton = {
            TextButton(
                enabled = canApply,
                onClick = {
                    onConfirm(selectedStart ?: start, selectedEnd ?: end)
                },
            ) {
                Text("Apply")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        },
    ) {
        DateRangePicker(state = state, modifier = Modifier.padding(8.dp))
    }
}

private fun wasteColor(type: WasteType): Color = when (type) {
    WasteType.METAL -> Color(0xFF7C8799)
    WasteType.ORGANIC -> Color(0xFF25875A)
    WasteType.PAPER -> Color(0xFF4F7DF3)
    WasteType.OTHER -> Color(0xFFD9932F)
}
