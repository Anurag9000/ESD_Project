package com.example.smartbin.presentation.analytics

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.smartbin.domain.model.WasteType
import com.example.smartbin.domain.usecase.AnalyticsResult
import com.patrykandpatrick.vico.compose.axis.horizontal.rememberBottomAxis
import com.patrykandpatrick.vico.compose.axis.vertical.rememberStartAxis
import com.patrykandpatrick.vico.compose.chart.Chart
import com.patrykandpatrick.vico.compose.chart.column.columnChart
import com.patrykandpatrick.vico.core.entry.entryModelOf

@Composable
fun AnalyticsScreen(
    viewModel: AnalyticsViewModel,
    modifier: Modifier = Modifier
) {
    val state by viewModel.state.collectAsState()

    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text(
            text = "Waste Analytics",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold
        )

        TimeFilterRow(
            selectedFilter = state.timeFilter,
            onFilterSelected = { viewModel.onTimeFilterChanged(it) }
        )

        if (state.isLoading) {
            CircularProgressIndicator(modifier = Modifier.align(Alignment.CenterHorizontally))
        } else if (state.analyticsResult != null) {
            SummaryCard(result = state.analyticsResult!!)
            WasteCompositionChart(result = state.analyticsResult!!)
        } else {
            Text(
                text = "Select bins on the map to see analytics.",
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.secondary
            )
        }
    }
}

@Composable
fun TimeFilterRow(
    selectedFilter: TimeFilter,
    onFilterSelected: (TimeFilter) -> Unit
) {
    ScrollableTabRow(
        selectedTabIndex = selectedFilter.ordinal,
        edgePadding = 0.dp,
        containerColor = Color.Transparent,
        divider = {}
    ) {
        TimeFilter.entries.forEach { filter ->
            Tab(
                selected = selectedFilter == filter,
                onClick = { onFilterSelected(filter) },
                text = { Text(filter.name) }
            )
        }
    }
}

@Composable
fun SummaryCard(result: AnalyticsResult) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text(text = "Total Waste Events", style = MaterialTheme.typography.labelLarge)
            Text(
                text = result.totalEvents.toString(),
                style = MaterialTheme.typography.displaySmall,
                fontWeight = FontWeight.Bold
            )
        }
    }
}

@Composable
fun WasteCompositionChart(result: AnalyticsResult) {
    val entries = result.countsByType.entries.mapIndexed { index, entry ->
        index.toFloat() to entry.value.toFloat()
    }
    val chartEntryModel = entryModelOf(*entries.toTypedArray())

    Column {
        Text(
            text = "Composition by Type",
            style = MaterialTheme.typography.titleLarge,
            modifier = Modifier.padding(bottom = 8.dp)
        )
        Chart(
            chart = columnChart(),
            model = chartEntryModel,
            startAxis = rememberStartAxis(),
            bottomAxis = rememberBottomAxis(),
            modifier = Modifier
                .fillMaxWidth()
                .height(250.dp)
        )
        
        Spacer(modifier = Modifier.height(8.dp))
        
        // Legend Mockup
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            WasteType.entries.forEach { type ->
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(text = type.name, style = MaterialTheme.typography.labelSmall)
                }
            }
        }
    }
}
