package com.example.smartbin.presentation

import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Place
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.collectAsState
import androidx.compose.ui.Modifier
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.example.smartbin.presentation.analytics.AnalyticsScreen
import com.example.smartbin.presentation.analytics.AnalyticsViewModel
import com.example.smartbin.presentation.map.BinMapScreen
import com.example.smartbin.presentation.map.MapViewModel

sealed class Screen(val route: String, val label: String, val icon: @Composable () -> Unit) {
    data object Map : Screen("map", "Fleet Map", { Icon(Icons.Default.Place, contentDescription = null) })
    data object Analytics : Screen("analytics", "Analytics", { Icon(Icons.Default.Info, contentDescription = null) })
}

@Composable
fun MainScreen() {
    val navController = rememberNavController()
    val mapViewModel: MapViewModel = hiltViewModel()
    val analyticsViewModel: AnalyticsViewModel = hiltViewModel()

    val mapState by mapViewModel.state.collectAsState()
    val analyticsState by analyticsViewModel.state.collectAsState()
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentDestination = navBackStackEntry?.destination

    LaunchedEffect(mapState.selectedBinIds, mapState.selectedLocality) {
        analyticsViewModel.onSelectionChanged(
            binIds = mapState.selectedBinIds,
            localities = mapState.selectedLocality?.let { setOf(it) } ?: emptySet(),
        )
    }

    Scaffold(
        bottomBar = {
            NavigationBar {
                listOf(Screen.Map, Screen.Analytics).forEach { screen ->
                    NavigationBarItem(
                        icon = screen.icon,
                        label = { Text(screen.label) },
                        selected = currentDestination?.hierarchy?.any { it.route == screen.route } == true,
                        onClick = {
                            navController.navigate(screen.route) {
                                popUpTo(navController.graph.findStartDestination().id) {
                                    saveState = true
                                }
                                launchSingleTop = true
                                restoreState = true
                            }
                        },
                    )
                }
            }
        },
        floatingActionButton = if (currentDestination?.hierarchy?.any { it.route == Screen.Map.route } == true) {
            {
            FloatingActionButton(onClick = { mapViewModel.triggerDemoEvent() }) {
                Text("Live")
            }
            }
        } else {
            {}
        },
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = Screen.Map.route,
            modifier = Modifier.padding(innerPadding),
        ) {
            composable(Screen.Map.route) {
                BinMapScreen(
                    state = mapState,
                    onMarkerTapped = mapViewModel::onMarkerTapped,
                    onToggleBinSelection = mapViewModel::toggleBinSelection,
                    onSelectLocality = mapViewModel::selectLocality,
                    onSelectVisibleBins = mapViewModel::selectVisibleBins,
                    onClearSelection = mapViewModel::clearSelection,
                    onDismissDetails = mapViewModel::dismissBinDetails,
                    onTriggerDemoEvent = mapViewModel::triggerDemoEvent,
                    onToggleAppMode = mapViewModel::toggleAppMode,
                )
            }
            composable(Screen.Analytics.route) {
                AnalyticsScreen(
                    state = analyticsState,
                    selectedBins = mapState.selectedBins,
                    selectedLocality = mapState.selectedLocality,
                    onTimeFilterChanged = analyticsViewModel::onTimeFilterChanged,
                    onShowCustomDateDialog = analyticsViewModel::showCustomDateDialog,
                    onDismissCustomDateDialog = analyticsViewModel::dismissCustomDateDialog,
                    onCustomDateRangeSelected = analyticsViewModel::onCustomDateRangeChanged,
                )
            }
        }
    }
}
