package com.example.smartbin.presentation

import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Analytics
import androidx.compose.material.icons.filled.BugReport
import androidx.compose.material.icons.filled.Map
import androidx.compose.material3.*
import androidx.compose.runtime.*
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
import com.example.smartbin.presentation.map.BinMap
import com.example.smartbin.presentation.map.MapViewModel

sealed class Screen(val route: String, val label: String, val icon: @Composable () -> Unit) {
    object Map : Screen("map", "Fleet Map", { Icon(Icons.Default.Map, contentDescription = null) })
    object Analytics : Screen("analytics", "Analytics", { Icon(Icons.Default.Analytics, contentDescription = null) })
}

@Composable
fun MainScreen() {
    val navController = rememberNavController()
    val mapViewModel: MapViewModel = hiltViewModel()
    val analyticsViewModel: AnalyticsViewModel = hiltViewModel()
    
    val mapState by mapViewModel.state.collectAsState()
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentDestination = navBackStackEntry?.destination

    // Synchronize selection: Map -> Analytics
    LaunchedEffect(mapState.selectedBinId) {
        mapState.selectedBinId?.let { id ->
            analyticsViewModel.onBinsSelected(listOf(id))
        }
    }

    Scaffold(
        bottomBar = {
            NavigationBar {
                val items = listOf(Screen.Map, Screen.Analytics)
                items.forEach { screen ->
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
                        }
                    )
                }
            }
        },
        floatingActionButton = {
            FloatingActionButton(onClick = { mapViewModel.triggerDemoEvent() }) {
                Icon(Icons.Default.BugReport, contentDescription = "Trigger Demo Event")
            }
        }
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = Screen.Map.route,
            modifier = Modifier.padding(innerPadding)
        ) {
            composable(Screen.Map.route) {
                BinMap(
                    bins = mapState.bins,
                    onBinClick = { binId ->
                        mapViewModel.onBinSelected(binId)
                    }
                )
            }
            composable(Screen.Analytics.route) {
                AnalyticsScreen(viewModel = analyticsViewModel)
            }
        }
    }
}
