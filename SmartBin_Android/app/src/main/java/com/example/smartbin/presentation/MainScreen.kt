package com.example.smartbin.presentation

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Place
import androidx.compose.material.icons.filled.Settings
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
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.ContextCompat
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.example.smartbin.presentation.analytics.AnalyticsScreen
import com.example.smartbin.presentation.analytics.AnalyticsViewModel
import com.example.smartbin.presentation.config.ClassConfigurationScreen
import com.example.smartbin.presentation.config.ClassConfigurationViewModel
import com.example.smartbin.presentation.map.BinMapScreen
import com.example.smartbin.presentation.map.MapViewModel

sealed class Screen(val route: String, val label: String, val icon: @Composable () -> Unit) {
    data object Map : Screen("map", "Fleet Map", { Icon(Icons.Default.Place, contentDescription = null) })
    data object Analytics : Screen("analytics", "Analytics", { Icon(Icons.Default.Info, contentDescription = null) })
    data object Classes : Screen("classes", "Classes", { Icon(Icons.Default.Settings, contentDescription = null) })
}

@Composable
fun MainScreen(
    notificationOpenBinId: String? = null,
    onNotificationBinConsumed: () -> Unit = {},
) {
    val context = LocalContext.current
    val navController = rememberNavController()
    val mapViewModel: MapViewModel = hiltViewModel()
    val analyticsViewModel: AnalyticsViewModel = hiltViewModel()
    val classConfigurationViewModel: ClassConfigurationViewModel = hiltViewModel()

    val mapState by mapViewModel.state.collectAsState()
    val analyticsState by analyticsViewModel.state.collectAsState()
    val classConfigurationState by classConfigurationViewModel.state.collectAsState()
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentDestination = navBackStackEntry?.destination
    val notificationPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
        onResult = {},
    )

    LaunchedEffect(mapState.selectedBinIds, mapState.selectedLocalities) {
        analyticsViewModel.onSelectionChanged(
            binIds = if (mapState.selectedLocalities.isNotEmpty()) emptySet() else mapState.selectedBinIds,
            localities = mapState.selectedLocalities,
        )
    }
    LaunchedEffect(notificationOpenBinId) {
        val binId = notificationOpenBinId ?: return@LaunchedEffect
        navController.navigate(Screen.Map.route) {
            launchSingleTop = true
        }
        mapViewModel.openBinFromNotification(binId)
        onNotificationBinConsumed()
    }
    LaunchedEffect(mapState.watchedBinId) {
        if (
            mapState.watchedBinId != null &&
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
    }

    if (classConfigurationState.requiresInitialConfirmation) {
        ClassConfigurationScreen(
            state = classConfigurationState,
            onClassCountChanged = classConfigurationViewModel::onClassCountChanged,
            onPrimaryClassSelected = classConfigurationViewModel::onPrimaryClassSelected,
            onMergedRawClassToggled = classConfigurationViewModel::onMergedRawClassToggled,
            onReset = classConfigurationViewModel::resetDraft,
            onSave = classConfigurationViewModel::saveDraft,
            mandatory = true,
        )
        return
    }

    Scaffold(
        bottomBar = {
            NavigationBar {
                listOf(Screen.Map, Screen.Analytics, Screen.Classes).forEach { screen ->
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
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = Screen.Map.route,
            modifier = Modifier.padding(innerPadding),
        ) {
            composable(Screen.Map.route) {
                BinMapScreen(
                    state = mapState,
                    classConfiguration = classConfigurationState.savedConfiguration,
                    onMarkerTapped = mapViewModel::onMarkerTapped,
                    onToggleBinSelection = mapViewModel::toggleBinSelection,
                    onRemoveSelectedBin = mapViewModel::toggleBinSelection,
                    onSelectLocality = mapViewModel::selectLocality,
                    onSelectVisibleBins = mapViewModel::selectVisibleBins,
                    onClearSelection = mapViewModel::clearSelection,
                    onDismissDetails = mapViewModel::dismissBinDetails,
                    onDismissWatchedAlert = mapViewModel::dismissWatchedAlert,
                    onTriggerDemoEvent = mapViewModel::triggerDemoEvent,
                    onToggleWatchedBin = mapViewModel::toggleWatchedBin,
                    onToggleAppMode = mapViewModel::toggleAppMode,
                )
            }
            composable(Screen.Analytics.route) {
                AnalyticsScreen(
                    state = analyticsState,
                    classConfiguration = classConfigurationState.savedConfiguration,
                    selectedBins = mapState.selectedBins,
                    selectedLocalities = mapState.selectedLocalities,
                    onTimeFilterChanged = analyticsViewModel::onTimeFilterChanged,
                    onShowCustomDateDialog = analyticsViewModel::showCustomDateDialog,
                    onDismissCustomDateDialog = analyticsViewModel::dismissCustomDateDialog,
                    onCustomDateRangeSelected = analyticsViewModel::onCustomDateRangeChanged,
                )
            }
            composable(Screen.Classes.route) {
                ClassConfigurationScreen(
                    state = classConfigurationState,
                    onClassCountChanged = classConfigurationViewModel::onClassCountChanged,
                    onPrimaryClassSelected = classConfigurationViewModel::onPrimaryClassSelected,
                    onMergedRawClassToggled = classConfigurationViewModel::onMergedRawClassToggled,
                    onReset = classConfigurationViewModel::resetDraft,
                    onSave = classConfigurationViewModel::saveDraft,
                )
            }
        }
    }
}
