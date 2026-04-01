@file:OptIn(ExperimentalLayoutApi::class, androidx.compose.material3.ExperimentalMaterial3Api::class)

package com.example.smartbin.presentation.map

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.smartbin.domain.model.ResolvedWasteClassConfiguration
import com.example.smartbin.data.repository.AppMode
import com.example.smartbin.domain.model.Bin
import com.example.smartbin.domain.model.BinStatus
import com.example.smartbin.presentation.wasteClassColor
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import org.maplibre.android.annotations.IconFactory
import org.maplibre.android.annotations.Marker
import org.maplibre.android.annotations.MarkerOptions
import org.maplibre.android.camera.CameraPosition
import org.maplibre.android.geometry.LatLng
import org.maplibre.android.geometry.LatLngBounds
import org.maplibre.android.maps.MapLibreMap
import org.maplibre.android.maps.MapView
import org.maplibre.android.maps.Style

@Composable
fun BinMapScreen(
    state: MapState,
    classConfiguration: ResolvedWasteClassConfiguration,
    onMarkerTapped: (String) -> Unit,
    onToggleBinSelection: (String) -> Unit,
    onSelectLocality: (String?) -> Unit,
    onSelectVisibleBins: () -> Unit,
    onClearSelection: () -> Unit,
    onDismissDetails: () -> Unit,
    onDismissWatchedAlert: () -> Unit,
    onTriggerDemoEvent: (String?) -> Unit,
    onToggleWatchedBin: (String) -> Unit,
    onToggleAppMode: () -> Unit,
) {
    val detailBin = state.detailBin
    val bottomSheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)

    Box(modifier = Modifier.fillMaxSize()) {
        Column(modifier = Modifier.fillMaxSize()) {
            FleetMapHeader(
                state = state,
                onSelectLocality = onSelectLocality,
                onSelectVisibleBins = onSelectVisibleBins,
                onClearSelection = onClearSelection,
                onToggleAppMode = onToggleAppMode,
            )
            if (state.errorMessage != null) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
                ) {
                    Text(
                        text = state.errorMessage,
                        modifier = Modifier.padding(12.dp),
                        color = MaterialTheme.colorScheme.onErrorContainer,
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
            }
            if (state.latestWatchedAlert != null) {
                WatchedBinAlertBanner(
                    alert = state.latestWatchedAlert,
                    classConfiguration = classConfiguration,
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 4.dp),
                    onDismiss = onDismissWatchedAlert,
                )
            }
                BinMap(
                    bins = state.visibleBins,
                    classConfiguration = classConfiguration,
                    selectedBinIds = state.selectedBinIds,
                activeBinIds = state.recentlyActiveBinIds,
                selectedLocalities = state.selectedLocalities,
                onMarkerTapped = onMarkerTapped,
                modifier = Modifier.weight(1f),
            )
        }

        SelectionSummaryCard(
            state = state,
            onTriggerDemoEvent = onTriggerDemoEvent,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(16.dp),
        )
    }

    if (detailBin != null) {
        ModalBottomSheet(
            onDismissRequest = onDismissDetails,
            sheetState = bottomSheetState,
            containerColor = MaterialTheme.colorScheme.surface,
        ) {
            BinDetailSheet(
                bin = detailBin,
                classConfiguration = classConfiguration,
                isSelected = detailBin.id in state.selectedBinIds,
                isWatched = detailBin.id == state.watchedBinId,
                onToggleSelection = { onToggleBinSelection(detailBin.id) },
                onTriggerDemoEvent = { onTriggerDemoEvent(detailBin.id) },
                onToggleWatchedBin = { onToggleWatchedBin(detailBin.id) },
            )
        }
    }
}

@Composable
private fun WatchedBinAlertBanner(
    alert: WatchedBinAlert,
    classConfiguration: ResolvedWasteClassConfiguration,
    onDismiss: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val displayLabel = classConfiguration.toRuntimeDisplayLabel(alert.predictedClass)
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(containerColor = SmartGreen.copy(alpha = 0.14f)),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(2.dp),
            ) {
                Text(
                    text = "Watched bin alert",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                )
                Text(
                    text = "${alert.binName} detected ${displayLabel.lowercase()} at ${(alert.confidence * 100).toInt()}% confidence",
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
            TextButton(onClick = onDismiss) {
                Text("Dismiss")
            }
        }
    }
}

@Composable
private fun FleetMapHeader(
    state: MapState,
    onSelectLocality: (String?) -> Unit,
    onSelectVisibleBins: () -> Unit,
    onClearSelection: () -> Unit,
    onToggleAppMode: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(MaterialTheme.colorScheme.surface)
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column {
                Text(
                    text = "SmartBin Fleet",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    text = when {
                        state.appMode == AppMode.MOCK -> "Mock demo mode"
                        state.streamConnected -> "Live server connected"
                        else -> "Live server unavailable"
                    },
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.secondary,
                )
            }
            Surface(
                shape = RoundedCornerShape(999.dp),
                color = if (state.streamConnected) SmartGreen.copy(alpha = 0.16f) else SmartAmber.copy(alpha = 0.16f),
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Box(
                        modifier = Modifier
                            .width(10.dp)
                            .height(10.dp)
                            .background(if (state.streamConnected) SmartGreen else SmartAmber, CircleShape),
                    )
                    Text(
                        text = if (state.appMode == AppMode.MOCK) "Mock feed" else if (state.streamConnected) "Live feed" else "Offline",
                        style = MaterialTheme.typography.labelLarge,
                        color = MaterialTheme.colorScheme.onSurface,
                    )
                }
            }
        }

        FlowRow(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            FilterChip(
                selected = state.selectedLocalities.isEmpty(),
                onClick = { onSelectLocality(null) },
                label = { Text("All localities") },
            )
            state.localities.forEach { locality ->
                FilterChip(
                    selected = locality in state.selectedLocalities,
                    onClick = { onSelectLocality(locality) },
                    label = { Text(locality) },
                    colors = FilterChipDefaults.filterChipColors(
                        selectedContainerColor = MaterialTheme.colorScheme.primaryContainer,
                    ),
                )
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            AssistChip(
                onClick = onToggleAppMode,
                label = { Text(if (state.appMode == AppMode.MOCK) "Switch to live server" else "Switch to mock mode") },
            )
            AssistChip(
                onClick = onSelectVisibleBins,
                label = { Text("Select visible bins") },
                colors = AssistChipDefaults.assistChipColors(
                    containerColor = MaterialTheme.colorScheme.secondaryContainer,
                ),
            )
            if (state.selectedBins.isNotEmpty()) {
                AssistChip(
                    onClick = onClearSelection,
                    label = { Text("Clear selection") },
                )
            }
        }
    }
}

@Composable
private fun SelectionSummaryCard(
    state: MapState,
    onTriggerDemoEvent: (String?) -> Unit,
    modifier: Modifier = Modifier,
) {
    val hasExplicitTarget = state.explicitTriggerTargetBinId != null
    val explicitTargetName = state.bins.firstOrNull { it.id == state.explicitTriggerTargetBinId }?.name
    val watchedBin = state.watchedBin
    val selectionMetricLabel = if (state.selectedLocalities.isNotEmpty()) "Bins in scope" else "Selected"
    ElevatedCard(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.elevatedCardColors(containerColor = MaterialTheme.colorScheme.surface),
        elevation = CardDefaults.elevatedCardElevation(defaultElevation = 8.dp),
    ) {
        Column(
            modifier = Modifier.fillMaxWidth().padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text(
                text = "Live selection",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                SummaryMetric("Visible bins", state.visibleBins.size.toString())
                SummaryMetric(selectionMetricLabel, state.selectedBinIds.size.toString())
                SummaryMetric("Recent events", state.recentlyActiveBinIds.size.toString())
            }
            Text(
                text = if (state.selectedBins.isEmpty()) {
                    if (state.selectedLocalities.isEmpty()) {
                        "Tap map pins or locality chips to build a custom analytics group."
                    } else {
                        "Localities selected: ${state.selectedLocalities.joinToString()}. Tap an individual bin to switch back to custom bin mode."
                    }
                } else {
                    state.selectedBins.joinToString(separator = " • ") { it.name }
                },
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            if (watchedBin != null) {
                Text(
                    text = "Phone alerts enabled for ${watchedBin.name}",
                    style = MaterialTheme.typography.bodyMedium,
                    color = SmartGreen,
                    fontWeight = FontWeight.Medium,
                )
            }
            TextButton(
                enabled = hasExplicitTarget,
                onClick = { onTriggerDemoEvent(null) },
                modifier = Modifier.align(Alignment.End),
            ) {
                Text(
                    if (hasExplicitTarget) {
                        "Trigger live demo event${explicitTargetName?.let { " for $it" }.orEmpty()}"
                    } else {
                        "Open or watch one bin to trigger an event"
                    },
                )
            }
        }
    }
}

@Composable
private fun SummaryMetric(label: String, value: String) {
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            text = value,
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold,
        )
    }
}

@Composable
private fun BinDetailSheet(
    bin: Bin,
    classConfiguration: ResolvedWasteClassConfiguration,
    isSelected: Boolean,
    isWatched: Boolean,
    onToggleSelection: () -> Unit,
    onTriggerDemoEvent: () -> Unit,
    onToggleWatchedBin: () -> Unit,
) {
    val formatter = remember { DateTimeFormatter.ofPattern("dd MMM, HH:mm") }
    val zoneId = remember { ZoneId.systemDefault() }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 20.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(
            text = bin.name,
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold,
        )

        FlowRow(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            AssistChip(onClick = {}, label = { Text(bin.id) })
            AssistChip(
                onClick = {},
                label = { Text(bin.locality) },
                colors = AssistChipDefaults.assistChipColors(containerColor = MaterialTheme.colorScheme.primaryContainer),
            )
            AssistChip(
                onClick = {},
                label = { Text(bin.status.label) },
                colors = AssistChipDefaults.assistChipColors(containerColor = statusColor(bin.status).copy(alpha = 0.18f)),
            )
        }

        DetailRow(
            "Last event",
            bin.lastPredictedClass?.let(classConfiguration::toRuntimeDisplayLabel) ?: "No recent event",
        )
        DetailRow(
            "Last seen",
            bin.lastSeenAt?.atZone(zoneId)?.format(formatter) ?: "No data yet",
        )
        DetailRow("Today's events", bin.totalEventsToday.toString())

        HorizontalDivider()

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            TextButton(onClick = onToggleSelection) {
                Text(if (isSelected) "Remove from analytics" else "Add to analytics")
            }
            TextButton(onClick = onToggleWatchedBin) {
                Text(if (isWatched) "Disable phone alerts" else "Enable phone alerts")
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.End,
        ) {
            TextButton(onClick = onTriggerDemoEvent) {
                Text("Simulate event")
            }
        }
    }
}

@Composable
private fun DetailRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(text = label, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(text = value, style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Medium)
    }
}

@Composable
fun BinMap(
    bins: List<Bin>,
    classConfiguration: ResolvedWasteClassConfiguration,
    selectedBinIds: Set<String>,
    activeBinIds: Set<String>,
    selectedLocalities: Set<String>,
    onMarkerTapped: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val mapView = remember { MapView(context) }
    var mapInstance by remember { mutableStateOf<MapLibreMap?>(null) }
    var styleLoaded by remember { mutableStateOf(false) }
    var lastCameraKey by remember { mutableStateOf<String?>(null) }
    val markerByBinId = remember { mutableStateMapOf<String, Marker>() }
    val markerVisualStateByBinId = remember { mutableStateMapOf<String, String>() }

    DisposableEffect(mapView) {
        mapView.onStart()
        mapView.onResume()
        onDispose {
            mapView.onPause()
            mapView.onStop()
            mapView.onDestroy()
        }
    }

    LaunchedEffect(mapView) {
        mapView.getMapAsync { map ->
            map.setStyle(Style.Builder().fromUri("https://demotiles.maplibre.org/style.json")) {
                map.setOnMarkerClickListener { marker ->
                    marker.snippet?.let(onMarkerTapped)
                    true
                }
                mapInstance = map
                styleLoaded = true
            }
        }
    }

    LaunchedEffect(styleLoaded, bins, selectedBinIds, activeBinIds) {
        if (!styleLoaded) return@LaunchedEffect
        val map = mapInstance ?: return@LaunchedEffect
        val iconFactory = IconFactory.getInstance(context)
        val currentBinIds = bins.mapTo(linkedSetOf()) { it.id }
        val staleBinIds = markerByBinId.keys.toSet() - currentBinIds
        staleBinIds.forEach { staleBinId ->
            markerByBinId.remove(staleBinId)?.let(map::removeMarker)
            markerVisualStateByBinId.remove(staleBinId)
        }
        bins.forEach { bin ->
            val visualKey = listOf(
                bin.latitude.toString(),
                bin.longitude.toString(),
                bin.status.name,
                classConfiguration.toRuntimeDisplayLabel(bin.lastPredictedClass),
                (bin.id in selectedBinIds).toString(),
                (bin.id in activeBinIds).toString(),
            ).joinToString("|")
            if (markerVisualStateByBinId[bin.id] == visualKey) {
                return@forEach
            }
            markerByBinId.remove(bin.id)?.let(map::removeMarker)
            val marker = map.addMarker(
                MarkerOptions()
                    .position(LatLng(bin.latitude, bin.longitude))
                    .title(bin.name)
                    .snippet(bin.id)
                    .icon(
                        iconFactory.fromBitmap(
                            createMarkerBitmap(
                                bin = bin,
                                classConfiguration = classConfiguration,
                                isSelected = bin.id in selectedBinIds,
                                isActive = bin.id in activeBinIds,
                            ),
                        ),
                    ),
            )
            markerByBinId[bin.id] = marker
            markerVisualStateByBinId[bin.id] = visualKey
        }
        val cameraKey = buildString {
            append(selectedLocalities.sorted().joinToString("|"))
            append('#')
            append(bins.map { it.id }.sorted().joinToString("|"))
        }
        if (bins.isNotEmpty() && lastCameraKey != cameraKey) {
            if (bins.size == 1) {
                val firstBin = bins.first()
                map.animateCamera(
                    org.maplibre.android.camera.CameraUpdateFactory.newCameraPosition(
                        CameraPosition.Builder()
                            .target(LatLng(firstBin.latitude, firstBin.longitude))
                            .zoom(14.2)
                            .build(),
                    ),
                )
            } else {
                val boundsBuilder = LatLngBounds.Builder()
                bins.forEach { boundsBuilder.include(LatLng(it.latitude, it.longitude)) }
                map.animateCamera(
                    org.maplibre.android.camera.CameraUpdateFactory.newLatLngBounds(
                        boundsBuilder.build(),
                        120,
                    ),
                )
            }
            lastCameraKey = cameraKey
        }
    }

    AndroidMapContainer(mapView = mapView, modifier = modifier)
}

@Composable
private fun AndroidMapContainer(mapView: MapView, modifier: Modifier) {
    androidx.compose.ui.viewinterop.AndroidView(
        factory = { mapView },
        modifier = modifier.fillMaxSize(),
    )
}

private fun createMarkerBitmap(
    bin: Bin,
    classConfiguration: ResolvedWasteClassConfiguration,
    isSelected: Boolean,
    isActive: Boolean,
): Bitmap {
    val width = 128
    val height = 168
    val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(bitmap)

    val pinColor = when {
        isActive -> SmartGreen.toArgb()
        isSelected -> SmartBlue.toArgb()
        else -> statusColor(bin.status).toArgb()
    }

    val shadowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = android.graphics.Color.argb(48, 10, 16, 32)
    }
    canvas.drawOval(RectF(32f, 128f, 96f, 148f), shadowPaint)

    val bodyPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = pinColor
    }
    val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = android.graphics.Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }
    val path = Path().apply {
        moveTo(width / 2f, 156f)
        cubicTo(18f, 106f, 20f, 24f, width / 2f, 24f)
        cubicTo(width - 20f, 24f, width - 18f, 106f, width / 2f, 156f)
        close()
    }
    canvas.drawPath(path, bodyPaint)
    canvas.drawPath(path, strokePaint)

    val innerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = android.graphics.Color.WHITE
    }
    canvas.drawCircle(width / 2f, 68f, 26f, innerPaint)

    val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = wasteClassColor(classConfiguration.toRuntimeDisplayLabel(bin.lastPredictedClass)).toArgb()
    }
    canvas.drawCircle(width / 2f, 68f, 14f, dotPaint)

    return bitmap
}

private fun statusColor(status: BinStatus): Color = when (status) {
    BinStatus.ONLINE -> SmartBlue
    BinStatus.OFFLINE -> SmartSlate
    BinStatus.DEGRADED -> SmartAmber
}

private val SmartBlue = Color(0xFF1B5E9A)
private val SmartGreen = Color(0xFF25875A)
private val SmartAmber = Color(0xFFD9932F)
private val SmartSlate = Color(0xFF667085)
