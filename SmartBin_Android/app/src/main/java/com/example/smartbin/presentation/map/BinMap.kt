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
import com.example.smartbin.data.repository.AppMode
import com.example.smartbin.domain.model.Bin
import com.example.smartbin.domain.model.BinStatus
import com.example.smartbin.domain.model.WasteType
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import org.maplibre.android.annotations.IconFactory
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
    onMarkerTapped: (String) -> Unit,
    onToggleBinSelection: (String) -> Unit,
    onSelectLocality: (String?) -> Unit,
    onSelectVisibleBins: () -> Unit,
    onClearSelection: () -> Unit,
    onDismissDetails: () -> Unit,
    onTriggerDemoEvent: (String?) -> Unit,
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
            BinMap(
                bins = state.visibleBins,
                selectedBinIds = state.selectedBinIds,
                activeBinId = state.recentlyActiveBinId,
                selectedLocality = state.selectedLocality,
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
                isSelected = detailBin.id in state.selectedBinIds,
                onToggleSelection = { onToggleBinSelection(detailBin.id) },
                onTriggerDemoEvent = { onTriggerDemoEvent(detailBin.id) },
            )
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
                selected = state.selectedLocality == null,
                onClick = { onSelectLocality(null) },
                label = { Text("All localities") },
            )
            state.localities.forEach { locality ->
                FilterChip(
                    selected = state.selectedLocality == locality,
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
                SummaryMetric("Selected", state.selectedBinIds.size.toString())
                SummaryMetric("Recent event", state.recentlyActiveBinId ?: "None")
            }
            Text(
                text = if (state.selectedBins.isEmpty()) {
                    "Tap map pins to build a custom analytics group."
                } else {
                    state.selectedBins.joinToString(separator = " • ") { it.name }
                },
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            TextButton(onClick = { onTriggerDemoEvent(null) }, modifier = Modifier.align(Alignment.End)) {
                Text("Trigger live demo event")
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
    isSelected: Boolean,
    onToggleSelection: () -> Unit,
    onTriggerDemoEvent: () -> Unit,
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

        DetailRow("Last event", bin.lastWasteType?.label ?: "No recent event")
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
    selectedBinIds: Set<String>,
    activeBinId: String?,
    selectedLocality: String?,
    onMarkerTapped: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val mapView = remember { MapView(context) }
    var mapInstance by remember { mutableStateOf<MapLibreMap?>(null) }
    var styleLoaded by remember { mutableStateOf(false) }
    var lastCameraLocality by remember { mutableStateOf<String?>(null) }

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

    LaunchedEffect(styleLoaded, bins, selectedBinIds, activeBinId) {
        if (!styleLoaded) return@LaunchedEffect
        val map = mapInstance ?: return@LaunchedEffect
        map.clear()
        val iconFactory = IconFactory.getInstance(context)
        bins.forEach { bin ->
            map.addMarker(
                MarkerOptions()
                    .position(LatLng(bin.latitude, bin.longitude))
                    .title(bin.name)
                    .snippet(bin.id)
                    .icon(
                        iconFactory.fromBitmap(
                            createMarkerBitmap(
                                bin = bin,
                                isSelected = bin.id in selectedBinIds,
                                isActive = bin.id == activeBinId,
                            ),
                        ),
                    ),
            )
        }
        if (bins.isNotEmpty() && (lastCameraLocality != selectedLocality || lastCameraLocality == null)) {
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
            lastCameraLocality = selectedLocality
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

private fun createMarkerBitmap(bin: Bin, isSelected: Boolean, isActive: Boolean): Bitmap {
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
        color = wasteColor(bin.lastWasteType).toArgb()
    }
    canvas.drawCircle(width / 2f, 68f, 14f, dotPaint)

    return bitmap
}

private fun statusColor(status: BinStatus): Color = when (status) {
    BinStatus.ONLINE -> SmartBlue
    BinStatus.OFFLINE -> SmartSlate
    BinStatus.DEGRADED -> SmartAmber
}

private fun wasteColor(type: WasteType?): Color = when (type) {
    WasteType.METAL -> Color(0xFF7C8799)
    WasteType.ORGANIC -> SmartGreen
    WasteType.PAPER -> Color(0xFF4F7DF3)
    WasteType.OTHER, null -> SmartAmber
}

private val SmartBlue = Color(0xFF1B5E9A)
private val SmartGreen = Color(0xFF25875A)
private val SmartAmber = Color(0xFFD9932F)
private val SmartSlate = Color(0xFF667085)
