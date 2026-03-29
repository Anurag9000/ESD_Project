package com.example.smartbin.presentation.map

import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import com.example.smartbin.domain.model.Bin
import org.maplibre.android.MapLibre
import org.maplibre.android.annotations.MarkerOptions
import org.maplibre.android.camera.CameraPosition
import org.maplibre.android.geometry.LatLng
import org.maplibre.android.maps.MapView
import org.maplibre.android.maps.Style

@Composable
fun BinMap(
    bins: List<Bin>,
    modifier: Modifier = Modifier,
    onBinClick: (String) -> Unit = {}
) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val mapView = remember {
        MapView(context).apply {
            getMapAsync { map ->
                map.setStyle(Style.getPredefinedStyle("Streets"))
                map.cameraPosition = CameraPosition.Builder()
                    .target(LatLng(40.7128, -74.0060))
                    .zoom(12.0)
                    .build()

                bins.forEach { bin ->
                    map.addMarker(
                        MarkerOptions()
                            .position(LatLng(bin.latitude, bin.longitude))
                            .title(bin.name)
                            .snippet(bin.id)
                    )
                }

                map.setOnMarkerClickListener { marker ->
                    onBinClick(marker.snippet ?: "")
                    true
                }
            }
        }
    }

    DisposableEffect(mapView) {
        mapView.onStart()
        mapView.onResume()
        onDispose {
            mapView.onPause()
            mapView.onStop()
            mapView.onDestroy()
        }
    }

    AndroidView(factory = { mapView }, modifier = modifier)
}
