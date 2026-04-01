package com.example.smartbin.data.remote

import android.os.Build
import com.example.smartbin.BuildConfig

private const val DEFAULT_EMULATOR_HTTP = "http://10.0.2.2:8000/"
private const val DEFAULT_EMULATOR_WS = "ws://10.0.2.2:8000/events/stream"
private const val DEFAULT_DEVICE_HTTP = "http://127.0.0.1:8000/"
private const val DEFAULT_DEVICE_WS = "ws://127.0.0.1:8000/events/stream"

fun resolvedApiBaseUrl(): String {
    return if (BuildConfig.API_BASE_URL == DEFAULT_EMULATOR_HTTP && !isProbablyEmulator()) {
        DEFAULT_DEVICE_HTTP
    } else {
        BuildConfig.API_BASE_URL
    }
}

fun resolvedWsEventsUrl(): String {
    return if (BuildConfig.WS_EVENTS_URL == DEFAULT_EMULATOR_WS && !isProbablyEmulator()) {
        DEFAULT_DEVICE_WS
    } else {
        BuildConfig.WS_EVENTS_URL
    }
}

fun liveServerHint(): String {
    return if (isProbablyEmulator()) {
        "Expected dev server at 10.0.2.2:8000"
    } else {
        "Expected dev server at 127.0.0.1:8000 via adb reverse, or override SMARTBIN_API_BASE_URL / SMARTBIN_WS_EVENTS_URL"
    }
}

private fun isProbablyEmulator(): Boolean {
    val fingerprint = Build.FINGERPRINT.orEmpty()
    val model = Build.MODEL.orEmpty()
    val manufacturer = Build.MANUFACTURER.orEmpty()
    val brand = Build.BRAND.orEmpty()
    val device = Build.DEVICE.orEmpty()
    val product = Build.PRODUCT.orEmpty()
    return fingerprint.startsWith("generic") ||
        fingerprint.contains("emulator", ignoreCase = true) ||
        fingerprint.contains("vbox", ignoreCase = true) ||
        model.contains("sdk_gphone", ignoreCase = true) ||
        model.contains("Emulator", ignoreCase = true) ||
        manufacturer.contains("Genymotion", ignoreCase = true) ||
        (brand.startsWith("generic") && device.startsWith("generic")) ||
        product.contains("sdk", ignoreCase = true)
}
