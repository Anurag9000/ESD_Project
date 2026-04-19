package com.example.smartbin.notifications

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import com.example.smartbin.MainActivity
import com.example.smartbin.R
import com.example.smartbin.domain.model.WasteEvent
import com.example.smartbin.data.repository.WasteClassConfigStore
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class BinAlertNotifier @Inject constructor(
    @ApplicationContext private val context: Context,
    private val wasteClassConfigStore: WasteClassConfigStore,
) {

    fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val manager = context.getSystemService(NotificationManager::class.java) ?: return
        val channel = NotificationChannel(
            CHANNEL_ID,
            "SmartBin Alerts",
            NotificationManager.IMPORTANCE_HIGH,
        ).apply {
            description = "Waste detection alerts for watched SmartBin devices"
        }
        manager.createNotificationChannel(channel)
    }

    fun showWasteDetectedNotification(binId: String, binName: String, event: WasteEvent) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            return
        }
        val intent = Intent(context, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
            putExtra(EXTRA_OPEN_BIN_ID, binId)
        }
        val pendingIntent = PendingIntent.getActivity(
            context,
            event.id.hashCode(),
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val displayLabel = wasteClassConfigStore.resolvedConfiguration.value.toRuntimeDisplayLabel(event.predictedClass)
        val contentText = "$binName detected ${displayLabel.lowercase()} at ${(event.confidence * 100).toInt()}% confidence"
        val notification = NotificationCompat.Builder(context, CHANNEL_ID)
            .setSmallIcon(R.mipmap.ic_launcher_round)
            .setContentTitle("Waste detected in watched bin")
            .setContentText(contentText)
            .setStyle(NotificationCompat.BigTextStyle().bigText(contentText))
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setContentIntent(pendingIntent)
            .setAutoCancel(true)
            .build()
        NotificationManagerCompat.from(context).notify(event.id.hashCode(), notification)
    }

    companion object {
        const val CHANNEL_ID = "smartbin_waste_alerts"
        const val EXTRA_OPEN_BIN_ID = "extra_open_bin_id"
    }
}
