package com.example.smartbin

import android.app.Application
import android.content.pm.ApplicationInfo
import com.example.smartbin.notifications.BinAlertNotifier
import dagger.hilt.android.HiltAndroidApp
import javax.inject.Inject
import timber.log.Timber

@HiltAndroidApp
class SmartBinApplication : Application() {
    @Inject
    lateinit var binAlertNotifier: BinAlertNotifier

    override fun onCreate() {
        super.onCreate()
        binAlertNotifier.createNotificationChannel()
        if (applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0) {
            Timber.plant(Timber.DebugTree())
        }
    }
}
