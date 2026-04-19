package com.example.smartbin

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.example.smartbin.notifications.BinAlertNotifier
import com.example.smartbin.presentation.MainScreen
import com.example.smartbin.ui.theme.SmartBinTheme
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    private var notificationOpenBinId by mutableStateOf<String?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        notificationOpenBinId = intent?.getStringExtra(BinAlertNotifier.EXTRA_OPEN_BIN_ID)
        enableEdgeToEdge()
        setContent {
            SmartBinTheme {
                MainScreen(
                    notificationOpenBinId = notificationOpenBinId,
                    onNotificationBinConsumed = { notificationOpenBinId = null },
                )
            }
        }
    }

    override fun onNewIntent(intent: android.content.Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        notificationOpenBinId = intent.getStringExtra(BinAlertNotifier.EXTRA_OPEN_BIN_ID)
    }
}
