package com.example.smartbin.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val ColorWhite = Color.White

private val LightColorScheme = lightColorScheme(
    primary = SmartBlue,
    onPrimary = ColorWhite,
    primaryContainer = SmartSky,
    onPrimaryContainer = SmartNavy,
    secondary = SmartNavy,
    onSecondary = ColorWhite,
    secondaryContainer = Color(0xFFE6EEF8),
    onSecondaryContainer = SmartNavy,
    tertiary = WasteOther,
    onTertiary = ColorWhite,
    background = SmartSurface,
    onBackground = SmartText,
    surface = SmartCard,
    onSurface = SmartText,
    surfaceVariant = Color(0xFFE9EEF5),
    onSurfaceVariant = SmartMuted,
    outline = SmartOutline,
)

private val DarkColorScheme = darkColorScheme(
    primary = SmartBlueDark,
    onPrimary = Color(0xFF06233E),
    primaryContainer = Color(0xFF153552),
    onPrimaryContainer = SmartTextDark,
    secondary = SmartNavyDark,
    onSecondary = Color(0xFF0B1E36),
    secondaryContainer = Color(0xFF1B2940),
    onSecondaryContainer = SmartTextDark,
    tertiary = Color(0xFFFFC066),
    onTertiary = Color(0xFF2F2200),
    background = SmartSurfaceDark,
    onBackground = SmartTextDark,
    surface = SmartCardDark,
    onSurface = SmartTextDark,
    surfaceVariant = Color(0xFF1A2638),
    onSurfaceVariant = SmartMutedDark,
    outline = Color(0xFF33435D),
)

@Composable
fun SmartBinTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    MaterialTheme(
        colorScheme = if (darkTheme) DarkColorScheme else LightColorScheme,
        typography = Typography,
        content = content,
    )
}
