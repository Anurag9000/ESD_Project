package com.example.smartbin.presentation

import androidx.compose.ui.graphics.Color
import kotlin.math.absoluteValue

private val WasteClassPalette = listOf(
    Color(0xFF1B5E9A),
    Color(0xFF25875A),
    Color(0xFF7C8799),
    Color(0xFF8A5CF6),
    Color(0xFFE76F51),
    Color(0xFF2A9D8F),
    Color(0xFF4F7DF3),
    Color(0xFFC77DFF),
    Color(0xFFBC6C25),
)

fun wasteClassColor(label: String?): Color {
    if (label.isNullOrBlank()) return Color(0xFFD9932F)
    if (label.equals("Other", ignoreCase = true)) return Color(0xFFD9932F)
    val index = label.lowercase().hashCode().absoluteValue % WasteClassPalette.size
    return WasteClassPalette[index]
}
