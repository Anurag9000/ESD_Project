package com.example.smartbin.domain.model

import java.time.Instant

data class WasteClassOption(
    val rawClass: String,
    val displayLabel: String,
)

data class WasteClassCatalog(
    val availableRawClasses: List<String>,
    val recommendedPrimaryClasses: List<String>,
    val otherLabel: String = "Other",
) {
    companion object {
        val EMPTY = WasteClassCatalog(
            availableRawClasses = emptyList(),
            recommendedPrimaryClasses = emptyList(),
            otherLabel = "Other",
        )
    }
}

data class WasteClassConfiguration(
    val classCount: Int = 4,
    val selectedPrimaryClasses: List<String> = emptyList(),
    val mergedAssignments: Map<String, String> = emptyMap(),
    val userConfirmed: Boolean = false,
)

data class ResolvedWasteClassGroup(
    val primaryRawClass: String,
    val displayLabel: String,
    val rawClasses: List<String>,
)

data class ResolvedWasteClassConfiguration(
    val classCount: Int,
    val selectedPrimaryClasses: List<String>,
    val selectedDisplayLabels: List<String>,
    val groups: List<ResolvedWasteClassGroup>,
    val mergedAssignments: Map<String, String>,
    val otherLabel: String,
    val runtimeDisplayLabels: List<String>,
    val availableOptions: List<WasteClassOption>,
    val hasExplicitUserConfiguration: Boolean,
) {
    fun toRuntimeDisplayLabel(rawClass: String?): String {
        if (rawClass.isNullOrBlank()) return otherLabel
        val normalized = rawClass.trim()
        return groups.firstOrNull { normalized in it.rawClasses }?.displayLabel ?: otherLabel
    }

    fun remainingRawClasses(): List<String> {
        val groupedRawClasses = groups.flatMap { it.rawClasses }.toSet()
        return availableOptions.map { it.rawClass }.filterNot { it in groupedRawClasses }
    }

    fun mergedRawClassesFor(primaryRawClass: String): List<String> {
        return groups.firstOrNull { it.primaryRawClass == primaryRawClass }
            ?.rawClasses
            .orEmpty()
            .filterNot { it == primaryRawClass }
    }
}

fun formatWasteClassLabel(rawClass: String): String {
    return rawClass
        .trim()
        .replace('-', ' ')
        .replace('_', ' ')
        .split(Regex("\\s+"))
        .filter { it.isNotBlank() }
        .joinToString(" ") { token ->
            token.lowercase().replaceFirstChar { char ->
                if (char.isLowerCase()) char.titlecase() else char.toString()
            }
        }
}

fun WasteClassCatalog.resolve(configuration: WasteClassConfiguration): ResolvedWasteClassConfiguration {
    val available = availableRawClasses.distinct()
    val maxClassCount = available.size.coerceAtLeast(2)
    val normalizedClassCount = configuration.classCount.coerceIn(2, maxClassCount)
    val requiredPrimaryCount = (normalizedClassCount - 1).coerceAtLeast(1)
    val normalizedPrimary = linkedSetOf<String>()

    configuration.selectedPrimaryClasses.forEach { rawClass ->
        if (rawClass in available) {
            normalizedPrimary += rawClass
        }
    }
    recommendedPrimaryClasses.forEach { rawClass ->
        if (rawClass in available && normalizedPrimary.size < requiredPrimaryCount) {
            normalizedPrimary += rawClass
        }
    }
    available.forEach { rawClass ->
        if (normalizedPrimary.size < requiredPrimaryCount) {
            normalizedPrimary += rawClass
        }
    }

    val selectedPrimary = normalizedPrimary.take(requiredPrimaryCount)
    val selectedPrimarySet = selectedPrimary.toSet()
    val sanitizedMergedAssignments = linkedMapOf<String, String>()
    configuration.mergedAssignments.forEach { (rawClass, groupPrimary) ->
        if (rawClass !in available) return@forEach
        if (groupPrimary !in selectedPrimarySet) return@forEach
        if (rawClass in selectedPrimarySet) return@forEach
        sanitizedMergedAssignments[rawClass] = groupPrimary
    }
    val groups = selectedPrimary.map { primaryRawClass ->
        val groupedRawClasses = buildList {
            add(primaryRawClass)
            addAll(
                available.filter { rawClass ->
                    rawClass !in selectedPrimarySet && sanitizedMergedAssignments[rawClass] == primaryRawClass
                },
            )
        }
        ResolvedWasteClassGroup(
            primaryRawClass = primaryRawClass,
            displayLabel = formatWasteClassLabel(primaryRawClass),
            rawClasses = groupedRawClasses,
        )
    }
    val selectedDisplay = groups.map { it.displayLabel }
    return ResolvedWasteClassConfiguration(
        classCount = normalizedClassCount,
        selectedPrimaryClasses = selectedPrimary,
        selectedDisplayLabels = selectedDisplay,
        groups = groups,
        mergedAssignments = sanitizedMergedAssignments,
        otherLabel = otherLabel,
        runtimeDisplayLabels = selectedDisplay + otherLabel,
        availableOptions = available.map { rawClass ->
            WasteClassOption(rawClass = rawClass, displayLabel = formatWasteClassLabel(rawClass))
        },
        hasExplicitUserConfiguration = configuration.userConfirmed,
    )
}

enum class BinStatus(val label: String) {
    ONLINE("Online"),
    OFFLINE("Offline"),
    DEGRADED("Degraded");
}

data class Bin(
    val id: String,
    val name: String,
    val latitude: Double,
    val longitude: Double,
    val locality: String,
    val status: BinStatus,
    val lastSeenAt: Instant? = null,
    val installedAt: Instant? = null,
    val lastPredictedClass: String? = null,
    val totalEventsToday: Int = 0,
)

data class WasteEvent(
    val id: String,
    val binId: String,
    val predictedClass: String,
    val confidence: Float,
    val timestamp: Instant,
    val uploadedAt: Instant = timestamp,
    val sourceDeviceId: String = "raspberry-pi-4",
    val modelVersion: String = "efficientnet-b0-ce-v1",
)

data class TimeRange(
    val start: Instant,
    val end: Instant,
)
