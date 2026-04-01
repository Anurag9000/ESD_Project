package com.example.smartbin.data.repository

import android.content.Context
import com.example.smartbin.domain.model.ResolvedWasteClassConfiguration
import com.example.smartbin.domain.model.WasteClassCatalog
import com.example.smartbin.domain.model.WasteClassConfiguration
import com.example.smartbin.domain.model.resolve
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

@Serializable
private data class WasteClassCatalogAsset(
    @SerialName("available_raw_classes") val availableRawClasses: List<String>,
    @SerialName("recommended_primary_classes") val recommendedPrimaryClasses: List<String> = emptyList(),
    @SerialName("other_label") val otherLabel: String = "Other",
)

@Singleton
class WasteClassConfigStore @Inject constructor(
    @ApplicationContext context: Context,
) {
    private val preferences = context.getSharedPreferences("smartbin_class_config", Context.MODE_PRIVATE)
    private val json = Json { ignoreUnknownKeys = true }
    private val loadedCatalog = run {
        val assetText = context.assets.open("waste_class_catalog.json").bufferedReader().use { it.readText() }
        val asset = json.decodeFromString<WasteClassCatalogAsset>(assetText)
        WasteClassCatalog(
            availableRawClasses = asset.availableRawClasses,
            recommendedPrimaryClasses = asset.recommendedPrimaryClasses,
            otherLabel = asset.otherLabel,
        )
    }

    private val _catalog = MutableStateFlow(loadedCatalog)
    private val _configuration = MutableStateFlow(loadSavedConfiguration())
    private val _resolvedConfiguration = MutableStateFlow(loadedCatalog.resolve(_configuration.value))

    val catalog: StateFlow<WasteClassCatalog> = _catalog.asStateFlow()
    val configuration: StateFlow<WasteClassConfiguration> = _configuration.asStateFlow()
    val resolvedConfiguration: StateFlow<ResolvedWasteClassConfiguration> = _resolvedConfiguration.asStateFlow()

    fun saveConfiguration(classCount: Int, selectedPrimaryClasses: List<String>) {
        val configuration = WasteClassConfiguration(
            classCount = classCount,
            selectedPrimaryClasses = selectedPrimaryClasses,
            userConfirmed = true,
        )
        persist(configuration)
        _configuration.value = configuration
        _resolvedConfiguration.value = _catalog.value.resolve(configuration)
    }

    fun resetToResolvedDefaults() {
        val defaultConfiguration = WasteClassConfiguration()
        persist(defaultConfiguration)
        _configuration.value = defaultConfiguration
        _resolvedConfiguration.value = _catalog.value.resolve(defaultConfiguration)
    }

    private fun loadSavedConfiguration(): WasteClassConfiguration {
        val savedClassCount = preferences.getInt(KEY_CLASS_COUNT, 4)
        val savedSelection = preferences.getString(KEY_SELECTED_PRIMARY_CLASSES_JSON, null)
            ?.let { encoded ->
                runCatching { json.decodeFromString<List<String>>(encoded) }.getOrNull()
            }
            ?: preferences.getStringSet(KEY_SELECTED_PRIMARY_CLASSES, emptySet()).orEmpty().toList()
        val userConfirmed = preferences.getBoolean(KEY_USER_CONFIRMED, false)
        return WasteClassConfiguration(
            classCount = savedClassCount,
            selectedPrimaryClasses = savedSelection,
            userConfirmed = userConfirmed,
        )
    }

    private fun persist(configuration: WasteClassConfiguration) {
        preferences.edit()
            .putInt(KEY_CLASS_COUNT, configuration.classCount)
            .putString(
                KEY_SELECTED_PRIMARY_CLASSES_JSON,
                json.encodeToString(ListSerializer(String.serializer()), configuration.selectedPrimaryClasses),
            )
            .putStringSet(KEY_SELECTED_PRIMARY_CLASSES, configuration.selectedPrimaryClasses.toSet())
            .putBoolean(KEY_USER_CONFIRMED, configuration.userConfirmed)
            .apply()
    }

    companion object {
        private const val KEY_CLASS_COUNT = "class_count"
        private const val KEY_SELECTED_PRIMARY_CLASSES = "selected_primary_classes"
        private const val KEY_SELECTED_PRIMARY_CLASSES_JSON = "selected_primary_classes_json"
        private const val KEY_USER_CONFIRMED = "user_confirmed"
    }
}
