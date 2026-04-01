package com.example.smartbin.presentation.config

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.smartbin.data.repository.WasteClassConfigStore
import com.example.smartbin.domain.model.ResolvedWasteClassConfiguration
import com.example.smartbin.domain.model.WasteClassCatalog
import com.example.smartbin.domain.model.WasteClassConfiguration
import com.example.smartbin.domain.model.resolve
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class ClassConfigurationState(
    val catalog: WasteClassCatalog = WasteClassCatalog.EMPTY,
    val savedConfiguration: ResolvedWasteClassConfiguration = WasteClassCatalog.EMPTY.resolve(WasteClassConfiguration()),
    val draftClassCountText: String = "4",
    val draftSelectedPrimaryClasses: List<String> = emptyList(),
    val draftResolvedConfiguration: ResolvedWasteClassConfiguration = WasteClassCatalog.EMPTY.resolve(WasteClassConfiguration()),
    val requiresInitialConfirmation: Boolean = false,
)

@HiltViewModel
class ClassConfigurationViewModel @Inject constructor(
    private val wasteClassConfigStore: WasteClassConfigStore,
) : ViewModel() {

    private val _state = MutableStateFlow(ClassConfigurationState())
    val state: StateFlow<ClassConfigurationState> = _state.asStateFlow()

    init {
        viewModelScope.launch {
            wasteClassConfigStore.catalog.collectLatest { catalog ->
                val currentConfiguration = wasteClassConfigStore.configuration.value
                val resolved = catalog.resolve(currentConfiguration)
                _state.update {
                    it.copy(
                        catalog = catalog,
                        savedConfiguration = resolved,
                        draftClassCountText = resolved.classCount.toString(),
                        draftSelectedPrimaryClasses = resolved.selectedPrimaryClasses,
                        draftResolvedConfiguration = resolved,
                        requiresInitialConfirmation = !resolved.hasExplicitUserConfiguration,
                    )
                }
            }
        }
    }

    fun onClassCountChanged(value: String) {
        val digitsOnly = value.filter { it.isDigit() }
        val fallback = state.value.savedConfiguration.classCount
        val parsed = digitsOnly.toIntOrNull() ?: fallback
        updateDraft(
            classCount = parsed,
            selectedPrimaryClasses = state.value.draftSelectedPrimaryClasses,
            rawText = if (digitsOnly.isBlank()) fallback.toString() else digitsOnly,
        )
    }

    fun onPrimaryClassSelected(index: Int, rawClass: String) {
        val selections = state.value.draftSelectedPrimaryClasses.toMutableList()
        while (selections.size <= index) {
            selections += ""
        }
        selections[index] = rawClass
        updateDraft(
            classCount = state.value.draftClassCountText.toIntOrNull() ?: state.value.savedConfiguration.classCount,
            selectedPrimaryClasses = selections,
            rawText = state.value.draftClassCountText,
        )
    }

    fun resetDraft() {
        val resolved = state.value.savedConfiguration
        _state.update {
            it.copy(
                draftClassCountText = resolved.classCount.toString(),
                draftSelectedPrimaryClasses = resolved.selectedPrimaryClasses,
                draftResolvedConfiguration = resolved,
            )
        }
    }

    fun saveDraft() {
        val resolved = state.value.draftResolvedConfiguration
        wasteClassConfigStore.saveConfiguration(
            classCount = resolved.classCount,
            selectedPrimaryClasses = resolved.selectedPrimaryClasses,
        )
        _state.update {
            it.copy(
                savedConfiguration = resolved.copy(hasExplicitUserConfiguration = true),
                draftResolvedConfiguration = resolved.copy(hasExplicitUserConfiguration = true),
                requiresInitialConfirmation = false,
            )
        }
    }

    private fun updateDraft(classCount: Int, selectedPrimaryClasses: List<String>, rawText: String) {
        val catalog = state.value.catalog
        val draft = catalog.resolve(
            WasteClassConfiguration(
                classCount = classCount,
                selectedPrimaryClasses = selectedPrimaryClasses.filter { it.isNotBlank() },
                userConfirmed = state.value.savedConfiguration.hasExplicitUserConfiguration,
            ),
        )
        _state.update {
            it.copy(
                draftClassCountText = rawText,
                draftSelectedPrimaryClasses = draft.selectedPrimaryClasses,
                draftResolvedConfiguration = draft,
            )
        }
    }
}
