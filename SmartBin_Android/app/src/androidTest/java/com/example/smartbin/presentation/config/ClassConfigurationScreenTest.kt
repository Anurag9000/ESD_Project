package com.example.smartbin.presentation.config

import androidx.activity.ComponentActivity
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.hasSetTextAction
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performTextClearance
import androidx.compose.ui.test.performTextInput
import com.example.smartbin.domain.model.WasteClassCatalog
import com.example.smartbin.domain.model.WasteClassConfiguration
import com.example.smartbin.domain.model.resolve
import org.junit.Ignore
import org.junit.Rule
import org.junit.Test

class ClassConfigurationScreenTest {

    @get:Rule
    val composeRule = createAndroidComposeRule<ComponentActivity>()

    private val catalog = WasteClassCatalog(
        availableRawClasses = listOf("battery", "glass", "metal", "organic", "paper", "plastic"),
        recommendedPrimaryClasses = listOf("metal", "organic", "paper"),
        otherLabel = "Other",
    )

    @Ignore("Compose input automation on API 36 fails in this environment; covered by manual emulator verification.")
    @Test
    fun screenUpdatesClassCountAndOtherPreview() {
        var uiState by mutableStateOf(
            ClassConfigurationState(
                catalog = catalog,
                savedConfiguration = catalog.resolve(WasteClassConfiguration()),
                draftClassCountText = "4",
                draftSelectedPrimaryClasses = listOf("metal", "organic", "paper"),
                draftResolvedConfiguration = catalog.resolve(WasteClassConfiguration()),
                requiresInitialConfirmation = true,
            ),
        )

        fun updateDraft(classCountText: String, selectedPrimaryClasses: List<String>) {
            val resolved = catalog.resolve(
                WasteClassConfiguration(
                    classCount = classCountText.toIntOrNull() ?: 4,
                    selectedPrimaryClasses = selectedPrimaryClasses,
                    userConfirmed = false,
                ),
            )
            uiState = uiState.copy(
                draftClassCountText = classCountText,
                draftSelectedPrimaryClasses = resolved.selectedPrimaryClasses,
                draftResolvedConfiguration = resolved,
            )
        }

        composeRule.setContent {
            ClassConfigurationScreen(
                state = uiState,
                onClassCountChanged = { updateDraft(it, uiState.draftSelectedPrimaryClasses) },
                onPrimaryClassSelected = { index, rawClass ->
                    val selections = uiState.draftSelectedPrimaryClasses.toMutableList()
                    while (selections.size <= index) {
                        selections += ""
                    }
                    selections[index] = rawClass
                    updateDraft(uiState.draftClassCountText, selections)
                },
                onReset = {},
                onSave = {},
                mandatory = true,
            )
        }

        composeRule.onNodeWithText("Configure runtime classes").assertIsDisplayed()
        composeRule.onNodeWithText("Class 4").assertIsDisplayed()
        composeRule.onNodeWithText("Raw classes that currently collapse into Other: battery, glass, plastic").assertIsDisplayed()

        composeRule.onNode(hasSetTextAction()).performTextClearance()
        composeRule.onNode(hasSetTextAction()).performTextInput("5")

        composeRule.onNodeWithText("Class 5").assertIsDisplayed()
        composeRule.onNodeWithText("Raw classes that currently collapse into Other: plastic").assertIsDisplayed()
    }
}
