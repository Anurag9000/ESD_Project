package com.example.smartbin.presentation.config

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AssistChip
import androidx.compose.material3.AssistChipDefaults
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.smartbin.presentation.wasteClassColor

@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun ClassConfigurationScreen(
    state: ClassConfigurationState,
    onClassCountChanged: (String) -> Unit,
    onPrimaryClassSelected: (Int, String) -> Unit,
    onMergedRawClassToggled: (String, String) -> Unit,
    onReset: () -> Unit,
    onSave: () -> Unit,
    modifier: Modifier = Modifier,
    mandatory: Boolean = false,
) {
    val resolved = state.draftResolvedConfiguration
    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .navigationBarsPadding()
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp),
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(
                text = if (mandatory) "Configure runtime classes" else "Runtime class grouping",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = "The model still predicts every trained waste label. Here you choose how many runtime classes the app should expose. Each explicit runtime class can merge multiple raw model classes into one display bucket. The last class is always Other and contains every raw class you did not assign.",
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        Card(
            shape = MaterialTheme.shapes.extraLarge,
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow),
        ) {
            Column(
                modifier = Modifier.fillMaxWidth().padding(18.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                OutlinedTextField(
                    value = state.draftClassCountText,
                    onValueChange = onClassCountChanged,
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Number of runtime classes") },
                    supportingText = {
                        Text("Default is 4. Minimum is 2. Maximum is ${state.catalog.availableRawClasses.size}.")
                    },
                    singleLine = true,
                )

                resolved.selectedPrimaryClasses.forEachIndexed { index, currentRawClass ->
                    var expanded by remember(index, currentRawClass, resolved.availableOptions) { mutableStateOf(false) }
                    val selectedElsewhere = resolved.selectedPrimaryClasses.filterIndexed { currentIndex, _ -> currentIndex != index }.toSet()
                    val candidateOptions = resolved.availableOptions.filter { option ->
                        option.rawClass == currentRawClass || option.rawClass !in selectedElsewhere
                    }
                    ExposedDropdownMenuBox(
                        expanded = expanded,
                        onExpandedChange = { expanded = !expanded },
                    ) {
                        OutlinedTextField(
                            value = candidateOptions.firstOrNull { it.rawClass == currentRawClass }?.displayLabel.orEmpty(),
                            onValueChange = {},
                            readOnly = true,
                            modifier = Modifier
                                .menuAnchor()
                                .fillMaxWidth(),
                            label = { Text("Class ${index + 1}") },
                            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
                        )
                        ExposedDropdownMenu(
                            expanded = expanded,
                            onDismissRequest = { expanded = false },
                        ) {
                            candidateOptions.forEach { option ->
                                DropdownMenuItem(
                                    text = { Text(option.displayLabel) },
                                    onClick = {
                                        onPrimaryClassSelected(index, option.rawClass)
                                        expanded = false
                                    },
                                )
                            }
                        }
                    }

                    val mergedForCurrent = resolved.mergedRawClassesFor(currentRawClass)
                    val mergeCandidates = resolved.availableOptions.filter { option ->
                        option.rawClass != currentRawClass &&
                            option.rawClass !in selectedElsewhere &&
                            (
                                resolved.mergedAssignments[option.rawClass] == null ||
                                    resolved.mergedAssignments[option.rawClass] == currentRawClass
                                )
                    }
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(
                            text = "Also show these raw classes as ${formatLabelForSummary(currentRawClass)}",
                            style = MaterialTheme.typography.labelLarge,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        if (mergeCandidates.isEmpty()) {
                            Text(
                                text = "No additional raw classes available for this bucket.",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        } else {
                            FlowRow(
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                verticalArrangement = Arrangement.spacedBy(8.dp),
                            ) {
                                mergeCandidates.forEach { option ->
                                    val isSelected = option.rawClass in mergedForCurrent
                                    AssistChip(
                                        onClick = { onMergedRawClassToggled(currentRawClass, option.rawClass) },
                                        label = { Text(option.displayLabel) },
                                        colors = AssistChipDefaults.assistChipColors(
                                            containerColor = if (isSelected) {
                                                MaterialTheme.colorScheme.primaryContainer
                                            } else {
                                                MaterialTheme.colorScheme.surfaceContainerHighest
                                            },
                                        ),
                                    )
                                }
                            }
                        }
                        if (mergedForCurrent.isNotEmpty()) {
                            Text(
                                text = "Merged here: ${mergedForCurrent.joinToString { formatLabelForSummary(it) }}",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                }

                OutlinedTextField(
                    value = resolved.otherLabel,
                    onValueChange = {},
                    readOnly = true,
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Class ${resolved.classCount}") },
                    supportingText = {
                        Text("This bucket is fixed. It merges every raw class not selected above.")
                    },
                )
            }
        }

        Card(
            shape = MaterialTheme.shapes.extraLarge,
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLow),
        ) {
            Column(
                modifier = Modifier.fillMaxWidth().padding(18.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                Text("Runtime classes preview", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    resolved.runtimeDisplayLabels.forEach { label ->
                        AssistChip(
                            onClick = {},
                            label = { Text(label) },
                            leadingIcon = {
                                Box(
                                    modifier = Modifier.clip(CircleShape),
                                ) {
                                    Surface(
                                        modifier = Modifier.padding(2.dp),
                                        shape = CircleShape,
                                        color = wasteClassColor(label),
                                    ) {
                                        Box(modifier = Modifier.padding(5.dp))
                                    }
                                }
                            },
                        )
                    }
                }
                Text(
                    text = "Raw classes that currently collapse into ${resolved.otherLabel}: ${resolved.remainingRawClasses().joinToString { formatLabelForSummary(it) }}",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp, Alignment.End),
        ) {
            if (!mandatory) {
                TextButton(onClick = onReset) {
                    Text("Reset")
                }
            }
            Button(onClick = onSave) {
                Text(if (mandatory) "Save and continue" else "Save configuration")
            }
        }
    }
}

private fun formatLabelForSummary(rawClass: String): String = rawClass
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
