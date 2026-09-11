#!/usr/bin/env python3
"""Static/source audit for the ESD v1 central scientific catalog."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "training_control"
if str(CONTROL) not in sys.path:
    sys.path.insert(0, str(CONTROL))

import esd_scientific_authority_v1 as authority  # noqa: E402

OUTPUT = ROOT / "artifacts" / "training_control" / "esd_scientific_authority_v1.json"


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    source = authority.audit_source_contract()
    ids: set[str] = set()
    families: Counter[str] = Counter()
    training_count = 0
    nontraining_count = 0
    malformed: list[dict[str, object]] = []
    main_cells: set[tuple[str, str, str, str, str, str, str, bool]] = set()
    phase0_cells: set[tuple[str, str, str, str]] = set()

    for job in authority.iter_jobs():
        job_id = str(job.get("id") or "")
        if not job_id:
            malformed.append({"reason": "missing_id"})
            continue
        if job_id in ids:
            malformed.append({"id": job_id, "reason": "duplicate_id"})
        ids.add(job_id)
        family = str(job.get("family") or "")
        families[family] += 1
        training = bool(job.get("is_training_job"))
        if training:
            training_count += 1
            if job.get("resume_strategy") != "exact_checkpoint":
                malformed.append({"id": job_id, "reason": "training_not_exact_checkpoint"})
            if job.get("early_stopping_applicable") is not True:
                malformed.append({"id": job_id, "reason": "training_missing_early_stopping"})
            if not job.get("checkpoint_artifacts"):
                malformed.append({"id": job_id, "reason": "training_missing_checkpoint_artifacts"})
            if not job.get("exact_resume_source") or not job.get("early_stopping_source"):
                malformed.append({"id": job_id, "reason": "training_missing_source_evidence"})
            command = [str(value) for value in job.get("command", [])]
            if "training_control/esd_pressure_runner_v1.py" not in command:
                malformed.append({"id": job_id, "reason": "training_bypasses_pressure_bridge"})
        else:
            nontraining_count += 1

        if family == "supervised-selector-matrix":
            main_cells.add(
                (
                    str(job["backbone"]),
                    str(job["weights"]),
                    str(job["optimizer"]),
                    str(job["precision"]),
                    str(job["classifier_train_mode"]),
                    str(job["classifier_early_stopping_metric"]),
                    str(job["sampling_strategy"]),
                    bool(job["supcon_enabled"]),
                )
            )
        elif family == "phase0_mim":
            phase0_cells.add(
                (
                    str(job["backbone"]),
                    str(job["weights"]),
                    str(job["mim_loss"]),
                    str(job["mim_scheduler"]),
                )
            )

    expected_main = set(
        __import__("itertools").product(
            authority.BACKBONES,
            authority.WEIGHTS,
            authority.OPTIMIZERS,
            authority.PRECISIONS,
            authority.CLASSIFIER_MODES,
            authority.CLASSIFIER_METRICS,
            authority.SAMPLING_STRATEGIES,
            authority.SUPCON_MODES,
        )
    )
    expected_phase0 = set(
        __import__("itertools").product(
            authority.BACKBONES,
            authority.WEIGHTS,
            authority.MIM_LOSSES,
            authority.MIM_SCHEDULERS,
        )
    )
    missing_main = sorted(expected_main - main_cells)
    missing_phase0 = sorted(expected_phase0 - phase0_cells)
    extra_main = sorted(main_cells - expected_main)
    extra_phase0 = sorted(phase0_cells - expected_phase0)
    if missing_main or extra_main:
        malformed.append(
            {
                "reason": "supervised_selector_matrix_mismatch",
                "missing_count": len(missing_main),
                "extra_count": len(extra_main),
                "missing_preview": [list(value) for value in missing_main[:30]],
                "extra_preview": [list(value) for value in extra_main[:30]],
            }
        )
    if missing_phase0 or extra_phase0:
        malformed.append(
            {
                "reason": "phase0_matrix_mismatch",
                "missing_count": len(missing_phase0),
                "extra_count": len(extra_phase0),
                "missing_preview": [list(value) for value in missing_phase0[:30]],
                "extra_preview": [list(value) for value in extra_phase0[:30]],
            }
        )

    payload = source.to_dict()
    payload.update(
        {
            "status": "PASS" if not malformed else "FAIL",
            "job_count": len(ids),
            "training_job_count": training_count,
            "nontraining_job_count": nontraining_count,
            "family_job_counts": dict(sorted(families.items())),
            "observed_main_selector_cells": len(main_cells),
            "expected_main_selector_cells": len(expected_main),
            "observed_phase0_cells": len(phase0_cells),
            "expected_phase0_cells": len(expected_phase0),
            "malformed": malformed,
            "native_step_checkpoint_resume": True,
            "semantic_early_stopping": True,
            "opf_checkpoint_ack_bridge": True,
            "post_training_evaluation_export_quantization": True,
        }
    )
    _atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not malformed else 2


if __name__ == "__main__":
    raise SystemExit(main())
