from __future__ import annotations

import unittest

from training_control import esd_dataset_cohort_catalog_v2 as catalog


class ESDDatasetCohortCatalogV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiled = catalog.compile_catalog()

    def test_logical_training_inventory_maps_exactly_once(self):
        expected = set(self.compiled["logical_jobs"])
        observed = [
            job_id
            for group in self.compiled["dataset_groups"]
            for lane in group["lanes"]
            for job_id in lane["logical_job_ids"]
        ]
        self.assertEqual(set(observed), expected)
        self.assertEqual(len(observed), len(expected))

    def test_overlap_parent_is_last(self):
        groups = self.compiled["dataset_groups"]
        self.assertTrue(self.compiled["overlap_group_last"])
        overlap_indices = [index for index, group in enumerate(groups) if group["overlap"]]
        if overlap_indices:
            self.assertEqual(overlap_indices, [len(groups) - 1])

    def test_groups_descend_by_family_then_model_count_before_overlap(self):
        non_overlap = [group for group in self.compiled["dataset_groups"] if not group["overlap"]]
        ordering = [
            (-int(group["model_family_count"]), -int(group["model_count"]), str(group["dataset_key"]))
            for group in non_overlap
        ]
        self.assertEqual(ordering, sorted(ordering))

    def test_each_lane_has_one_classifier_phase_plan(self):
        observed_plans = set()
        for group in self.compiled["dataset_groups"]:
            for lane in group["lanes"]:
                lane_plan = str(lane["classifier_phase_plan"])
                observed_plans.add(lane_plan)
                for job_id in lane["logical_job_ids"]:
                    self.assertEqual(
                        self.compiled["logical_jobs"][job_id]["classifier_phase_plan"],
                        lane_plan,
                    )
        self.assertIn("progressive", observed_plans)
        self.assertIn("full_model", observed_plans)
        self.assertIn("not_applicable", observed_plans)

    def test_progressive_and_full_model_never_share_a_lane(self):
        for group in self.compiled["dataset_groups"]:
            for lane in group["lanes"]:
                plans = {
                    self.compiled["logical_jobs"][job_id]["classifier_phase_plan"]
                    for job_id in lane["logical_job_ids"]
                }
                self.assertLessEqual(len(plans), 1)

    def test_uniform_batch_contract_is_lane_local(self):
        for group in self.compiled["dataset_groups"]:
            for lane in group["lanes"]:
                contracts = {
                    self.compiled["logical_jobs"][job_id]["physical_batch_contract"]
                    for job_id in lane["logical_job_ids"]
                }
                self.assertLessEqual(len(contracts), 1)


if __name__ == "__main__":
    unittest.main()
