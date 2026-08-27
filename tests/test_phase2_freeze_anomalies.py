import unittest

import pandas as pd

from src.endpoints import build_endpoint


class Phase2KnownAnomalyFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.subjects = pd.read_parquet("data/cohort/subjects.parquet")

    def test_undefined_holter_rhythm_code_4_is_not_decoded(self):
        rows = self.subjects.loc[self.subjects["holter_rhythm_raw"].eq("4")].set_index("patient_id")
        self.assertEqual(set(rows.index), {"P0269", "P0304", "P0924", "P0998"})
        self.assertTrue(rows["rhythm_raw"].eq("4").all())
        self.assertTrue(rows["rhythm_decoded"].isna().all())

        official_hr_af = rows["ecg_rhythm_raw"].eq("1")
        self.assertEqual(set(rows.index[official_hr_af]), {"P0269", "P0924"})
        self.assertTrue(rows.loc[official_hr_af, "af_flag"].all())
        self.assertTrue(rows.loc[official_hr_af, "rhythm_group"].eq("AF").all())
        self.assertTrue(rows.loc[~official_hr_af, "rhythm_group"].eq("UNKNOWN").all())

    def test_survivor_exit_blanks_remain_raw_null_and_are_evaluable(self):
        rows = self.subjects.loc[
            self.subjects["cause_of_death_raw"].eq("0")
            & self.subjects["exit_status_raw"].isna()
        ]
        self.assertEqual(len(rows), 695)
        self.assertTrue(rows["Exit of the study"].isna().all())
        self.assertTrue(rows["exit_status_raw"].isna().all())
        self.assertTrue(rows["event_source_valid"].all())
        for horizon in (90, 180, 365, 730):
            with self.subTest(horizon=horizon):
                endpoint = build_endpoint(self.subjects, horizon).set_index("patient_id").loc[rows["patient_id"]]
                self.assertFalse(endpoint["endpoint_state"].eq("UNKNOWN").any())
                self.assertTrue(endpoint["endpoint_state"].eq("NEGATIVE").all())

    def test_short_followup_without_scd_is_never_negative(self):
        source = self.subjects.set_index("patient_id")
        for horizon in (90, 180, 365, 730):
            with self.subTest(horizon=horizon):
                endpoint = build_endpoint(self.subjects, horizon).set_index("patient_id")
                short_no_scd = source["followup_days"].lt(horizon) & source["cause_of_death_raw"].ne("3")
                states = set(endpoint.loc[short_no_scd, "endpoint_state"])
                self.assertTrue(states.issubset({"CENSORED", "COMPETING_EVENT"}))
                self.assertNotIn("NEGATIVE", states)


if __name__ == "__main__":
    unittest.main()
