import unittest

import pandas as pd

from src.endpoints import build_endpoint


def fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["positive", "negative", "censored", "competing", "unknown", "late_scd", "late_competing"],
            "followup_days": [50.0, 800.0, 50.0, 50.0, 50.0, 800.0, 800.0],
            "cause_of_death_raw": ["3", "0", "0", "1", "3", "3", "6"],
            "event_source_valid": [True, True, True, True, False, True, True],
        }
    )


class EndpointTests(unittest.TestCase):
    def test_all_states_at_all_supported_horizons(self):
        for horizon in (90, 180, 365, 730):
            with self.subTest(horizon=horizon):
                result = build_endpoint(fixture(), horizon).set_index("patient_id")
                self.assertEqual(result.at["positive", "endpoint_state"], "POSITIVE")
                self.assertEqual(result.at["negative", "endpoint_state"], "NEGATIVE")
                self.assertEqual(result.at["censored", "endpoint_state"], "CENSORED")
                self.assertEqual(result.at["competing", "endpoint_state"], "COMPETING_EVENT")
                self.assertEqual(result.at["unknown", "endpoint_state"], "UNKNOWN")
                self.assertEqual(result.at["late_scd", "endpoint_state"], "NEGATIVE")
                self.assertEqual(result.at["late_competing", "endpoint_state"], "NEGATIVE")

    def test_short_followup_without_scd_is_not_negative(self):
        result = build_endpoint(fixture(), 365).set_index("patient_id")
        self.assertEqual(result.at["censored", "endpoint_state"], "CENSORED")
        self.assertTrue(pd.isna(result.at["censored", "binary_label_if_evaluable"]))

    def test_binary_labels_only_for_evaluable_states(self):
        result = build_endpoint(fixture(), 365).set_index("patient_id")
        self.assertEqual(result.at["positive", "binary_label_if_evaluable"], 1)
        self.assertEqual(result.at["negative", "binary_label_if_evaluable"], 0)
        self.assertTrue(pd.isna(result.at["competing", "binary_label_if_evaluable"]))
        self.assertTrue(pd.isna(result.at["unknown", "binary_label_if_evaluable"]))

    def test_input_is_not_mutated(self):
        source = fixture()
        expected = source.copy(deep=True)
        build_endpoint(source, 90)
        pd.testing.assert_frame_equal(source, expected)

    def test_invalid_horizon_and_missing_columns(self):
        with self.assertRaises(ValueError):
            build_endpoint(fixture(), 100)
        with self.assertRaises(KeyError):
            build_endpoint(fixture().drop(columns="event_source_valid"), 90)


if __name__ == "__main__":
    unittest.main()
