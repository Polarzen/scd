import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_music.py"
SPEC = importlib.util.spec_from_file_location("audit_music", SCRIPT)
audit_music = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit_music)


class AuditUnitTests(unittest.TestCase):
    def test_legacy_window_rule(self):
        self.assertEqual(audit_music.legacy_windows(179.999), 0)
        self.assertEqual(audit_music.legacy_windows(180), 1)
        self.assertEqual(audit_music.legacy_windows(3780), 2)
        self.assertEqual(audit_music.legacy_windows(10**9), 24)

    def test_full_window_rule(self):
        self.assertEqual(audit_music.full_windows(299.999), 0)
        self.assertEqual(audit_music.full_windows(300), 1)
        self.assertEqual(audit_music.full_windows(899.999), 2)

    def test_header_parser_does_not_need_signal_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            header = Path(tmp) / "P0001.hea"
            header.write_text(
                "P0001 3 200 60000\n"
                "P0001.dat 16+0 100 16 0 X\n"
                "P0001.dat 16+0 100 16 0 Y\n"
                "P0001.dat 16+0 100 16 0 Z\n",
                encoding="utf-8",
            )
            parsed = audit_music.parse_header(header, "HOLTER")
            self.assertEqual(parsed["patient_id"], "P0001")
            self.assertEqual(parsed["duration_sec"], 300)
            self.assertEqual(parsed["lead_names"], ["X", "Y", "Z"])
            self.assertEqual(parsed["expected_dat_sizes"], {"P0001.dat": 360000})


class GeneratedReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        report_path = Path(__file__).resolve().parents[1] / "reports" / "DATA_AUDIT.json"
        if not report_path.exists():
            raise unittest.SkipTest("Run scripts/audit_music.py first")
        cls.report = json.loads(report_path.read_text(encoding="utf-8"))

    def test_phase_and_safety_scope(self):
        self.assertEqual(self.report["audit"]["phase"], "PHASE 1 - DATA AUDIT")
        self.assertFalse(self.report["audit"]["waveform_content_read"])
        self.assertFalse(self.report["audit"]["feature_extraction_performed"])
        self.assertEqual(self.report["integrity"]["dat_content_hashes_checked"], 0)
        self.assertTrue(self.report["hard_gate"]["passed"])
        self.assertFalse(self.report["hard_gate"]["phase_2_authorized"])

    def test_patient_identity(self):
        cohort = self.report["cohort"]
        self.assertEqual(cohort["official_subject_rows"], cohort["unique_patient_ids"])
        self.assertEqual(cohort["duplicate_patient_ids"], [])
        self.assertEqual(cohort["blank_patient_id_rows"], 0)

    def test_header_referential_integrity(self):
        records = self.report["records"]
        self.assertEqual(records["unmatched_header_patient_ids"], [])
        self.assertEqual(records["records_without_header"], [])
        self.assertEqual(records["headers_not_in_records"], [])

    def test_small_source_hashes(self):
        integrity = self.report["integrity"]
        self.assertGreater(integrity["small_file_and_header_hashes_checked"], 0)
        self.assertEqual(integrity["small_file_and_header_hash_mismatches"], [])

    def test_endpoint_horizons_are_monotone(self):
        horizons = self.report["outcomes"]["scd_by_horizon_days"]
        values = [horizons[str(day)] for day in (90, 180, 365, 730)]
        self.assertEqual(values, sorted(values))
        self.assertLessEqual(values[-1], self.report["outcomes"]["scd_total"])


if __name__ == "__main__":
    unittest.main()
