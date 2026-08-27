import unittest

from scripts.verify_all import verify_records


class RecordIntegrityTests(unittest.TestCase):
    def test_record_contract(self):
        self.assertEqual(verify_records(), [])


if __name__ == "__main__":
    unittest.main()
