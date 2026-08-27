import unittest

from scripts.verify_all import verify_provenance


class ReferentialIntegrityTests(unittest.TestCase):
    def test_subject_record_provenance_mapping(self):
        self.assertEqual(verify_provenance(), [])


if __name__ == "__main__":
    unittest.main()
