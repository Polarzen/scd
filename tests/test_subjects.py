import unittest

from scripts.verify_all import verify_subjects


class SubjectIntegrityTests(unittest.TestCase):
    def test_complete_subject_state(self):
        self.assertEqual(verify_subjects(), [])


if __name__ == "__main__":
    unittest.main()
