import unittest

from scripts.verify_all import verify_hashes, verify_source_exact


class SourceIntegrityTests(unittest.TestCase):
    def test_compact_hashes(self):
        self.assertEqual(verify_hashes(), [])

    def test_source_exact(self):
        self.assertEqual(verify_source_exact(), [])


if __name__ == "__main__":
    unittest.main()
