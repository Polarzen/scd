import unittest

from scripts.verify_all import verify_data_contract


class DataContractTests(unittest.TestCase):
    def test_contract_matches_artifacts(self):
        self.assertEqual(verify_data_contract(), [])


if __name__ == "__main__":
    unittest.main()
