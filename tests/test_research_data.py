import unittest
from src.research_data import load_split


class DataTests(unittest.TestCase):
    def test_circo_train_rejected(self):
        with self.assertRaises(ValueError):
            load_split("data", "circo", "train")

    def test_invalid_category_rejected(self):
        with self.assertRaises(ValueError):
            load_split("data", "fashioniq", category="invalid")


if __name__ == "__main__":
    unittest.main()
