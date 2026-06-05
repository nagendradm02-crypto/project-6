# test_preprocessing.py

import unittest
import pandas as pd
import numpy as np

# Import your preprocessing function
from data_preprocessing import preprocess_data


class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        """
        Create sample dataset for testing.
        """
        self.sample_data = pd.DataFrame({
            "Vehicle_Count": [100, 200, np.nan, 150],
            "Speed": [60, 45, 55, np.nan],
            "Traffic_Density": [30, 50, 40, 35]
        })

    def test_missing_values_removed(self):
        """
        Check if missing values are handled.
        """
        processed_df = preprocess_data(self.sample_data)

        self.assertEqual(
            processed_df.isnull().sum().sum(),
            0
        )

    def test_dataframe_not_empty(self):
        """
        Check if processed dataframe is not empty.
        """
        processed_df = preprocess_data(self.sample_data)

        self.assertFalse(processed_df.empty)

    def test_columns_exist(self):
        """
        Verify important columns remain.
        """
        processed_df = preprocess_data(self.sample_data)

        self.assertIn("Vehicle_Count", processed_df.columns)
        self.assertIn("Speed", processed_df.columns)
        self.assertIn("Traffic_Density", processed_df.columns)

    def test_output_type(self):
        """
        Verify output is DataFrame.
        """
        processed_df = preprocess_data(self.sample_data)

        self.assertIsInstance(processed_df, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
