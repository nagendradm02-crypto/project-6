# test_model.py

import unittest
import joblib
import numpy as np


class TestTrafficModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Load model and scaler once before all tests.
        """
        cls.model = joblib.load("trained_model.pkl")
        cls.scaler = joblib.load("scaler.pkl")

    def test_model_loaded(self):
        """
        Check model is loaded successfully.
        """
        self.assertIsNotNone(self.model)

    def test_scaler_loaded(self):
        """
        Check scaler is loaded successfully.
        """
        self.assertIsNotNone(self.scaler)

    def test_prediction_output(self):
        """
        Check model produces predictions.
        """
        sample_data = np.array([
            [45, 2, 1, 0, 30]
        ])

        sample_scaled = self.scaler.transform(sample_data)
        prediction = self.model.predict(sample_scaled)

        self.assertEqual(len(prediction), 1)

    def test_prediction_numeric(self):
        """
        Check prediction is numeric.
        """
        sample_data = np.array([
            [45, 2, 1, 0, 30]
        ])

        sample_scaled = self.scaler.transform(sample_data)
        prediction = self.model.predict(sample_scaled)

        self.assertIsInstance(
            prediction[0],
            (int, float, np.integer, np.floating)
        )

    def test_prediction_not_nan(self):
        """
        Check prediction is not NaN.
        """
        sample_data = np.array([
            [45, 2, 1, 0, 30]
        ])

        sample_scaled = self.scaler.transform(sample_data)
        prediction = self.model.predict(sample_scaled)

        self.assertFalse(np.isnan(prediction[0]))
