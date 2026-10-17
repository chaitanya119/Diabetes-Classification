"""Tests for the real diabetes data, training, and predicted-class helper."""
import numpy as np
import unittest
from sklearn.pipeline import Pipeline

from model import DEFAULT_DATA, FEATURES, load_dataset, predict_class, train_model
from train_model import MODEL_PATH, load_model_artifact


class DiabetesClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = train_model()

    def test_real_dataset_has_expected_schema_and_rows(self):
        features, labels = load_dataset(DEFAULT_DATA)
        self.assertEqual(features.shape, (768, len(FEATURES)))
        self.assertEqual(len(labels), 768)
        self.assertEqual(set(labels.tolist()), {0, 1})
        self.assertTrue(np.isnan(features[:, 1:6]).any())

    def test_training_returns_fitted_logistic_regression_pipeline(self):
        self.assertIsInstance(self.result.model, Pipeline)
        self.assertIn("classifier", self.result.model.named_steps)
        self.assertEqual(self.result.train_size + self.result.test_size, 768)
        self.assertGreaterEqual(self.result.accuracy, 0.0)
        self.assertLessEqual(self.result.accuracy, 1.0)
        self.assertEqual(sum(map(sum, self.result.confusion)), self.result.test_size)

    def test_joblib_artifact_is_the_fitted_pipeline(self):
        saved_model = load_model_artifact(MODEL_PATH)
        self.assertIsInstance(saved_model, Pipeline)
        self.assertEqual(saved_model.named_steps["classifier"].classes_.tolist(), [0, 1])

    def test_predict_class_returns_binary_label(self):
        values = [2, 120, 70, 25, 80, 30, 0.5, 33]
        self.assertIn(predict_class(self.result.model, values), (0, 1))

    def test_prediction_rejects_wrong_number_of_values(self):
        with self.assertRaises(ValueError):
            predict_class(self.result.model, [100, 30])

    def test_prediction_rejects_infinite_values(self):
        values = [2, 120, 70, 25, 80, 30, 0.5, float("inf")]
        with self.assertRaises(ValueError):
            predict_class(self.result.model, values)


if __name__ == "__main__":
    unittest.main(verbosity=2)
