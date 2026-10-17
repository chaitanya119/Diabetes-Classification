"""Flask route and input-validation tests for the practical."""
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import app
from model import FEATURES


class DiabetesAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config.update(TESTING=True)
        cls.temp_dir = tempfile.TemporaryDirectory()
        app.config["PREDICTION_STORE"] = Path(cls.temp_dir.name) / "results.sqlite3"
        cls.client = app.test_client()
        cls.valid_form = {
            "Pregnancies": "2",
            "Glucose": "120",
            "BloodPressure": "70",
            "SkinThickness": "25",
            "Insulin": "80",
            "BMI": "30",
            "DiabetesPedigreeFunction": "0.5",
            "Age": "33",
        }

    def test_home_route_renders_form(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Diabetes Classification", response.data)
        self.assertIn(b"/static/style.css", response.data)

    def test_initial_form_is_blank_and_has_no_prediction(self):
        with patch("app.predict_class") as predict:
            response = self.client.get("/")
        page = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("Predicted class:", page)
        self.assertNotIn("Probabilities:", page)
        self.assertNotIn('class="notice success"', page)
        self.assertFalse(predict.called)
        for feature in FEATURES:
            input_match = re.search(rf'<input[^>]*name="{feature}"[^>]*>', page)
            self.assertIsNotNone(input_match, f"Missing input for {feature}")
            self.assertNotRegex(input_match.group(0), r'\bvalue\s*=')
            self.assertIn("required", input_match.group(0))

    def test_health_route_returns_ok(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok"})

    def test_prediction_get_route_renders_form(self):
        response = self.client.get("/prediction")
        self.assertEqual(response.status_code, 200)

    def test_prediction_route_returns_predicted_class(self):
        response = self.client.post("/prediction", data=self.valid_form)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["Location"], "/prediction/result")
        result = self.client.get(response.headers["Location"])
        self.assertEqual(result.status_code, 200)
        self.assertRegex(result.get_data(as_text=True), r"Predicted class: [01]")
        self.assertIn(b"Probabilities:", result.data)

    def test_fresh_home_after_prediction_is_blank_and_result_is_one_time(self):
        post = self.client.post("/prediction", data=self.valid_form)
        self.assertEqual(post.status_code, 303)
        result = self.client.get(post.headers["Location"])
        self.assertIn(b"Predicted class:", result.data)

        home = self.client.get("/")
        page = home.get_data(as_text=True)
        self.assertEqual(home.status_code, 200)
        self.assertNotIn("Predicted class:", page)
        self.assertNotIn("Probabilities:", page)
        for feature in FEATURES:
            input_match = re.search(rf'<input[^>]*name="{feature}"[^>]*>', page)
            self.assertIsNotNone(input_match, f"Missing input for {feature}")
            self.assertNotRegex(input_match.group(0), r'\bvalue\s*=')

        replay = self.client.get("/prediction/result", follow_redirects=True)
        self.assertEqual(replay.request.path, "/")
        self.assertNotIn(b"Predicted class:", replay.data)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_prediction_rejects_missing_measurement(self):
        values = dict(self.valid_form)
        del values[FEATURES[0]]
        response = self.client.post("/prediction", data=values)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"is required", response.data)

    def test_prediction_rejects_non_numeric_measurement(self):
        values = dict(self.valid_form, Glucose="high")
        response = self.client.post("/prediction", data=values)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"must be a number", response.data)

    def test_prediction_rejects_out_of_range_measurement(self):
        values = dict(self.valid_form, Age="12")
        response = self.client.post("/prediction", data=values)
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"must be between", response.data)

    def test_static_stylesheet_is_served(self):
        response = self.client.get("/static/style.css")
        try:
            self.assertEqual(response.status_code, 200)
            self.assertIn(b".grid", response.data)
        finally:
            response.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
