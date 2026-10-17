"""Train Logistic Regression and save a reusable model artifact."""
from pathlib import Path

import joblib

from model import DEFAULT_DATA, train_model as fit_model

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model.pkl"


def train_and_save_model(data_path: Path = DEFAULT_DATA, model_path: Path = MODEL_PATH):
    """Fit the practical model, evaluate it, and save it for app.py."""
    result = fit_model(data_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, model_path)
    return result


def load_model_artifact(model_path: Path = MODEL_PATH):
    """Load and validate the model artifact saved by this script."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}. Run train_model.py first.")
    artifact = joblib.load(model_path)
    # Also accept the dictionary artifact created by the preceding project version.
    if isinstance(artifact, dict) and "model" in artifact:
        artifact = artifact["model"]
    if not callable(getattr(artifact, "predict", None)):
        raise ValueError("The model artifact has an unexpected format. Run train_model.py again.")
    return artifact


if __name__ == "__main__":
    trained = train_and_save_model()
    print(f"Trained on {trained.train_size} rows; tested on {trained.test_size} rows.")
    print(f"Holdout accuracy: {trained.accuracy:.3f}")
    print(f"Confusion matrix [true 0/1 by predicted 0/1]: {trained.confusion}")
    print(f"Saved fitted pipeline with Joblib: {MODEL_PATH}")
