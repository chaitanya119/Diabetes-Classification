"""Load data, train Logistic Regression, and predict the diabetes class."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_DATA = DATA_DIR / "pima_indians_diabetes.csv"
SYNTHETIC_DATA = DATA_DIR / "diabetes_synthetic.csv"

@dataclass
class TrainingResult:
    model: Pipeline
    accuracy: float
    confusion: list[list[int]]
    train_size: int
    test_size: int

def load_dataset(path: Path = DEFAULT_DATA) -> tuple[np.ndarray, np.ndarray]:
    """Read the CSV and check its columns and labels before training."""
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    expected = [*FEATURES, "Outcome"]
    if table.dtype.names != tuple(expected):
        raise ValueError(f"CSV columns must be: {', '.join(expected)}")
    values = np.column_stack([table[name] for name in expected])
    if values.ndim != 2 or values.shape[0] < 20 or np.isinf(values).any():
        raise ValueError("CSV must contain at least 20 numeric records without infinite values.")
    labels = values[:, -1].astype(int)
    if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError("Outcome must contain both class labels: 0 and 1.")
    features = values[:, :-1].copy()
    # In the Pima source, zero stands for an unrecorded value in these tests.
    # Keep it as missing so the training pipeline can impute it from train data.
    for column in range(1, 6):
        features[features[:, column] == 0, column] = np.nan
    if np.isnan(features).all(axis=0).any():
        raise ValueError("Each measurement column must contain at least one recorded value.")
    return features, labels

def train_model(path: Path = DEFAULT_DATA) -> TrainingResult:
    """Split, scale and train a binary Logistic Regression model."""
    features, labels = load_dataset(path)
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=42, stratify=labels)
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    return TrainingResult(model, float(accuracy_score(y_test, predicted)), confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(), len(y_train), len(y_test))

def predict_class(model: Pipeline, measurements: list[float]) -> int:
    """Return class 0 or 1 for one set of eight measurements."""
    if len(measurements) != len(FEATURES):
        raise ValueError(f"Enter exactly {len(FEATURES)} measurements.")
    row = np.asarray(measurements, dtype=float)
    if not np.isfinite(row).all():
        raise ValueError("Measurements must be finite numbers.")
    for column in range(1, 6):
        if row[column] == 0:
            row[column] = np.nan
    return int(model.predict(row.reshape(1, -1))[0])
