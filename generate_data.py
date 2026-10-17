"""Create a reproducible, synthetic dataset for the classification practical."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "diabetes_synthetic.csv"
FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

def generate_dataset(rows: int = 600, seed: int = 42) -> None:
    """Save simulated measurements; they do not represent real people."""
    rng = np.random.default_rng(seed)
    pregnancies = rng.poisson(2.5, rows).clip(0, 13)
    age = rng.normal(34, 11, rows).clip(18, 80)
    glucose = rng.normal(112, 27, rows).clip(55, 200)
    blood_pressure = rng.normal(72, 12, rows).clip(40, 120)
    skin_thickness = rng.normal(25, 10, rows).clip(5, 60)
    insulin = rng.lognormal(4.4, 0.75, rows).clip(10, 400)
    bmi = rng.normal(31, 7, rows).clip(16, 55)
    pedigree = rng.lognormal(-1.1, 0.65, rows).clip(0.05, 2.5)
    score = -5.4 + 0.026 * glucose + 0.045 * bmi + 0.025 * age + 0.13 * pregnancies + 0.65 * pedigree
    probability = 1 / (1 + np.exp(-score))
    outcome = rng.binomial(1, probability)
    values = np.column_stack([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age, outcome])
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(DATA_PATH, values, delimiter=",", header=",".join([*FEATURES, "Outcome"]), comments="", fmt="%.4f")
    print(f"Created {len(values)} synthetic rows at {DATA_PATH}")

if __name__ == "__main__":
    generate_dataset()
