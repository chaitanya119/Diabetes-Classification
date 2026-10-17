# Diabetes Classification with Logistic Regression

This practical trains a Logistic Regression classifier and returns its predicted
class (`0` or `1`) through a small Flask web application.

## Practical steps in simple English

1. Install the Python packages listed in `requirements.txt` inside `.venv`.
2. Read the documented Pima Indians dataset from `data/pima_indians_diabetes.csv`.
   The separate `data/diabetes_synthetic.csv` file is simulated teaching data.
3. Separate the eight input measurements from the `Outcome` label.
4. Split records into training and test sets. The pipeline fills zero-coded
   missing measurements with training-set medians, scales values, and trains
   Logistic Regression.
5. Check test accuracy and the confusion matrix. Save the fitted pipeline as
   `model.pkl` with Joblib.
6. Run the Flask app. Submit measurements on `/prediction` to see the predicted
   class. `/health` reports whether the app is responding.

## Verified training result

The latest verified local run used a stratified 80/20 split with random state
42: 614 training rows and 154 test rows. Holdout accuracy was `0.708`; the
confusion matrix (actual classes by predicted classes) was `[[82, 18], [27, 27]]`.
These are measured results for this fixed dataset and split, not a clinical
performance claim.

Dataset source, provenance, zero-value handling, and limits are described in
[`data/README.md`](data/README.md). The real dataset is used by default. The
synthetic data generator is optional and does not overwrite the real data file.
This historical dataset represents a narrow cohort and model output is not
medical advice or a diagnosis.

The requested `2CEAI702_PRACTICAL_FILE__FORMAT.docx` was not present in the
workspace. The available `ml ops theory/2ceai702 ml ops.pdf` is a course syllabus;
its practical section only says that practicals follow the syllabus and does
not provide a Practical 1 writeup format. This README records the objective,
data, method, measured result, commands, and screenshot checklist for this
implementation without claiming to reproduce unavailable instructions.

## PowerShell setup and run

Run these PowerShell commands from the workspace:

```powershell
Set-Location -LiteralPath "C:\Users\chait\OneDrive\Desktop\7 TH\ml ops project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_model.py
python app.py
```

Open `http://127.0.0.1:5000`. The app also listens on port 5000 inside its
container. Stop the local server with `Ctrl+C`.

## Tests

With `.venv` activated, run the pytest suite:

```powershell
pytest -q
```

The tests cover the home, health, and prediction routes, stylesheet delivery,
predicted class output, one-time Post/Redirect/Get results, blank fresh page
loads, and missing, non-numeric, and out-of-range input.

## Docker

Build and run the container locally:

```powershell
docker build -t diabetes-classifier .
docker run --rm -p 5000:5000 diabetes-classifier
```

The image installs the requirements, trains the model from the included real
CSV, and starts Flask. No cloud account or deployment is required.

## Screenshot checklist

Capture these screenshots yourself after running each step; the repository does
not include fabricated screenshots.

1. VS Code Explorer showing the source, `data`, `templates`, `static`, and
   `tests` folders, plus `Dockerfile` and `.dockerignore`.
2. `python train_model.py` output showing actual training/test row counts,
   accuracy, confusion matrix, and the saved `model.pkl` path.
3. Browser at `/` showing the prediction form.
4. Browser after a valid submission showing the predicted class.
5. Browser or terminal showing `/health` returning `{"status":"ok"}`.
6. `pytest -q` output with the real passing test count.
7. `docker build -t diabetes-classifier .` output after a successful build.
8. Optional: the container running with `docker run --rm -p 5000:5000
   diabetes-classifier` and `/health` responding.
