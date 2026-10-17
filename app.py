"""Flask interface for the Logistic Regression diabetes practical."""
import os
import json
import secrets
import sqlite3
from contextlib import closing
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from model import FEATURES, predict_class, train_model
from train_model import MODEL_PATH, load_model_artifact

PROJECT_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(PROJECT_DIR / "templates"), static_folder=str(PROJECT_DIR / "static"))
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config["PREDICTION_STORE"] = PROJECT_DIR / "instance" / "prediction_results.sqlite3"

# Training is a separate reproducible step. The fallback keeps first-run use
# possible, while the documented workflow trains and saves model.pkl first.
MODEL = load_model_artifact() if MODEL_PATH.is_file() else train_model().model

FEATURE_LIMITS = {
    "Pregnancies": (0, 20),
    "Glucose": (1, 300),
    "BloodPressure": (1, 200),
    "SkinThickness": (0, 120),
    "Insulin": (0, 1000),
    "BMI": (1, 80),
    "DiabetesPedigreeFunction": (0, 3),
    "Age": (21, 120),
}


def store_prediction_once(token, result, submitted):
    """Keep a one-time result on the server; the cookie contains only its token."""
    store_path = Path(app.config["PREDICTION_STORE"])
    store_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(store_path)) as connection:
        with connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS prediction_results "
                "(token TEXT PRIMARY KEY, result TEXT NOT NULL, submitted TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO prediction_results (token, result, submitted) VALUES (?, ?, ?)",
                (token, result, json.dumps(submitted)),
            )


def take_prediction_once(token):
    """Fetch and delete one result atomically so it cannot be replayed."""
    store_path = Path(app.config["PREDICTION_STORE"])
    if not store_path.exists():
        return None
    with closing(sqlite3.connect(store_path)) as connection:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT result, submitted FROM prediction_results WHERE token = ?", (token,)
            ).fetchone()
            if row is not None:
                connection.execute("DELETE FROM prediction_results WHERE token = ?", (token,))
    return {"result": row[0], "submitted": json.loads(row[1])} if row else None


def render_form(result=None, error=None, status=200, submitted=None):
    submitted = submitted or {}
    fields = [
        {
            "name": name,
            "minimum": FEATURE_LIMITS[name][0],
            "maximum": FEATURE_LIMITS[name][1],
            "value": submitted.get(name, ""),
        }
        for name in FEATURES
    ]
    response = app.make_response(
        (render_template("index.html", fields=fields, result=result, error=error), status)
    )
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


@app.get("/")
def index():
    """Show the prediction form."""
    return render_form()


@app.get("/health")
def health():
    """Provide a simple health check for local and container use."""
    return jsonify(status="ok")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    """Validate measurements; successful POSTs redirect to a one-time result."""
    if request.method == "GET":
        return render_form()

    submitted = {name: request.form.get(name, "") for name in FEATURES}
    values = []
    for name in FEATURES:
        raw_value = submitted[name].strip()
        if not raw_value:
            return render_form(error=f"{name} is required.", status=400, submitted=submitted)
        try:
            value = float(raw_value)
        except ValueError:
            return render_form(error=f"{name} must be a number.", status=400, submitted=submitted)
        minimum, maximum = FEATURE_LIMITS[name]
        if not (minimum <= value <= maximum):
            return render_form(
                error=f"{name} must be between {minimum} and {maximum}.",
                status=400,
                submitted=submitted,
            )
        values.append(value)

    try:
        predicted = predict_class(MODEL, values)
        probabilities = MODEL.predict_proba([values])[0]
    except Exception:
        app.logger.exception("Prediction failed")
        return render_form(
            error="The model could not generate a prediction. Please check your measurements and try again.",
            status=500,
            submitted=submitted,
        )
    label = "positive" if predicted == 1 else "negative"
    probability_text = " ".join(
        f"Class {class_id}: {probability:.1%}"
        for class_id, probability in zip(MODEL.classes_, probabilities)
    )
    result_text = f"Predicted class: {predicted} ({label}). Probabilities: {probability_text}."
    result_token = secrets.token_urlsafe(32)
    store_prediction_once(result_token, result_text, submitted)
    session["prediction_result_token"] = result_token
    return redirect(url_for("prediction_result"), code=303)


@app.get("/prediction/result")
def prediction_result():
    """Display and consume the successful POST result once."""
    result_token = session.pop("prediction_result_token", None)
    stored = take_prediction_once(result_token) if result_token else None
    if stored is None:
        return redirect(url_for("index"), code=303)
    return render_form(result=stored["result"], submitted=stored["submitted"])


if __name__ == "__main__":
    print(f"Using fitted model artifact: {MODEL_PATH}")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
