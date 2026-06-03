"""Flask backend for the Student Exam Score Predictor app.

This file serves the home page and exposes a JSON API endpoint at /predict.
The frontend sends student details to /predict, and Flask returns a predicted
exam score from the trained Linear Regression model.
"""

from pathlib import Path
import pickle
import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "student_model.pkl"

app = Flask(__name__)


def load_model():
    """Load the trained model from disk once when the app starts."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run `python train_model.py` before starting Flask."
        )

    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


model = load_model()


@app.route("/")
def home():
    """Render the main prediction page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Receive form data as JSON and return the predicted exam score."""
    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({"error": "No input data received."}), 400

        required_fields = [
            "study_hours",
            "attendance",
            "sleep_hours",
            "previous_marks",
        ]

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

        study_hours = float(data["study_hours"])
        attendance = float(data["attendance"])
        sleep_hours = float(data["sleep_hours"])
        previous_marks = float(data["previous_marks"])

        validation_error = validate_input(
            study_hours, attendance, sleep_hours, previous_marks
        )
        if validation_error:
            return jsonify({"error": validation_error}), 400

        features = pd.DataFrame(
            [
                {
                    "study_hours": study_hours,
                    "attendance": attendance,
                    "sleep_hours": sleep_hours,
                    "previous_marks": previous_marks,
                }
            ]
        )
        prediction = float(model.predict(features)[0])

        # Exam marks should stay in a realistic 0-100 range.
        prediction = max(0, min(100, prediction))

        return jsonify({"prediction": round(prediction, 2)})

    except ValueError:
        return jsonify({"error": "Please enter valid numeric values."}), 400
    except Exception:
        return jsonify({"error": "Something went wrong while predicting."}), 500


def validate_input(study_hours, attendance, sleep_hours, previous_marks):
    """Validate the input ranges used by the frontend and backend."""
    if not 0 <= study_hours <= 16:
        return "Study hours must be between 0 and 16."
    if not 0 <= attendance <= 100:
        return "Attendance must be between 0 and 100."
    if not 0 <= sleep_hours <= 12:
        return "Sleep hours must be between 0 and 12."
    if not 0 <= previous_marks <= 100:
        return "Previous marks must be between 0 and 100."
    return None


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
