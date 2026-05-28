"""Train and save the Student Exam Score Predictor model.

The dataset is synthetic so beginners can run this project without downloading
anything. We create realistic-looking student records, train Linear Regression,
save the CSV dataset, and store the trained model with pickle.
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"
DATASET_PATH = DATASET_DIR / "student_data.csv"
MODEL_PATH = MODEL_DIR / "student_model.pkl"


def generate_dataset(rows=500, random_seed=42):
    """Create a synthetic dataset with student learning habits and exam marks."""
    rng = np.random.default_rng(random_seed)

    study_hours = rng.uniform(0.5, 10, rows)
    attendance = rng.uniform(45, 100, rows)
    sleep_hours = rng.uniform(4, 10, rows)
    previous_marks = rng.uniform(30, 95, rows)

    noise = rng.normal(0, 5, rows)

    exam_score = (
        4.2 * study_hours
        + 0.25 * attendance
        + 2.0 * sleep_hours
        + 0.45 * previous_marks
        - 25
        + noise
    )

    # Clip keeps the target inside the normal exam score range.
    exam_score = np.clip(exam_score, 0, 100)

    return pd.DataFrame(
        {
            "study_hours": np.round(study_hours, 2),
            "attendance": np.round(attendance, 2),
            "sleep_hours": np.round(sleep_hours, 2),
            "previous_marks": np.round(previous_marks, 2),
            "exam_score": np.round(exam_score, 2),
        }
    )


def train_model():
    """Train Linear Regression and save both the dataset and model."""
    DATASET_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    data = generate_dataset()
    data.to_csv(DATASET_PATH, index=False)

    features = data[["study_hours", "attendance", "sleep_hours", "previous_marks"]]
    target = data["exam_score"]

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    with MODEL_PATH.open("wb") as file:
        pickle.dump(model, file)

    print("Training completed successfully!")
    print(f"Dataset saved to: {DATASET_PATH}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")


if __name__ == "__main__":
    train_model()
