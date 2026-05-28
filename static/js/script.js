// Frontend behavior for validating the form and calling the Flask API.

const form = document.getElementById("predictionForm");
const button = form.querySelector(".predict-button");
const message = document.getElementById("message");
const resultCard = document.getElementById("resultCard");
const predictionValue = document.getElementById("predictionValue");
const scoreFill = document.getElementById("scoreFill");

const fieldRules = {
    study_hours: { label: "Study hours", min: 0, max: 16 },
    attendance: { label: "Attendance", min: 0, max: 100 },
    sleep_hours: { label: "Sleep hours", min: 0, max: 12 },
    previous_marks: { label: "Previous marks", min: 0, max: 100 },
};

function showMessage(text, type) {
    message.textContent = text;
    message.className = `message ${type}`;
}

function setLoading(isLoading) {
    button.disabled = isLoading;
    button.classList.toggle("loading", isLoading);
}

function getFormData() {
    return {
        study_hours: Number(document.getElementById("study_hours").value),
        attendance: Number(document.getElementById("attendance").value),
        sleep_hours: Number(document.getElementById("sleep_hours").value),
        previous_marks: Number(document.getElementById("previous_marks").value),
    };
}

function validateForm(data) {
    for (const [field, rule] of Object.entries(fieldRules)) {
        const value = data[field];

        if (Number.isNaN(value)) {
            return `${rule.label} is required.`;
        }

        if (value < rule.min || value > rule.max) {
            return `${rule.label} must be between ${rule.min} and ${rule.max}.`;
        }
    }

    return "";
}

function showPrediction(score) {
    predictionValue.textContent = score.toFixed(2);
    scoreFill.style.width = `${Math.min(score, 100)}%`;
    resultCard.classList.remove("hidden");
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const data = getFormData();
    const validationError = validateForm(data);

    resultCard.classList.add("hidden");
    scoreFill.style.width = "0%";

    if (validationError) {
        showMessage(validationError, "error");
        return;
    }

    setLoading(true);
    showMessage("Predicting score...", "success");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(data),
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Prediction failed.");
        }

        showMessage("Prediction completed successfully.", "success");
        showPrediction(result.prediction);
    } catch (error) {
        showMessage(error.message || "Unable to connect to the prediction API.", "error");
    } finally {
        setLoading(false);
    }
});
