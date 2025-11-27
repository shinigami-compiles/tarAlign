import os
import sys
import pandas as pd
import numpy as np
import json
import joblib
import logging
from datetime import datetime
from threading import Lock
from flask import Flask, request, render_template, jsonify, send_file
import io

# --- Make sure Python can see the src/ modules (feature_engineering, etc.) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
    print("DEBUG: Added SRC_DIR to sys.path ->", SRC_DIR)

# ------------------------------
# INITIAL SETUP
# ------------------------------
app = Flask(__name__, static_folder='static', template_folder='templates')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
history_lock = Lock()

# ------------------------------
# PROJECT PATHS
# ------------------------------

MODELS_DIR = os.path.join(BASE_DIR, "models")
HISTORY_DIR = os.path.join(BASE_DIR, "history")

MODEL_PATH = os.path.join(MODELS_DIR, "logisticregression_model.joblib")
print("DEBUG MODEL_PATH:", MODEL_PATH, "Exists:", os.path.exists(MODEL_PATH))
HISTORY_FILE = os.path.join(HISTORY_DIR, "records.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# ------------------------------
# 20 CORE FEATURES (HTML + ML)
# ------------------------------
CORE_20_FEATURES = [
    {
        'name': 'minutes_goal',
        'question': 'How many minutes did you actively work on your main goal today?',
        'default': 60,
        'placeholder': 'e.g., 90  (you studied/worked for 1.5 hours)'
    },
    {
        'name': 'consistency_index',
        'question': 'How consistent were you with your planned routine today? (0–1, 1 = fully consistent)',
        'default': 0.5,
        'placeholder': 'e.g., 0.7  (you followed most of your plan)'
    },
    {
        'name': 'sleep_hours',
        'question': 'How many hours did you sleep last night?',
        'default': 7.0,
        'placeholder': 'e.g., 7.5  (7 and a half hours)'
    },
    {
        'name': 'avg_minutes_last_week',
        'question': 'On average, how many minutes per day did you work on this goal over the last 7 days?',
        'default': 60,
        'placeholder': 'e.g., 75  (a bit more than 1 hour/day)'
    },
    {
        'name': 'avg_consistency_last_week',
        'question': 'Average consistency with your routine over the last 7 days (0–1).',
        'default': 0.6,
        'placeholder': 'e.g., 0.65'
    },
    {
        'name': 'avg_sleep_last_week',
        'question': 'Average sleep per night over the last 7 days (in hours).',
        'default': 7.0,
        'placeholder': 'e.g., 7.0'
    },
    {
        'name': 'avg_minutes_last_month',
        'question': 'On average, how many minutes per day did you work on this goal over the last 30 days?',
        'default': 60,
        'placeholder': 'e.g., 70'
    },
    {
        'name': 'momentum_last_month',
        'question': 'Overall progress momentum over the last 30 days (0–1, higher = better flow).',
        'default': 0.5,
        'placeholder': 'e.g., 0.6'
    },
    {
        'name': 'friction_last_month',
        'question': 'Overall friction/difficulty over the last 30 days (0–1, higher = more friction).',
        'default': 0.5,
        'placeholder': 'e.g., 0.4  (things felt relatively smooth)'
    },
    {
        'name': 'avg_sleep_last_month',
        'question': 'Average sleep per night over the last 30 days (in hours).',
        'default': 7.0,
        'placeholder': 'e.g., 6.8'
    },
    {
        'name': 'baseline_goal_minutes',
        'question': 'When you started (around 90 days ago), how many minutes per day did you usually work on this goal?',
        'default': 30,
        'placeholder': 'e.g., 45'
    },
    {
        'name': 'baseline_consistency',
        'question': 'When you started, how consistent were you with this goal? (0–1)',
        'default': 0.4,
        'placeholder': 'e.g., 0.5'
    },
    {
        'name': 'baseline_sleep_hours',
        'question': 'When you started, how many hours did you typically sleep per night?',
        'default': 6.5,
        'placeholder': 'e.g., 6.5'
    },
    {
        'name': 'task_switch_avg',
        'question': 'How often did you switch tasks or get distracted in the last 30 days? (0–1, higher = more switching)',
        'default': 0.3,
        'placeholder': 'e.g., 0.4'
    },
    {
        'name': 'exercise_avg',
        'question': 'On average, how many minutes per day did you exercise in the last 30 days?',
        'default': 30,
        'placeholder': 'e.g., 25'
    },
    {
        'name': 'day_of_week',
        'question': 'What day of the week is this entry for? (0 = Monday, …, 6 = Sunday)',
        'default': 0,
        'placeholder': 'e.g., 2  (Wednesday)'
    },
    {
        'name': 'is_weekend',
        'question': 'Is this entry for a weekend day? (0 = No, 1 = Yes)',
        'default': 0,
        'placeholder': 'e.g., 0  (it is a weekday)'
    },
    {
        'name': 'goal_weight_career',
        'question': 'Out of your long-term goals, how important is career/professional growth? (0–1)',
        'default': 0.4,
        'placeholder': 'e.g., 0.5'
    },
    {
        'name': 'goal_weight_fitness',
        'question': 'Out of your long-term goals, how important is fitness/health? (0–1)',
        'default': 0.3,
        'placeholder': 'e.g., 0.3'
    },
    {
        'name': 'goal_weight_learning',
        'question': 'Out of your long-term goals, how important is learning/skills? (0–1)',
        'default': 0.3,
        'placeholder': 'e.g., 0.2'
    },
]


FEATURE_LIST = [f['name'] for f in CORE_20_FEATURES]

# ------------------------------
# LOAD MODEL PIPELINE
# ------------------------------
def load_pipeline():
    try:
        logger.info(f"Trying to load model from: {MODEL_PATH}")
        pipeline = joblib.load(MODEL_PATH)
        logger.info("✅ Pipeline loaded successfully.")
        return pipeline
    except Exception as e:
        logger.error(f"❌ Failed loading model pipeline from {MODEL_PATH}: {e}", exc_info=True)
        return None

pipeline = load_pipeline()

# ------------------------------
# HISTORY HANDLING
# ------------------------------
def ensure_history_file():
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'w') as f:
            json.dump([], f)

def save_record(user_id, score, pred_class, method):
    ensure_history_file()
    with history_lock:
        with open(HISTORY_FILE, 'r') as f:
            records = json.load(f)

        records.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": user_id,
            "alignment_score": float(score),
            "aligned_label": int(pred_class),
            "alignment_class": "Aligned" if pred_class == 1 else "Misaligned",
            "input_method": method,
        })

        with open(HISTORY_FILE, 'w') as f:
            json.dump(records, f, indent=2)

def load_history():
    ensure_history_file()
    with open(HISTORY_FILE, 'r') as f:
        return json.load(f)[::-1]

# ------------------------------
# ROUTES
# ------------------------------

@app.route("/")
def index():
    return render_template("index.html", features=CORE_20_FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    if pipeline is None:
        return render_template("index.html", features=CORE_20_FEATURES,
                               error="Model is not loaded. Train model first!")

    input_type = request.form.get("input_type", "")
    user_id = "Anonymous"

    try:
        # --------------------------
        # CSV UPLOAD INPUT
        # --------------------------
        if input_type == "csv":
            file = request.files.get("csv_file")
            if not file:
                return render_template("index.html", features=CORE_20_FEATURES, error="No CSV uploaded.")

            df = pd.read_csv(file.stream)

            missing = set(FEATURE_LIST) - set(df.columns)
            if missing:
                return render_template(
                    "index.html",
                    features=CORE_20_FEATURES,
                    error=f"Missing columns in CSV: {', '.join(missing)}"
                )

            latest = df.iloc[-1]
            user_inputs = {name: float(latest[name]) for name in FEATURE_LIST}

            if "user_id" in df.columns:
                user_id = str(latest["user_id"])

        # --------------------------
        # MANUAL INPUT
        # --------------------------
        elif input_type == "manual":
            user_inputs = {
                f['name']: float(request.form.get(f['name'], f['default']))
                for f in CORE_20_FEATURES
            }

        else:
            return render_template("index.html", features=CORE_20_FEATURES,
                                   error="Invalid input type.")

        # --------------------------
        # PREDICTION (20 features)
        # --------------------------
        input_df = pd.DataFrame([user_inputs])

        pred_proba = float(pipeline.predict_proba(input_df)[0, 1])
        pred_class = int(pipeline.predict(input_df)[0])
        label = "Aligned" if pred_class == 1 else "Misaligned"

        save_record(user_id, pred_proba, pred_class, input_type)

        return render_template(
            "results.html",
            alignment_score=f"{pred_proba:.4f}",
            alignment_label=label,
            pred_class=pred_class,
            shap_plot_b64=None,
            users=[]
        )

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}", exc_info=True)
        return render_template("index.html", features=CORE_20_FEATURES,
                               error=f"Prediction failed: {str(e)}")

@app.route("/records")
def records():
    return render_template("records.html", records=load_history())

@app.route("/help")
def help_page():
    return render_template("help.html", features=CORE_20_FEATURES)

@app.route("/download_template")
def download_template():
    template = ",".join(["user_id", "date"] + FEATURE_LIST) + "\n"
    return send_file(
        io.BytesIO(template.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name="taralign_template.csv"
    )

@app.route("/clear_history", methods=["POST"])
def clear_history():
    ensure_history_file()
    with history_lock:
        with open(HISTORY_FILE, 'w') as f:
            json.dump([], f)
    return jsonify({"success": True})

if __name__ == "__main__":
    ensure_history_file()
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model missing at: {MODEL_PATH}")
        sys.exit(1)

    print("🚀 TarAlign Flask App Running at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
