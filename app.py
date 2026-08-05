import os
import json
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import gradio as gr
import uvicorn

# Conditional spaces package load to support both local run and HF Spaces
# This prevents local ModuleNotFoundError while ensuring the HF ZeroGPU runtime finds the real package
ON_HF = "SPACE_ID" in os.environ
if ON_HF:
    import spaces
else:
    class spaces:
        @staticmethod
        def GPU(func):
            return func

# Initialize FastAPI application
app = FastAPI(title="DiaNo Clinical Portal")

# Load model, features order, and threshold config
MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgb_tuned_model.json")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "model_features.json")
THRESHOLD_PATH = os.path.join(os.path.dirname(__file__), "optimal_threshold.json")

# Load model features
if not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError(f"Missing features configuration at {FEATURES_PATH}")
with open(FEATURES_PATH, "r") as f:
    FEATURES_ORDER = json.load(f)

# Load threshold
DEFAULT_THRESHOLD = 0.60
if os.path.exists(THRESHOLD_PATH):
    with open(THRESHOLD_PATH, "r") as f:
        threshold_data = json.load(f)
        THRESHOLD = threshold_data.get("optimal_threshold", DEFAULT_THRESHOLD)
else:
    THRESHOLD = DEFAULT_THRESHOLD

# Load XGBoost Model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing XGBoost model at {MODEL_PATH}")
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

# Define prediction function wrapped with ZeroGPU decorator
@spaces.GPU
def predict_inference(features_array):
    return model.predict_proba(features_array)

def validate_and_convert_inputs(data):
    errors = {}
    clean_data = {}
    
    # Binary fields validation
    binary_fields = {
        "HighBP": "High blood pressure",
        "HighChol": "High cholesterol",
        "CholCheck": "Cholesterol check 5 years",
        "Smoker": "Smoker status",
        "Stroke": "Stroke history",
        "HeartDiseaseorAttack": "Heart disease or attack",
        "PhysActivity": "Physical activity",
        "Fruits": "Fruit consumption",
        "Veggies": "Vegetable consumption",
        "HvyAlcoholConsump": "Heavy alcohol consumption",
        "AnyHealthcare": "Healthcare coverage",
        "NoDocbcCost": "Doctor cost barrier",
        "DiffWalk": "Difficulty walking",
        "Sex": "Sex"
    }
    
    for field, label in binary_fields.items():
        if field not in data or data[field] is None:
            errors[field] = f"{label} is required."
        else:
            try:
                val = int(data[field])
                if val not in [0, 1]:
                    errors[field] = f"Value must be 0 (No) or 1 (Yes)."
                else:
                    clean_data[field] = val
            except (ValueError, TypeError):
                errors[field] = "Must be a binary choice (0 or 1)."
                
    # BMI validation
    if "BMI" not in data or data["BMI"] is None:
        errors["BMI"] = "BMI is required."
    else:
        try:
            val = float(data["BMI"])
            if val < 10.0 or val > 99.9:
                errors["BMI"] = f"BMI must be between 10.0 and 99.9. Got {val}."
            else:
                clean_data["BMI"] = val
        except (ValueError, TypeError):
            errors["BMI"] = "BMI must be a number."
            
    # GenHlth (1 to 5)
    if "GenHlth" not in data or data["GenHlth"] is None:
        errors["GenHlth"] = "General Health indicator is required."
    else:
        try:
            val = int(data["GenHlth"])
            if val < 1 or val > 5:
                errors["GenHlth"] = f"General Health must be between 1 (Excellent) and 5 (Poor)."
            else:
                clean_data["GenHlth"] = val
        except (ValueError, TypeError):
            errors["GenHlth"] = "Must be an integer between 1 and 5."

    # MentHlth (0 to 30)
    if "MentHlth" not in data or data["MentHlth"] is None:
        errors["MentHlth"] = "Mental Health days is required."
    else:
        try:
            val = int(data["MentHlth"])
            if val < 0 or val > 30:
                errors["MentHlth"] = "Mental Health days must be between 0 and 30."
            else:
                clean_data["MentHlth"] = val
        except (ValueError, TypeError):
            errors["MentHlth"] = "Must be an integer between 0 and 30."

    # PhysHlth (0 to 30)
    if "PhysHlth" not in data or data["PhysHlth"] is None:
        errors["PhysHlth"] = "Physical Health days is required."
    else:
        try:
            val = int(data["PhysHlth"])
            if val < 0 or val > 30:
                errors["PhysHlth"] = "Physical Health days must be between 0 and 30."
            else:
                clean_data["PhysHlth"] = val
        except (ValueError, TypeError):
            errors["PhysHlth"] = "Must be an integer between 0 and 30."

    # Age category (1 to 13)
    if "Age" not in data or data["Age"] is None:
        errors["Age"] = "Age category is required."
    else:
        try:
            val = int(data["Age"])
            if val < 1 or val > 13:
                errors["Age"] = "Age category must be between 1 and 13."
            else:
                clean_data["Age"] = val
        except (ValueError, TypeError):
            errors["Age"] = "Must be an integer between 1 and 13."

    # Education category (1 to 6)
    if "Education" not in data or data["Education"] is None:
        errors["Education"] = "Education category is required."
    else:
        try:
            val = int(data["Education"])
            if val < 1 or val > 6:
                errors["Education"] = "Education category must be between 1 and 6."
            else:
                clean_data["Education"] = val
        except (ValueError, TypeError):
            errors["Education"] = "Must be an integer between 1 and 6."

    # Income category (1 to 8)
    if "Income" not in data or data["Income"] is None:
        errors["Income"] = "Income category is required."
    else:
        try:
            val = int(data["Income"])
            if val < 1 or val > 8:
                errors["Income"] = "Income category must be between 1 and 8."
            else:
                clean_data["Income"] = val
        except (ValueError, TypeError):
            errors["Income"] = "Must be an integer between 1 and 8."

    return clean_data, errors

# Serve static clinical landing page (Registered first so "/" matches our custom view)
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

# API Configuration route
@app.get("/api/config")
async def get_config():
    return JSONResponse({
        "features": FEATURES_ORDER,
        "decision_threshold": THRESHOLD
    })

# API Inference route
@app.post("/api/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        if not data:
            return JSONResponse({"success": False, "error": "No input data provided."}, status_code=400)

        clean_inputs, validation_errors = validate_and_convert_inputs(data)
        if validation_errors:
            return JSONResponse({
                "success": False,
                "error": "Validation failed",
                "validation_errors": validation_errors
            }, status_code=400)

        # Construct input array in the exact order model expects
        ordered_features = []
        for feature in FEATURES_ORDER:
            ordered_features.append(clean_inputs[feature])

        # Prepare for inference
        features_array = np.array([ordered_features], dtype=np.float32)

        # Run inference using ZeroGPU-wrapped function
        probabilities = predict_inference(features_array)
        diabetes_prob = float(probabilities[0][1])

        # Binary decision based on decision threshold of 0.60
        prediction = 1 if diabetes_prob >= THRESHOLD else 0
        prediction_label = "High Likelihood of Diabetes" if prediction == 1 else "Low Likelihood of Diabetes"

        return JSONResponse({
            "success": True,
            "probability": diabetes_prob,
            "prediction": prediction,
            "prediction_label": prediction_label,
            "threshold": THRESHOLD,
            "input_aligned": {name: clean_inputs[name] for name in FEATURES_ORDER}
        })

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

# Build a simple dummy Gradio App to satisfy Hugging Face Space checks
with gr.Blocks(title="DiaNo Clinical Portal") as demo:
    gr.Markdown("# DiaNo Clinical Portal")
    gr.Markdown("The interactive clinical dashboard is served directly on the root path `/` of this Space.")
    gr.Markdown("You can navigate directly to the Space URL to view the main clinical screening form.")

# Mount the Gradio App onto FastAPI at root "/"
# This allows standard Gradio system endpoints (/config, /info) to be served, passing the platform health check.
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    # Hugging Face sets PORT env variable automatically
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
