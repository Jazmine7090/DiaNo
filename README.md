---
title: DiaNo Diabetes Predictor
emoji: 🩺
colorFrom: teal
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# DiaNo - Clinical Diabetes Risk Assessment Portal

DiaNo is a machine learning-powered web application for diabetes screening and risk prediction. 

This repository is configured for deployment to **Hugging Face Spaces** using the free **Gradio SDK** as a container runner.

## Model Details
* **Model**: Tuned XGBoost Classifier
* **Dataset**: CDC BRFSS 2015 Diabetes Health Indicators
* **Imbalance Handling**: `scale_pos_weight`
* **Internal Threshold**: 0.60
* **Features**: 21 health indicator inputs (including physiological readings, lifestyle questions, and demographic info).

## Local Development
To run this application locally:

1. Clone or copy these repository files.
2. Initialize virtual environment and install requirements:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```
4. Access the clinical portal at `http://127.0.0.1:7860`.
