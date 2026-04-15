# src/interpret.py
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import shap
import matplotlib.pyplot as plt
import argparse

from src.data_loader import load_and_preprocess_data
from src import config

def interpret_model(model_type: str):
    print(f"--- Generating SHAP interpretation for {model_type.upper()} Model ---")
    
    # 1. Load data to get feature names and a sample for explanation
    _, X_test, _, _, _, _, feature_names = load_and_preprocess_data()
    X_sample = X_test[:1000] # SHAP can be slow, so we use a sample

    # 2. Load the specified trained classifier
    print(f"Loading {model_type.upper()} classifier...")
    if model_type == 'xgb':
        classifier = xgb.XGBClassifier()
        classifier.load_model(f'{config.MODEL_DIR}xgb_classifier.json')
    elif model_type == 'lgbm':
        classifier = joblib.load(f'{config.MODEL_DIR}lgbm_classifier.joblib')
    elif model_type == 'rf':
        classifier = joblib.load(f'{config.MODEL_DIR}rf_classifier.joblib')
    else:
        raise ValueError("Invalid model_type. Choose 'xgb', 'lgbm', or 'rf'.")

    # 3. Create a SHAP explainer and calculate SHAP values
    print("Calculating SHAP values... (This may take a moment)")
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_sample)

    # 4. Generate and save the summary plot
    class_names = np.load(f'{config.MODEL_DIR}label_encoder_classes.npy', allow_pickle=True)
    
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, class_names=class_names, show=False)
    plt.title(f"SHAP Summary Plot - {model_type.upper()}")
    plt.tight_layout()
    
    plot_filename = f"{config.REPORTS_DIR}shap_summary_plot_{model_type}.png"
    plt.savefig(plot_filename, dpi=150)
    plt.close()
    
    print(f"SHAP summary plot saved to '{plot_filename}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interpret a trained classifier using SHAP.")
    parser.add_argument('--model_type', type=str, required=True, choices=['xgb', 'lgbm', 'rf'])
    args = parser.parse_args()
    interpret_model(args.model_type)