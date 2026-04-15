# # # src/evaluate.py
# # import torch
# # import numpy as np
# # import xgboost as xgb
# # import joblib
# # from sklearn.metrics import classification_report, confusion_matrix
# # import seaborn as sns
# # import matplotlib.pyplot as plt

# # from src.data_loader import load_and_preprocess_data
# # from src.models import Autoencoder
# # from src import config

# # def evaluate():
# #     """
# #     Loads the trained models and evaluates their performance on the test set.
# #     """
# #     # 1. Load the test data and the saved label encoder
# #     print("Loading test data and saved models...")
# #     _, X_test, _, y_test, _, label_encoder = load_and_preprocess_data()
    
# #     # Load the trained Autoencoder
# #     autoencoder = Autoencoder()
# #     autoencoder.load_state_dict(torch.load(f'{config.MODEL_DIR}autoencoder.pth'))
# #     autoencoder.eval() # Set the model to evaluation mode
    
# #     # Load the trained XGBoost classifier
# #     xgb_classifier = xgb.XGBClassifier()
# #     xgb_classifier.load_model(f'{config.MODEL_DIR}xgb_classifier.json')
    
# #     # 2. Transform the test data using the autoencoder's encoder
# #     print("Transforming test features...")
# #     with torch.no_grad(): # Disable gradient calculation for inference
# #         X_test_tensor = torch.FloatTensor(X_test)
# #         X_test_encoded = autoencoder.encoder(X_test_tensor).numpy()
        
# #     # 3. Make predictions with the XGBoost model
# #     print("Making predictions...")
# #     y_pred = xgb_classifier.predict(X_test_encoded)
    
# #     # 4. Generate and save the performance reports
# #     print("\n--- Classification Report ---")
# #     # Generate the report with actual class names
# #     report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
# #     print(report)
    
# #     # Save the report to a text file
# #     with open(f"{config.REPORTS_DIR}classification_report.txt", "w") as f:
# #         f.write(report)

# #     # Generate and save the confusion matrix plot
# #     print("Generating confusion matrix...")
# #     cm = confusion_matrix(y_test, y_pred)
# #     plt.figure(figsize=(12, 10))
# #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
# #                 xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
# #     plt.title('Confusion Matrix')
# #     plt.ylabel('Actual Label')
# #     plt.xlabel('Predicted Label')
# #     plt.tight_layout()
# #     plt.savefig(f"{config.REPORTS_DIR}confusion_matrix.png")
    
# #     print(f"\nEvaluation complete. Report and confusion matrix saved in '{config.REPORTS_DIR}' directory.")
    

# # if __name__ == '__main__':
# #     evaluate()



# # src/evaluate.py
# import torch
# import joblib
# import xgboost as xgb
# import lightgbm as lgb
# import seaborn as sns
# import matplotlib.pyplot as plt
# import argparse  # Used to handle command-line arguments

# from sklearn.metrics import classification_report, confusion_matrix
# from src.data_loader import load_and_preprocess_data
# from src.models import Autoencoder
# from src import config

# def evaluate(model_type: str):
#     """
#     Loads a trained model and evaluates its performance on the test set.

#     Args:
#         model_type (str): The type of model to evaluate ('xgb' or 'lgbm').
#     """
#     print(f"--- Starting Evaluation for {model_type.upper()} Model ---")

#     # 1. Load the test data
#     print("Loading test data...")
#     _, X_test, _, y_test, _, label_encoder = load_and_preprocess_data()
    
#     # 2. Load the pre-trained Autoencoder for feature transformation
#     print("Loading pre-trained autoencoder...")
#     autoencoder = Autoencoder()
#     autoencoder.load_state_dict(torch.load(f'{config.MODEL_DIR}autoencoder.pth'))
#     autoencoder.eval()
    
#     # 3. Load the specified trained classifier
#     print(f"Loading {model_type.upper()} classifier...")
#     if model_type == 'xgb':
#         classifier = xgb.XGBClassifier()
#         classifier.load_model(f'{config.MODEL_DIR}xgb_classifier.json')
#     elif model_type == 'lgbm':
#         classifier = joblib.load(f'{config.MODEL_DIR}lgbm_classifier.joblib')
#     else:
#         raise ValueError("Invalid model_type specified. Choose 'xgb' or 'lgbm'.")

#     # 4. Transform test data features using the autoencoder's encoder
#     print("Transforming test features...")
#     with torch.no_grad():
#         X_test_tensor = torch.FloatTensor(X_test)
#         X_test_encoded = autoencoder.encoder(X_test_tensor).numpy()
        
#     # 5. Make predictions
#     print("Making predictions...")
#     y_pred = classifier.predict(X_test_encoded)
    
#     # 6. Generate and save the performance reports
#     print("\n--- Classification Report ---")
#     report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
#     print(report)
    
#     # Save the report to a text file, naming it after the model type
#     report_filename = f"{config.REPORTS_DIR}classification_report_{model_type}.txt"
#     with open(report_filename, "w") as f:
#         f.write(report)

#     # Generate and save the confusion matrix plot
#     print("Generating confusion matrix...")
#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
#     plt.title(f'Confusion Matrix - {model_type.upper()} Model')
#     plt.ylabel('Actual Label')
#     plt.xlabel('Predicted Label')
#     plt.tight_layout()
    
#     # Save the plot, naming it after the model type
#     plot_filename = f"{config.REPORTS_DIR}confusion_matrix_{model_type}.png"
#     plt.savefig(plot_filename)
    
#     print(f"\nEvaluation complete. Report and plot saved to '{config.REPORTS_DIR}'.")

# if __name__ == '__main__':
#     # Set up the command-line argument parser
#     parser = argparse.ArgumentParser(description="Evaluate a trained classifier.")
#     parser.add_argument(
#         '--model_type', 
#         type=str, 
#         required=True, 
#         choices=['xgb', 'lgbm'],
#         help="The type of model to evaluate: 'xgb' for XGBoost or 'lgbm' for LightGBM."
#     )
#     args = parser.parse_args()
    
#     # Run the evaluation function with the specified model type
#     evaluate(args.model_type)




# src/evaluate.py
import torch
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.data_loader import load_and_preprocess_data
from src.models import Autoencoder
from src import config

def evaluate(model_type: str):
    print(f"--- Starting Evaluation for {model_type.upper()} Model ---")
    _, X_test, _, y_test, _, label_encoder, _ = load_and_preprocess_data()
    
    # Save the encoder classes for use in other scripts
    np.save(f'{config.MODEL_DIR}label_encoder_classes.npy', label_encoder.classes_)
    
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

    # NOTE: We are no longer using the Autoencoder in this high-accuracy approach
    # The powerful models can work directly on the scaled features.
    X_test_final = X_test
        
    print("Making predictions...")
    y_pred = classifier.predict(X_test_final)
    
    print("\n--- Classification Report ---")
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, digits=4)
    print(report)
    
    report_filename = f"{config.REPORTS_DIR}classification_report_{model_type}.txt"
    with open(report_filename, "w") as f:
        f.write(report)

    print("Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - {model_type.upper()} Model')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plot_filename = f"{config.REPORTS_DIR}confusion_matrix_{model_type}.png"
    plt.savefig(plot_filename)
    
    print(f"\nEvaluation complete. Report and plot saved to '{config.REPORTS_DIR}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier.")
    parser.add_argument('--model_type', type=str, required=True, choices=['xgb', 'lgbm', 'rf'])
    args = parser.parse_args()
    evaluate(args.model_type)