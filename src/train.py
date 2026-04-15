# # src/train.py
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import xgboost as xgb
# import joblib
# import os

# from src.data_loader import load_and_preprocess_data
# from src.models import Autoencoder
# from src import config

# def train():
#     """
#     The main training function. It orchestrates the entire model training pipeline.
#     """
#     # Create model and report directories if they don't exist
#     os.makedirs(config.MODEL_DIR, exist_ok=True)
#     os.makedirs(config.REPORTS_DIR, exist_ok=True)

#     # 1. Load and preprocess the data
#     # We only need the training data and the label encoder for this script.
#     X_train, _, y_train, _, _, label_encoder = load_and_preprocess_data()
    
#     # ===================================================================
#     #   Part 1: Train the Autoencoder for Feature Extraction
#     # ===================================================================
#     print("\n--- Starting Autoencoder Training ---")
    
#     # Isolate the benign training data to teach the autoencoder what "normal" looks like
#     benign_label_name = 'BenignTraffic'
#     benign_label_encoded = label_encoder.transform([benign_label_name])[0]
    
#     X_train_benign = X_train[y_train == benign_label_encoded]
#     X_train_benign_tensor = torch.FloatTensor(X_train_benign)
    
#     # Initialize the model, loss function, and optimizer
#     autoencoder = Autoencoder()
#     criterion = nn.MSELoss() # Mean Squared Error is a good choice for reconstruction loss
#     optimizer = optim.Adam(autoencoder.parameters(), lr=config.AE_LEARNING_RATE)

#     # The training loop
#     for epoch in range(config.AE_EPOCHS):
#         # Forward pass: compute predicted outputs by passing inputs to the model
#         outputs = autoencoder(X_train_benign_tensor)
#         # Calculate the loss
#         loss = criterion(outputs, X_train_benign_tensor)
        
#         # Backward pass and optimization
#         optimizer.zero_grad() # Clear previous gradients
#         loss.backward()       # Compute gradients
#         optimizer.step()      # Update weights
        
#         if (epoch+1) % 10 == 0:
#             print(f'Epoch [{epoch+1}/{config.AE_EPOCHS}], Loss: {loss.item():.6f}')
            
#     # Save the trained autoencoder model's state
#     torch.save(autoencoder.state_dict(), f'{config.MODEL_DIR}autoencoder.pth')
#     print("Autoencoder trained successfully and saved to 'models/'.")

#     # ===================================================================
#     #   Part 2: Train the XGBoost Classifier
#     # ===================================================================
#     print("\n--- Starting XGBoost Classifier Training ---")
    
#     # First, use the trained encoder to transform the *entire* training set
#     print("Transforming features with the trained encoder...")
#     autoencoder.eval() # Set the model to evaluation mode
#     with torch.no_grad(): # Disable gradient calculations for inference
#         X_train_tensor = torch.FloatTensor(X_train)
#         X_train_encoded = autoencoder.encoder(X_train_tensor).numpy()

#     # Now, train XGBoost on these new, powerful features
#     print("Training XGBoost classifier on encoded features...")
#     xgb_classifier = xgb.XGBClassifier(
#         objective='multi:softprob',
#         use_label_encoder=False,
#         eval_metric='mlogloss',
#         n_estimators=config.XGB_N_ESTIMATORS,
#         learning_rate=config.XGB_LEARNING_RATE,
#         n_jobs=-1 # Use all available CPU cores
#     )
#     xgb_classifier.fit(X_train_encoded, y_train)
    
#     # Save the trained XGBoost model and the label encoder for later use in evaluation
#     xgb_classifier.save_model(f'{config.MODEL_DIR}xgb_classifier.json')
#     joblib.dump(label_encoder, f'{config.MODEL_DIR}label_encoder.joblib')
#     print("XGBoost classifier trained successfully and saved to 'models/'.")

# if __name__ == '__main__':
#     train()



# src/train.py
import xgboost as xgb
from src.data_loader import load_and_preprocess_data

X_train, _, y_train, _, _, _, _ = load_and_preprocess_data()

print("\n--- Training Final XGBoost Classifier ---")
xgb_classifier = xgb.XGBClassifier(
    objective='multi:softmax', use_label_encoder=False, eval_metric='mlogloss',
    n_estimators=200, learning_rate=0.1, tree_method='gpu_hist', random_state=42
)
xgb_classifier.fit(X_train, y_train)
xgb_classifier.save_model('models/xgb_classifier.json')
print("Final XGBoost model trained and saved.")