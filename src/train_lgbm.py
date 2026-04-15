# # src/train_lgbm.py
# import torch
# import lightgbm as lgb
# import joblib
# import os

# from src.data_loader import load_and_preprocess_data
# from src.models import Autoencoder
# from src import config

# def train_lgbm():
#     """
#     Trains a LightGBM classifier on the autoencoder's features.
#     """
#     os.makedirs(config.MODEL_DIR, exist_ok=True)
#     os.makedirs(config.REPORTS_DIR, exist_ok=True)

#     # 1. Load data using our final, high-accuracy data loader
#     X_train, _, y_train, _, _, label_encoder = load_and_preprocess_data()
    
#     # 2. Load the pre-trained autoencoder to transform features
#     print("Loading pre-trained autoencoder to transform features...")
#     autoencoder = Autoencoder()
#     autoencoder.load_state_dict(torch.load(f'{config.MODEL_DIR}autoencoder.pth'))
#     autoencoder.eval()
    
#     with torch.no_grad():
#         X_train_tensor = torch.FloatTensor(X_train)
#         X_train_encoded = autoencoder.encoder(X_train_tensor).numpy()

#     # 3. Train the LightGBM Classifier
#     # print("\n--- Starting LightGBM Classifier Training ---")
#     # lgbm_classifier = lgb.LGBMClassifier(
#     #     objective='multiclass',
#     #     metric='multi_logloss',
#     #     n_estimators=config.XGB_N_ESTIMATORS, # Can reuse the same number
#     #     learning_rate=config.XGB_LEARNING_RATE, # Can reuse the same rate
#     #     n_jobs=-1
#     # )
    
#     # lgbm_classifier.fit(X_train_encoded, y_train)
    
#     # src/train_lgbm.py

# # ...
# # 3. Train the LightGBM Classifier with better parameters
#     print("\n--- Starting LightGBM Classifier Training with Constrained Parameters ---")
#     lgbm_classifier = lgb.LGBMClassifier(
#         objective='multiclass',
#         metric='multi_logloss',
#         # --- KEY CHANGES ARE HERE ---
#         n_estimators=400,         # More trees to compensate for shallower depth
#         learning_rate=0.05,       # A slightly smaller learning rate
#         max_depth=9,              # **Crucial:** Prevents the trees from getting too deep
#         num_leaves=50,            # Another way to control complexity (must be < 2^max_depth)
#         # ---
#         n_jobs=-1
#     )

#     lgbm_classifier.fit(X_train_encoded, y_train)
# # ...

#     # Save the trained LGBM model
#     joblib.dump(lgbm_classifier, f'{config.MODEL_DIR}lgbm_classifier.joblib')
#     # Resave the label encoder just in case
#     joblib.dump(label_encoder, f'{config.MODEL_DIR}label_encoder.joblib')
#     print("LightGBM classifier trained successfully and saved to 'models/'.")

# if __name__ == '__main__':
#     train_lgbm()




# src/train_lgbm.py
import lightgbm as lgb
import joblib
from src.data_loader import load_and_preprocess_data

X_train, _, y_train, _, _, _, _ = load_and_preprocess_data()

print("\n--- Training Final LightGBM Classifier ---")
lgbm_classifier = lgb.LGBMClassifier(
    objective='multiclass', n_estimators=400, learning_rate=0.05,
    max_depth=9, num_leaves=50, random_state=42, n_jobs=-1
)
lgbm_classifier.fit(X_train, y_train)
joblib.dump(lgbm_classifier, 'models/lgbm_classifier.joblib')
print("Final LightGBM model trained and saved.")