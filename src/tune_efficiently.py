# src/tune_efficiently.py
import torch
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import time

from src.data_loader import load_and_preprocess_data
from src.models import Autoencoder
from src import config

def tune_efficiently():
    """
    Performs efficient hyperparameter tuning using RandomizedSearchCV.
    """
    print("Loading data and transforming features...")
    # 1. Load data and transform it using the pre-trained autoencoder
    X_train, _, y_train, _, _, _ = load_and_preprocess_data()
    
    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load(f'{config.MODEL_DIR}autoencoder.pth'))
    autoencoder.eval()
    
    with torch.no_grad():
        X_train_tensor = torch.FloatTensor(X_train)
        X_train_encoded = autoencoder.encoder(X_train_tensor).numpy()

    print("Feature transformation complete. Starting efficient hyperparameter tuning...")
    
    # 2. Define the hyperparameter search space
    # This is the same wide grid as before, but we will only sample from it.
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }

    # 3. Initialize the XGBoost classifier and RandomizedSearchCV
    xgb_classifier = xgb.XGBClassifier(
        objective='multi:softprob',
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_jobs=-1  # Use all available CPU cores
    )
    
    # This is the key change.
    # n_iter=10: Randomly picks and tests only 10 parameter combinations.
    # cv=2: Uses 2-fold cross-validation, which is faster than 3.
    random_search = RandomizedSearchCV(
        estimator=xgb_classifier,
        param_distributions=param_distributions,
        n_iter=10,
        scoring='f1_weighted',
        cv=2,
        verbose=2,
        random_state=42 # for reproducible results
    )
    
    # 4. Run the search
    start_time = time.time()
    random_search.fit(X_train_encoded, y_train)
    end_time = time.time()
    
    print(f"Efficient search completed in { (end_time - start_time) / 60:.2f} minutes.")
    
    # 5. Print the best parameters and save the best model
    print("Best parameters found: ", random_search.best_params_)
    print("Best weighted F1-score found: ", random_search.best_score_)
    
    best_xgb_model = random_search.best_estimator_
    best_xgb_model.save_model(f'{config.MODEL_DIR}xgb_classifier_tuned.json')
    print(f"Tuned XGBoost model saved to '{config.MODEL_DIR}xgb_classifier_tuned.json'")

if __name__ == '__main__':
    tune_efficiently()