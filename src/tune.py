# src/tune.py
import torch
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import joblib
import time

from src.data_loader import load_and_preprocess_data
from src.models import Autoencoder
from src import config

def tune_hyperparameters():
    """
    Performs hyperparameter tuning for the XGBoost model using GridSearchCV.
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

    print("Feature transformation complete. Starting hyperparameter tuning...")
    
    # 2. Define the hyperparameter grid to search
    # These are the parameters we want to test. I've chosen a few key ones.
    param_grid = {
        'n_estimators': [100, 200, 300],         # Number of trees in the forest
        'max_depth': [3, 5, 7],                  # Maximum depth of a tree
        'learning_rate': [0.01, 0.1],            # Step size shrinkage
        'subsample': [0.8, 1.0],                 # Fraction of samples to be used for fitting the individual base learners
        'colsample_bytree': [0.8, 1.0]           # Fraction of columns to be used for each tree
    }

    # 3. Initialize the XGBoost classifier and GridSearchCV
    xgb_classifier = xgb.XGBClassifier(
        objective='multi:softprob',
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_jobs=-1  # Use all available CPU cores
    )
    
    # GridSearchCV will test all combinations of the parameters in param_grid.
    # cv=3 means it will use 3-fold cross-validation.
    # scoring='f1_weighted' tells it to optimize for the weighted F1-score,
    # which is a good metric for imbalanced classes.
    grid_search = GridSearchCV(
        estimator=xgb_classifier,
        param_grid=param_grid,
        scoring='f1_weighted',
        cv=3,
        verbose=2 # This will print progress updates
    )
    
    # 4. Run the Grid Search
    start_time = time.time()
    grid_search.fit(X_train_encoded, y_train)
    end_time = time.time()
    
    print(f"Grid search completed in { (end_time - start_time) / 60:.2f} minutes.")
    
    # 5. Print the best parameters and save the best model
    print("Best parameters found: ", grid_search.best_params_)
    print("Best weighted F1-score found: ", grid_search.best_score_)
    
    best_xgb_model = grid_search.best_estimator_
    best_xgb_model.save_model(f'{config.MODEL_DIR}xgb_classifier_tuned.json')
    print("Tuned XGBoost model saved to 'models/xgb_classifier_tuned.json'")

if __name__ == '__main__':
    tune_hyperparameters()