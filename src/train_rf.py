# src/train_rf.py
from sklearn.ensemble import RandomForestClassifier
import joblib
from src.data_loader import load_and_preprocess_data

X_train, _, y_train, _, _, _, _ = load_and_preprocess_data()

print("\n--- Training Final RandomForest Classifier ---")
rf_classifier = RandomForestClassifier(
    n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1
)
rf_classifier.fit(X_train, y_train)
joblib.dump(rf_classifier, 'models/rf_classifier.joblib')
print("Final RandomForest model trained and saved.")