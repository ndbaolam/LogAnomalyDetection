import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from features import build_features
import mlflow
import joblib
import mlflow.sklearn

# ======================
# 1. Load & feature
# ======================
df = pd.read_csv("data/processed/one_event_sequences.csv")
X, tfidf_vecs = build_features(df, mode="train")

# ======================
# 2. Train / Test split
# ======================
X_train, X_test = train_test_split(
    X,
    test_size=0.2,
    random_state=42
)

# ======================
# 3. Model
# ======================
model = IsolationForest(
    n_estimators=1000,
    contamination="auto",
    max_samples="auto",
    random_state=42
)

mlflow.sklearn.autolog()

with mlflow.start_run():

    # ======================
    # 4. Train
    # ======================
    model.fit(X_train)

    # ======================
    # 5. Basic anomaly metric
    # ======================
    scores = model.decision_function(X_test)
    threshold = np.percentile(scores, 1)
    anomaly_rate = (scores < threshold).mean()
    
    mlflow.log_metric("anomaly_rate_test", anomaly_rate)
    mlflow.log_metric("threshold", threshold)

    # ======================
    # 6. Save model to MLflow
    # ======================
    mlflow.sklearn.log_model(
        sk_model=model,
        name="isolation_forest",
        registered_model_name="Anomaly-Detection"
    )

    joblib.dump(tfidf_vecs, "tfidf_vectorizers.joblib")
    mlflow.log_artifact("tfidf_vectorizers.joblib")

print("✅ Model metrics logged to MLflow")
