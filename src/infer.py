import pandas as pd
import mlflow.sklearn
from features import build_features
import joblib
import shap
import numpy as np

model_name = "Anomaly-Detection"
model_version = "latest"

model = mlflow.sklearn.load_model(
    f"models:/{model_name}/{model_version}"
)

explainer = shap.TreeExplainer(model)

tfidf_vecs = joblib.load("tfidf_vectorizers.joblib")

df_infer = pd.DataFrame([{
    "Context": "req-00004059-0950-420f-a281-70a1f6723694 113d3a99c3da401fbd62cc2caa5b96d2 54fadb412c4e40cdbaed9335e4c35a9e - - -",
    "EventId": "c1d6825f",
    "timestamp": "2017-05-16 03:47:57.786",
    "Component": "nova.osapi_compute.wsgi.server",
    "Level": "INFO",
    "ParameterList": "['10.11.10.1', '/v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail', '/1.1', '111', '1893', '0.2651320']",
    "EventCount": 1
}])

X_infer, _ = build_features(
    df_infer,
    tfidf_vectorizers=tfidf_vecs,
    mode="infer"
)


def explain_anomaly(i, shap_values, X):
    contrib = shap_values[i]
    top = np.argsort(np.abs(contrib))[:][::-1]

    explanation = []
    for j in top:
        explanation.append(
            f"{X.columns[j]}: {X.iloc[i, j]:2f} (impact={contrib[j]:.3f})"
        )
    return explanation

shap_values = explainer.shap_values(X_infer)

print("decision_function:", model.decision_function(X_infer)[-1])
print("predict (1: normal, -1: anomaly):", model.predict(X_infer)[-1])
print("Top feature contributions to anomaly score:")
explanation = explain_anomaly(-1, shap_values, X_infer)
for line in explanation:
    print(" ", line)