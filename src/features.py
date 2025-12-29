import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict

TFIDF_COLS = ["EventId", "Component", "status_code"]

def build_features(
    df: pd.DataFrame,
    *,
    tfidf_vectorizers: Dict[str, TfidfVectorizer] = None,
    mode: str = "train",  # "train" | "infer"
):
    X = df.copy()

    # ======================
    # Numeric / time features
    # ======================
    params = X["ParameterList"].apply(ast.literal_eval)
    ts = pd.to_datetime(X["timestamp"], errors="coerce")
    hour = ts.dt.hour

    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    X["is_off_hour"] = ((hour < 8) | (hour > 18)).astype(float)

    def safe_get_num(lst, idx, cast=float, default=np.nan):
        try:
            v = lst[idx]
            if isinstance(v, str):
                v = v.strip().strip("'\"")
            return cast(v)
        except Exception:
            return default

    X["latency"] = params.apply(lambda x: safe_get_num(x, -1, float))
    X["response_size"] = params.apply(lambda x: safe_get_num(x, -2, float, default=0))
    X["status_code"] = params.apply(lambda x: safe_get_num(x, -3, float, default=0))

    # ======================
    # TF-IDF features (FIXED)
    # ======================
    if tfidf_vectorizers is None:
        tfidf_vectorizers = {}

    for col in TFIDF_COLS:
        corpus = X[col].astype(str).values

        if mode == "train":
            tfidf = TfidfVectorizer(
                token_pattern=r"[^ ]+",
                lowercase=False,
                norm=None,
                use_idf=True,
                smooth_idf=True,
            )
            mat = tfidf.fit_transform(corpus)
            tfidf_vectorizers[col] = tfidf
        else:
            tfidf = tfidf_vectorizers[col]
            mat = tfidf.transform(corpus)

        # 🔑 scalar, stable, same dimension
        X[f"tfidf_{col}"] = mat.max(axis=1).toarray().ravel()

    DROP_COLS = [
        "Level",
        "EventId",
        "Component",
        "status_code",
        "ParameterList",
        "timestamp",
        "Context",
        "EventCount",
    ]

    X_final = X.drop(columns=DROP_COLS).dropna().reset_index(drop=True)

    return X_final, tfidf_vectorizers
