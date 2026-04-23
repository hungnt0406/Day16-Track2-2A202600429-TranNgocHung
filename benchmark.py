"""
Lab 16 — Phần 7.6 — LightGBM Credit-Card Fraud benchmark.

Run on the cloud CPU node after installing dependencies and downloading the
Kaggle `mlg-ulb/creditcardfraud` dataset:

    pip3 install lightgbm scikit-learn pandas numpy
    kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p .
    python3 benchmark.py

Produces:
    - stdout: human-readable metrics table (screenshot this for Phần 7.8)
    - benchmark_result.json: machine-readable metrics (submit this file)
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path("creditcard.csv")
RESULT_PATH = Path("benchmark_result.json")
RANDOM_STATE = 42


def log(msg: str) -> None:
    print(msg, flush=True)


def _download_from_openml() -> None:
    """Fetch the same mlg-ulb/creditcardfraud dataset from OpenML (no auth)."""
    from sklearn.datasets import fetch_openml

    log("creditcard.csv not found — fetching from OpenML (dataset id=1597)...")
    t0 = time.perf_counter()
    ds = fetch_openml(data_id=1597, as_frame=True, parser="liac-arff")
    elapsed = time.perf_counter() - t0
    df = ds.frame.copy()
    if "Class" not in df.columns and "class" in df.columns:
        df = df.rename(columns={"class": "Class"})
    df["Class"] = df["Class"].astype(int)
    df.to_csv(DATA_PATH, index=False)
    log(f"OpenML download done in {elapsed:.1f}s "
        f"({len(df):,} rows, saved to {DATA_PATH}).")


def load_data() -> tuple[pd.DataFrame, pd.Series, float]:
    if not DATA_PATH.exists():
        try:
            _download_from_openml()
        except Exception as exc:
            sys.exit(
                f"ERROR: {DATA_PATH} not found and OpenML fallback failed "
                f"({exc!r}).\nDownload it manually:\n"
                "  kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p ."
            )
    t0 = time.perf_counter()
    df = pd.read_csv(DATA_PATH)
    load_time = time.perf_counter() - t0
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)
    return X, y, load_time


def train(X_train, y_train, X_val, y_val) -> tuple[lgb.Booster, float]:
    params = {
        "objective": "binary",
        "metric": "auc",
        "is_unbalance": True,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": RANDOM_STATE,
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    t0 = time.perf_counter()
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )
    train_time = time.perf_counter() - t0
    return model, train_time


def evaluate(model: lgb.Booster, X_test, y_test) -> dict:
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y_test, y_pred_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    }


def measure_latency(model: lgb.Booster, X_test: pd.DataFrame) -> dict:
    one_row = X_test.iloc[[0]].to_numpy()
    for _ in range(20):
        model.predict(one_row, num_iteration=model.best_iteration)

    n_single = 1000
    t0 = time.perf_counter()
    for _ in range(n_single):
        model.predict(one_row, num_iteration=model.best_iteration)
    single_elapsed = time.perf_counter() - t0
    latency_1row_ms = single_elapsed / n_single * 1000

    batch = X_test.iloc[:1000].to_numpy()
    for _ in range(5):
        model.predict(batch, num_iteration=model.best_iteration)

    n_batches = 50
    t0 = time.perf_counter()
    for _ in range(n_batches):
        model.predict(batch, num_iteration=model.best_iteration)
    batch_elapsed = time.perf_counter() - t0
    avg_batch_time = batch_elapsed / n_batches
    throughput_1000rows = 1000.0 / avg_batch_time

    return {
        "inference_latency_1row_ms": float(latency_1row_ms),
        "inference_throughput_1000rows_per_sec": float(throughput_1000rows),
        "inference_batch_time_1000rows_ms": float(avg_batch_time * 1000),
    }


def main() -> None:
    log("=" * 60)
    log("Lab 16 — Phần 7.6 — LightGBM Credit-Card Fraud benchmark")
    log("=" * 60)
    log(f"Python:   {platform.python_version()}")
    log(f"Platform: {platform.platform()}")
    log(f"LightGBM: {lgb.__version__}")
    log(f"NumPy:    {np.__version__}")
    log(f"Pandas:   {pd.__version__}")
    log("")

    log("[1/4] Loading data ...")
    X, y, load_time = load_data()
    log(f"      rows={len(X):,}  positives={int(y.sum()):,}  "
        f"pos_rate={y.mean():.4%}  load_time={load_time:.3f}s")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    log(f"      train={len(X_train):,}  test={len(X_test):,}")

    log("\n[2/4] Training LightGBM ...")
    model, train_time = train(X_train, y_train, X_test, y_test)
    best_iter = int(model.best_iteration or model.current_iteration())
    log(f"      training_time={train_time:.2f}s  best_iteration={best_iter}")

    log("\n[3/4] Evaluating ...")
    metrics = evaluate(model, X_test, y_test)
    for k, v in metrics.items():
        log(f"      {k:10s} = {v:.6f}")

    log("\n[4/4] Measuring inference latency / throughput ...")
    latency = measure_latency(model, X_test)
    log(f"      latency_1row      = {latency['inference_latency_1row_ms']:.4f} ms")
    log(f"      batch_time_1k     = {latency['inference_batch_time_1000rows_ms']:.4f} ms")
    log(f"      throughput (r/s)  = {latency['inference_throughput_1000rows_per_sec']:,.1f}")

    result = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "lightgbm": lgb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "dataset": {
            "path": str(DATA_PATH),
            "rows": int(len(X)),
            "positives": int(y.sum()),
            "pos_rate": float(y.mean()),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "load_data_time_sec": float(load_time),
        "training_time_sec": float(train_time),
        "best_iteration": best_iter,
        **metrics,
        **latency,
    }

    RESULT_PATH.write_text(json.dumps(result, indent=2))
    log(f"\nSaved metrics to {RESULT_PATH.resolve()}")
    log("\n--- benchmark_result.json ---")
    log(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()