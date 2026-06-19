from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
TARGET = "Class"
DEFAULT_FALSE_POSITIVE_COST = 8.0
DEFAULT_FALSE_NEGATIVE_HANDLING_COST = 35.0
DEFAULT_MAX_ALERT_RATE = 0.003
NEAR_OPTIMAL_COST_TOLERANCE_PCT = 0.05
MODEL_SELECTION_COST_TOLERANCE_PCT = 0.05
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
ENGINEERED_FEATURES = [
    "Amount",
    "Amount_log1p",
    "Amount_is_zero",
    "Hour_sin",
    "Hour_cos",
]
FEATURE_NAMES = PCA_FEATURES + ENGINEERED_FEATURES


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-values))


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def fraud_loss_amount(frame: pd.DataFrame, handling_cost: float) -> np.ndarray:
    amount = pd.to_numeric(frame["Amount"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    label = frame[TARGET].to_numpy(dtype=int)
    return np.where(label == 1, np.clip(amount, 0.0, None) + handling_cost, 0.0).astype(float)


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    return value


def download_dataset(raw_path: Path, data_url: str = DATA_URL, force: bool = False) -> Path:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists() and not force:
        return raw_path

    temp_path = raw_path.with_suffix(raw_path.suffix + ".part")
    try:
        with urllib.request.urlopen(data_url, timeout=120) as response, temp_path.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
    except (urllib.error.URLError, TimeoutError, PermissionError, OSError) as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(
            "Could not download the credit card fraud dataset. "
            f"Place creditcard.csv at {raw_path} and rerun with --no-download. "
            f"Original error: {exc}"
        ) from exc

    temp_path.replace(raw_path)
    return raw_path


def validate_schema(df: pd.DataFrame) -> None:
    expected = {"Time", "Amount", TARGET, *PCA_FEATURES}
    missing = sorted(expected.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    unique_targets = set(pd.Series(df[TARGET]).dropna().astype(int).unique().tolist())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"{TARGET} must contain only 0/1 labels; found {sorted(unique_targets)}")


def load_dataset(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    validate_schema(df)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    amount = pd.to_numeric(out["Amount"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    event_time = pd.to_numeric(out["Time"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    hour = (event_time % 86400.0) / 3600.0

    out["Amount"] = amount
    out["Amount_log1p"] = np.log1p(np.clip(amount, 0.0, None))
    out["Amount_is_zero"] = (amount <= 0.0).astype(float)
    out["Hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["Hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["_row_id"] = np.arange(len(out))
    return out


def class_distribution(df: pd.DataFrame, split_name: str = "all") -> dict[str, Any]:
    total = int(len(df))
    frauds = int(df[TARGET].sum())
    non_frauds = total - frauds
    return {
        "split": split_name,
        "rows": total,
        "legitimate": non_frauds,
        "fraud": frauds,
        "fraud_rate": safe_divide(frauds, total),
    }


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    train_end = int(len(ordered) * train_frac)
    val_end = int(len(ordered) * (train_frac + val_frac))
    return (
        ordered.iloc[:train_end].copy(),
        ordered.iloc[train_end:val_end].copy(),
        ordered.iloc[val_end:].copy(),
    )


def stratified_split(
    df: pd.DataFrame,
    seed: int,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_idx: list[np.ndarray] = []
    val_idx: list[np.ndarray] = []
    test_idx: list[np.ndarray] = []

    labels = df[TARGET].to_numpy(dtype=int)
    for label in (0, 1):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        train_end = int(len(idx) * train_frac)
        val_end = int(len(idx) * (train_frac + val_frac))
        train_idx.append(idx[:train_end])
        val_idx.append(idx[train_end:val_end])
        test_idx.append(idx[val_end:])

    train = np.concatenate(train_idx)
    val = np.concatenate(val_idx)
    test = np.concatenate(test_idx)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return df.iloc[train].copy(), df.iloc[val].copy(), df.iloc[test].copy()


def split_dataset(df: pd.DataFrame, split: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if split == "chronological":
        return chronological_split(df)
    if split == "stratified":
        return stratified_split(df, seed=seed)
    raise ValueError(f"Unknown split: {split}")


@dataclass
class Standardizer:
    feature_names: list[str]
    mean: list[float]
    scale: list[float]

    @classmethod
    def fit(cls, frame: pd.DataFrame, feature_names: list[str]) -> "Standardizer":
        matrix = frame[feature_names].to_numpy(dtype=float)
        mean = np.nanmean(matrix, axis=0)
        scale = np.nanstd(matrix, axis=0)
        scale = np.where(scale < 1e-9, 1.0, scale)
        return cls(feature_names=feature_names, mean=mean.tolist(), scale=scale.tolist())

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = frame[self.feature_names].to_numpy(dtype=float)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        return (matrix - np.asarray(self.mean)) / np.asarray(self.scale)


def weighted_log_loss(
    y_true: np.ndarray,
    scores: np.ndarray,
    sample_weight: np.ndarray,
    weights: np.ndarray,
    l2: float,
) -> float:
    clipped = np.clip(scores, 1e-8, 1.0 - 1e-8)
    loss = -sample_weight * (y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped))
    return float(loss.sum() / sample_weight.sum() + 0.5 * l2 * np.dot(weights, weights))


class WeightedLogisticRegression:
    def __init__(
        self,
        *,
        learning_rate: float = 0.05,
        l2: float = 1e-3,
        epochs: int = 80,
        batch_size: int = 8192,
        class_weight: str | None = None,
        seed: int = 42,
        patience: int = 12,
    ) -> None:
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.seed = seed
        self.patience = patience
        self.weights: np.ndarray | None = None
        self.intercept = 0.0
        self.history: list[dict[str, float]] = []

    def _sample_weight(self, y: np.ndarray) -> np.ndarray:
        if self.class_weight != "balanced":
            return np.ones_like(y, dtype=float)

        positives = float(y.sum())
        negatives = float(len(y) - positives)
        if positives == 0.0 or negatives == 0.0:
            return np.ones_like(y, dtype=float)

        positive_weight = len(y) / (2.0 * positives)
        negative_weight = len(y) / (2.0 * negatives)
        return np.where(y == 1, positive_weight, negative_weight).astype(float)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> "WeightedLogisticRegression":
        rng = np.random.default_rng(self.seed)
        y_train = y_train.astype(float)
        y_val = y_val.astype(float)
        n_rows, n_features = x_train.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.intercept = 0.0
        sample_weight = self._sample_weight(y_train)

        best_score = -np.inf
        best_weights = self.weights.copy()
        best_intercept = self.intercept
        stale_epochs = 0

        for epoch in range(1, self.epochs + 1):
            order = rng.permutation(n_rows)
            for start in range(0, n_rows, self.batch_size):
                batch_idx = order[start : start + self.batch_size]
                x_batch = x_train[batch_idx]
                y_batch = y_train[batch_idx]
                w_batch = sample_weight[batch_idx]

                scores = sigmoid(x_batch @ self.weights + self.intercept)
                error = (scores - y_batch) * w_batch
                denom = max(float(w_batch.sum()), 1.0)
                grad_weights = (x_batch.T @ error) / denom + self.l2 * self.weights
                grad_intercept = float(error.sum() / denom)

                self.weights -= self.learning_rate * grad_weights
                self.intercept -= self.learning_rate * grad_intercept

            train_scores = self.predict_proba(x_train)
            val_scores = self.predict_proba(x_val)
            train_loss = weighted_log_loss(y_train, train_scores, sample_weight, self.weights, self.l2)
            val_ap = average_precision(y_val, val_scores)
            val_curve = precision_recall_curve(y_val, val_scores)
            threshold_info = select_threshold(val_curve, policy="f2")
            val_f2 = float(threshold_info["f2"])

            self.history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "val_average_precision": val_ap,
                    "val_f2": val_f2,
                }
            )

            selection_score = val_ap + 0.05 * val_f2
            if selection_score > best_score + 1e-6:
                best_score = selection_score
                best_weights = self.weights.copy()
                best_intercept = self.intercept
                stale_epochs = 0
            else:
                stale_epochs += 1

            if stale_epochs >= self.patience:
                break

        self.weights = best_weights
        self.intercept = best_intercept
        return self

    def predict_proba(self, x_matrix: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model has not been fit yet.")
        return sigmoid(x_matrix @ self.weights + self.intercept)


def precision_recall_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    positive_cost: np.ndarray | None = None,
) -> pd.DataFrame:
    y = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    if positive_cost is None:
        positive_cost = y.astype(float)
    else:
        positive_cost = np.asarray(positive_cost, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y[order]
    score_sorted = scores[order]
    cost_sorted = positive_cost[order]
    positives = int(y.sum())

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    captured_positive_cost = np.cumsum(np.where(y_sorted == 1, cost_sorted, 0.0))
    total_positive_cost = float(positive_cost[y == 1].sum())
    missed_positive_cost = total_positive_cost - captured_positive_cost
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = np.divide(tp, positives, out=np.zeros_like(tp, dtype=float), where=positives > 0)
    return pd.DataFrame(
        {
            "threshold": score_sorted,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "captured_positive_cost": captured_positive_cost,
            "missed_positive_cost": missed_positive_cost,
            "total_positive_cost": total_positive_cost,
        }
    )


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    curve = precision_recall_curve(y_true, scores)
    if curve.empty or int(np.asarray(y_true).sum()) == 0:
        return 0.0
    recall = curve["recall"].to_numpy(dtype=float)
    precision = curve["precision"].to_numpy(dtype=float)
    recall_delta = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(recall_delta * precision))


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return 0.0

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = float(ranks[y == 1].sum())
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def select_threshold(
    curve: pd.DataFrame,
    *,
    policy: str = "cost",
    min_precision: float | None = None,
    false_positive_cost: float = DEFAULT_FALSE_POSITIVE_COST,
    false_negative_handling_cost: float = DEFAULT_FALSE_NEGATIVE_HANDLING_COST,
    max_alert_rate: float = DEFAULT_MAX_ALERT_RATE,
) -> dict[str, float | str]:
    if curve.empty:
        return {
        "threshold": 1.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "f2": 0.0,
            "threshold_policy": policy,
            "false_positive_cost": float(false_positive_cost),
            "false_negative_cost_model": "fraud_amount_plus_handling",
            "false_negative_handling_cost": float(false_negative_handling_cost),
        }

    precision = curve["precision"].to_numpy(dtype=float)
    recall = curve["recall"].to_numpy(dtype=float)
    tp = curve["tp"].to_numpy(dtype=float)
    fp = curve["fp"].to_numpy(dtype=float)
    missed_positive_cost = curve["missed_positive_cost"].to_numpy(dtype=float)
    positives = float(tp[-1])
    total_rows = float(tp[-1] + fp[-1])
    flagged = tp + fp
    flagged_rate = np.divide(flagged, total_rows, out=np.zeros_like(flagged), where=total_rows > 0)
    expected_cost = false_positive_cost * fp + missed_positive_cost
    f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    f2 = np.divide(5.0 * precision * recall, 4.0 * precision + recall, out=np.zeros_like(precision), where=(4.0 * precision + recall) > 0)

    if min_precision is not None:
        precision_mask = precision >= min_precision
    else:
        precision_mask = np.ones_like(precision, dtype=bool)

    if policy == "cost":
        candidates = np.where(precision_mask, -expected_cost, -np.inf)
    elif policy == "capacity":
        capacity_mask = flagged_rate <= max_alert_rate
        mask = capacity_mask & precision_mask
        if not mask.any():
            mask = capacity_mask
        candidates = np.where(mask, recall + 1e-6 * precision, -np.inf)
    elif policy == "f2":
        candidates = np.where(precision_mask, f2, -np.inf)
    elif policy == "f1":
        candidates = np.where(precision_mask, f1, -np.inf)
    else:
        raise ValueError(f"Unknown threshold policy: {policy}")

    idx = int(np.nanargmax(candidates))
    no_model_cost = float(curve["total_positive_cost"].iloc[0])
    cost_savings = no_model_cost - float(expected_cost[idx])
    near_cost_delta = max(abs(float(expected_cost[idx])) * NEAR_OPTIMAL_COST_TOLERANCE_PCT, false_positive_cost * 5.0)
    near_mask = expected_cost <= float(expected_cost[idx]) + near_cost_delta
    near_thresholds = curve["threshold"].to_numpy(dtype=float)[near_mask]
    near_precision = precision[near_mask]
    near_recall = recall[near_mask]
    return {
        "threshold": float(curve.iloc[idx]["threshold"]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
        "f1": float(f1[idx]),
        "f2": float(f2[idx]),
        "alerts": float(flagged[idx]),
        "flagged_rate": float(flagged_rate[idx]),
        "expected_cost": float(expected_cost[idx]),
        "expected_cost_per_transaction": safe_divide(float(expected_cost[idx]), total_rows),
        "cost_savings_vs_no_model": float(cost_savings),
        "no_model_cost": float(no_model_cost),
        "false_positive_cost": float(false_positive_cost),
        "false_negative_cost_model": "fraud_amount_plus_handling",
        "false_negative_handling_cost": float(false_negative_handling_cost),
        "max_alert_rate": float(max_alert_rate),
        "threshold_policy": policy,
        "near_optimal_cost_tolerance_pct": float(NEAR_OPTIMAL_COST_TOLERANCE_PCT),
        "near_optimal_threshold_min": float(near_thresholds.min()) if len(near_thresholds) else float(curve.iloc[idx]["threshold"]),
        "near_optimal_threshold_max": float(near_thresholds.max()) if len(near_thresholds) else float(curve.iloc[idx]["threshold"]),
        "near_optimal_precision_min": float(near_precision.min()) if len(near_precision) else float(precision[idx]),
        "near_optimal_precision_max": float(near_precision.max()) if len(near_precision) else float(precision[idx]),
        "near_optimal_recall_min": float(near_recall.min()) if len(near_recall) else float(recall[idx]),
        "near_optimal_recall_max": float(near_recall.max()) if len(near_recall) else float(recall[idx]),
    }


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None = None) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    pred = np.asarray(y_pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    f2 = safe_divide(5.0 * precision * recall, 4.0 * precision + recall)
    accuracy = safe_divide(tp + tn, len(y))

    metrics = {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "accuracy": accuracy,
        "flagged_rate": safe_divide(tp + fp, len(y)),
        "false_positive_rate": safe_divide(fp, fp + tn),
    }
    if scores is not None:
        metrics["average_precision"] = average_precision(y, scores)
        metrics["roc_auc"] = roc_auc(y, scores)
    return metrics


def metrics_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    false_positive_cost: float = DEFAULT_FALSE_POSITIVE_COST,
    positive_cost: np.ndarray | None = None,
) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    prediction = (np.asarray(scores) >= threshold).astype(int)
    if positive_cost is None:
        positive_cost = y.astype(float)
    else:
        positive_cost = np.asarray(positive_cost, dtype=float)
    metrics = metrics_from_predictions(y, prediction, scores)
    metrics["threshold"] = float(threshold)
    metrics["alerts"] = float(metrics["tp"] + metrics["fp"])
    missed_cost = float(positive_cost[(prediction == 0) & (y == 1)].sum())
    no_model_cost = float(positive_cost[y == 1].sum())
    metrics["missed_fraud_cost"] = missed_cost
    metrics["expected_cost"] = float(false_positive_cost * metrics["fp"] + missed_cost)
    metrics["expected_cost_per_transaction"] = safe_divide(metrics["expected_cost"], len(y))
    metrics["cost_savings_vs_no_model"] = float(no_model_cost - metrics["expected_cost"])
    return metrics


def predict_scores(model: Any, x_matrix: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(x_matrix), dtype=float)
    if probabilities.ndim == 2:
        return probabilities[:, 1]
    return probabilities


def evaluate_model(
    model_name: str,
    model: Any,
    x_val: np.ndarray,
    y_val: np.ndarray,
    val_positive_cost: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_positive_cost: np.ndarray,
    *,
    threshold_policy: str,
    false_positive_cost: float,
    false_negative_handling_cost: float,
    max_alert_rate: float,
    min_precision: float | None,
) -> dict[str, Any]:
    val_scores = predict_scores(model, x_val)
    test_scores = predict_scores(model, x_test)
    val_curve = precision_recall_curve(y_val, val_scores, positive_cost=val_positive_cost)
    selected = select_threshold(
        val_curve,
        policy=threshold_policy,
        min_precision=min_precision,
        false_positive_cost=false_positive_cost,
        false_negative_handling_cost=false_negative_handling_cost,
        max_alert_rate=max_alert_rate,
    )
    threshold = selected["threshold"]
    return {
        "model_name": model_name,
        "threshold": threshold,
        "threshold_selection": selected,
        "validation": metrics_at_threshold(
            y_val,
            val_scores,
            threshold,
            false_positive_cost=false_positive_cost,
            positive_cost=val_positive_cost,
        ),
        "test": metrics_at_threshold(
            y_test,
            test_scores,
            threshold,
            false_positive_cost=false_positive_cost,
            positive_cost=test_positive_cost,
        ),
        "validation_scores": val_scores,
        "test_scores": test_scores,
        "validation_pr_curve": val_curve,
        "training_history": getattr(model, "history", []),
    }


def flatten_metrics(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        for split_name in ("validation", "test"):
            row = {"model": result["model_name"], "split": split_name}
            row.update(result[split_name])
            rows.append(row)
    return pd.DataFrame(rows)


def write_precision_recall_svg(curve: pd.DataFrame, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 760, 460
    left, right, top, bottom = 74, 24, 52, 64
    plot_width = width - left - right
    plot_height = height - top - bottom

    if len(curve) > 900:
        idx = np.linspace(0, len(curve) - 1, 900).astype(int)
        plot_curve = curve.iloc[idx].copy()
    else:
        plot_curve = curve.copy()

    points = []
    for _, row in plot_curve.iterrows():
        x = left + float(row["recall"]) * plot_width
        y = top + (1.0 - float(row["precision"])) * plot_height
        points.append(f"{x:.2f},{y:.2f}")

    path_points = " ".join(points)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{left}" y="30" font-family="Arial" font-size="20" font-weight="700" fill="#1f2937">{title}</text>
  <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#374151" stroke-width="1.5"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#374151" stroke-width="1.5"/>
  <text x="{left + plot_width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="14" fill="#374151">Recall</text>
  <text transform="translate(22 {top + plot_height / 2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14" fill="#374151">Precision</text>
  <text x="{left - 10}" y="{top + 5}" text-anchor="end" font-family="Arial" font-size="12" fill="#6b7280">1.0</text>
  <text x="{left - 10}" y="{top + plot_height + 4}" text-anchor="end" font-family="Arial" font-size="12" fill="#6b7280">0.0</text>
  <text x="{left}" y="{top + plot_height + 22}" text-anchor="middle" font-family="Arial" font-size="12" fill="#6b7280">0.0</text>
  <text x="{left + plot_width}" y="{top + plot_height + 22}" text-anchor="middle" font-family="Arial" font-size="12" fill="#6b7280">1.0</text>
  <polyline fill="none" stroke="#c2410c" stroke-width="3" points="{path_points}"/>
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def write_feature_driver_svg(drivers: pd.DataFrame, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chart = drivers.head(14).copy()
    width, height = 800, 520
    left, right, top, bottom = 160, 34, 58, 36
    plot_width = width - left - right
    row_height = (height - top - bottom) / max(len(chart), 1)
    value_column = "driver_value" if "driver_value" in chart.columns else "coefficient"
    abs_column = "abs_driver_value" if "abs_driver_value" in chart.columns else "abs_coefficient"
    max_abs = max(float(chart[abs_column].max()), 1e-9)
    center = left + plot_width / 2
    signed = bool((chart[value_column] < 0).any())

    bars = []
    for i, row in chart.iterrows():
        value = float(row[value_column])
        y = top + (i + 0.18) * row_height
        h = row_height * 0.64
        if signed:
            length = abs(value) / max_abs * (plot_width / 2 - 16)
            if value >= 0:
                x = center
                color = "#b91c1c"
            else:
                x = center - length
                color = "#1d4ed8"
            text_x = x + (length + 6 if value >= 0 else -6)
            anchor = "start" if value >= 0 else "end"
        else:
            length = abs(value) / max_abs * (plot_width - 16)
            x = left
            color = "#c2410c"
            text_x = x + length + 6
            anchor = "start"
        label_y = y + h * 0.68
        bars.append(
            f'<text x="{left - 10}" y="{label_y:.2f}" text-anchor="end" font-family="Arial" font-size="13" fill="#111827">{row["feature"]}</text>'
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{length:.2f}" height="{h:.2f}" rx="3" fill="{color}"/>'
            f'<text x="{text_x:.2f}" y="{label_y:.2f}" text-anchor="{anchor}" font-family="Arial" font-size="12" fill="#374151">{value:.3f}</text>'
        )

    legend = (
        f'<line x1="{center:.2f}" y1="{top}" x2="{center:.2f}" y2="{height - bottom}" stroke="#9ca3af" stroke-width="1"/>'
        f'<text x="{center - 12}" y="{height - 12}" text-anchor="end" font-family="Arial" font-size="12" fill="#1d4ed8">higher values lower score</text>'
        f'<text x="{center + 12}" y="{height - 12}" text-anchor="start" font-family="Arial" font-size="12" fill="#b91c1c">higher values raise score</text>'
        if signed
        else f'<text x="{left}" y="{height - 12}" text-anchor="start" font-family="Arial" font-size="12" fill="#c2410c">larger bar = larger validation PR-AUC drop when permuted</text>'
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{left}" y="32" font-family="Arial" font-size="20" font-weight="700" fill="#1f2937">{title}</text>
  {legend}
  {''.join(bars)}
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def has_linear_weights(model: Any) -> bool:
    return hasattr(model, "weights") and getattr(model, "weights") is not None


def feature_drivers(
    model: Any,
    feature_names: list[str],
    *,
    x_reference: np.ndarray,
    y_reference: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    if has_linear_weights(model):
        weights = np.asarray(model.weights, dtype=float)
        drivers = pd.DataFrame(
            {
                "feature": feature_names,
                "driver_value": weights,
                "abs_driver_value": np.abs(weights),
                "driver_type": "standardized_logistic_coefficient",
                "interpretation": np.where(
                    weights >= 0.0,
                    "higher values raise fraud score",
                    "higher values lower fraud score",
                ),
            }
        )
        return drivers.sort_values("abs_driver_value", ascending=False).reset_index(drop=True)

    baseline_scores = predict_scores(model, x_reference)
    baseline_ap = average_precision(y_reference, baseline_scores)
    rng = np.random.default_rng(seed)
    rows = []
    for col_idx, feature in enumerate(feature_names):
        permuted = x_reference.copy()
        permuted[:, col_idx] = rng.permutation(permuted[:, col_idx])
        permuted_ap = average_precision(y_reference, predict_scores(model, permuted))
        importance = max(0.0, baseline_ap - permuted_ap)
        rows.append(
            {
                "feature": feature,
                "driver_value": importance,
                "abs_driver_value": importance,
                "driver_type": "permutation_average_precision_drop",
                "interpretation": "validation PR-AUC drop when this feature is permuted",
            }
        )
    return pd.DataFrame(rows).sort_values("driver_value", ascending=False).reset_index(drop=True)


def coefficient_drivers(model: WeightedLogisticRegression, feature_names: list[str]) -> pd.DataFrame:
    if model.weights is None:
        raise RuntimeError("Model has not been fit yet.")
    drivers = pd.DataFrame(
        {
            "feature": feature_names,
            "driver_value": model.weights,
            "abs_driver_value": np.abs(model.weights),
            "driver_type": "standardized_logistic_coefficient",
            "interpretation": np.where(
                model.weights >= 0.0,
                "higher values raise fraud score",
                "higher values lower fraud score",
            ),
        }
    )
    return drivers.sort_values("abs_driver_value", ascending=False).reset_index(drop=True)


def linear_log_odds_contributions(
    test_df: pd.DataFrame,
    x_test: np.ndarray,
    y_test: np.ndarray,
    scores: np.ndarray,
    model: WeightedLogisticRegression,
    feature_names: list[str],
    threshold: float,
    max_rows: int = 30,
) -> pd.DataFrame:
    if model.weights is None:
        raise RuntimeError("Model has not been fit yet.")
    flagged = np.flatnonzero(scores >= threshold)
    if len(flagged) == 0:
        flagged = np.argsort(-scores)[:max_rows]
    else:
        flagged = flagged[np.argsort(-scores[flagged])[:max_rows]]

    contributions = x_test * model.weights
    rows = []
    for rank, idx in enumerate(flagged, start=1):
        contribution_row = contributions[idx]
        positive_idx = np.argsort(-contribution_row)[:5]
        rows.append(
            {
                "rank": rank,
                "row_id": int(test_df.iloc[idx]["_row_id"]),
                "actual_class": int(y_test[idx]),
                "predicted_probability": float(scores[idx]),
                "threshold": float(threshold),
                "amount": float(test_df.iloc[idx]["Amount"]),
                "time_seconds": float(test_df.iloc[idx]["Time"]),
                "top_positive_feature_1": feature_names[int(positive_idx[0])],
                "top_positive_log_odds_contribution_1": float(contribution_row[int(positive_idx[0])]),
                "top_positive_feature_2": feature_names[int(positive_idx[1])],
                "top_positive_log_odds_contribution_2": float(contribution_row[int(positive_idx[1])]),
                "top_positive_feature_3": feature_names[int(positive_idx[2])],
                "top_positive_log_odds_contribution_3": float(contribution_row[int(positive_idx[2])]),
                "top_positive_feature_4": feature_names[int(positive_idx[3])],
                "top_positive_log_odds_contribution_4": float(contribution_row[int(positive_idx[3])]),
                "top_positive_feature_5": feature_names[int(positive_idx[4])],
                "top_positive_log_odds_contribution_5": float(contribution_row[int(positive_idx[4])]),
            }
        )
    return pd.DataFrame(rows)


def high_value_false_positive_examples(
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    max_rows: int = 20,
) -> pd.DataFrame:
    score_array = np.asarray(scores, dtype=float)
    y_array = np.asarray(y_test).astype(int)
    flagged_positions = np.flatnonzero((score_array >= threshold) & (y_array == 0))
    flagged_legitimate = test_df.iloc[flagged_positions].copy()
    if flagged_legitimate.empty:
        return pd.DataFrame(
            columns=["row_id", "amount", "predicted_probability", "threshold", "time_seconds", "monitoring_use"]
        )
    flagged_legitimate["predicted_probability"] = score_array[flagged_positions]
    flagged_legitimate["threshold"] = float(threshold)
    flagged_legitimate["monitoring_use"] = "high-value false positive for segment-level friction monitoring"
    out = flagged_legitimate.sort_values("Amount", ascending=False).head(max_rows)
    return out.rename(columns={"Amount": "amount", "Time": "time_seconds"})[
        ["_row_id", "amount", "predicted_probability", "threshold", "time_seconds", "monitoring_use"]
    ].rename(columns={"_row_id": "row_id"})


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None, float_digits: int = 4) -> str:
    view = frame.copy()
    if columns:
        view = view[columns]
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{value:.{float_digits}f}")
    headers = list(view.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def write_report(
    report_path: Path,
    *,
    data_url: str,
    split: str,
    distribution: pd.DataFrame,
    metrics: pd.DataFrame,
    champion_name: str,
    champion_result: dict[str, Any],
    drivers: pd.DataFrame,
    baseline_metrics: dict[str, float],
    model_notes: list[str],
    elapsed_seconds: float,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    test_metrics = champion_result["test"]
    validation_metrics = champion_result["validation"]
    threshold = champion_result["threshold"]
    threshold_selection = champion_result["threshold_selection"]
    driver_cols = ["feature", "driver_value", "driver_type", "interpretation"]
    metric_cols = [
        "model",
        "split",
        "threshold",
        "precision",
        "recall",
        "f1",
        "f2",
        "average_precision",
        "roc_auc",
        "flagged_rate",
        "false_positive_rate",
        "missed_fraud_cost",
        "expected_cost",
        "cost_savings_vs_no_model",
    ]
    model_notes_md = "\n".join(f"- {note}" for note in model_notes)

    report = f"""# Fraud Detection Model Report

Generated: {datetime.now(timezone.utc).isoformat(timespec="seconds")}

## Dataset

Source: {data_url}

This project targets the classic credit-card fraud dataset: anonymized PCA transaction features (`V1` through `V28`), `Time`, `Amount`, and a binary `Class` label.

{markdown_table(distribution, float_digits=6)}

## Modeling Approach

- Split strategy: `{split}`. The default chronological split is closer to production monitoring than a random split because future transactions are held out from earlier training data.
- Feature engineering: `log1p(Amount)`, zero-amount indicator, and hour-of-day sine/cosine, alongside the anonymized PCA features. Raw time and day index are deliberately excluded from the model to avoid learning a split artifact from this short two-day dataset.
- Imbalance handling: the pipeline compares an unweighted logistic baseline, a class-weighted logistic model, and an optional boosted-tree challenger when scikit-learn is installed.
- Thresholding: the operating threshold is selected on the validation set with policy `{threshold_selection["threshold_policy"]}`. The default cost policy uses illustrative false-positive cost `{threshold_selection["false_positive_cost"]:.2f}` and amount-weighted missed-fraud cost `Amount + {threshold_selection["false_negative_handling_cost"]:.2f}`. The test set is used only after that threshold is chosen. In production, fraud operations, risk/finance, product, and data science should jointly own these assumptions.
- Primary metrics: precision, recall, F1, F2, average precision / PR-AUC, ROC-AUC, false positive rate, flagged rate, and expected operating cost. Accuracy is reported only to show why it is a poor fraud metric.

## Model Narrative

{model_notes_md}

## Accuracy Trap

An "always legitimate" classifier gets **{baseline_metrics["accuracy"]:.4f} accuracy** on the test split while catching **{baseline_metrics["recall"]:.4f}** of fraud. That is why this project optimizes precision/recall behavior instead of accuracy.

## Model Comparison

{markdown_table(metrics[metric_cols])}

## Champion

Champion: `{champion_name}`

- Validation precision: {validation_metrics["precision"]:.4f}
- Validation recall: {validation_metrics["recall"]:.4f}
- Validation expected cost: {validation_metrics["expected_cost"]:.2f}
- Test precision: {test_metrics["precision"]:.4f}
- Test recall: {test_metrics["recall"]:.4f}
- Test average precision / PR-AUC: {test_metrics["average_precision"]:.4f}
- Test expected cost: {test_metrics["expected_cost"]:.2f}
- Alert threshold: {threshold:.6f}
- Near-optimal validation cost band: thresholds {threshold_selection["near_optimal_threshold_min"]:.6f} to {threshold_selection["near_optimal_threshold_max"]:.6f} stay within {threshold_selection["near_optimal_cost_tolerance_pct"]:.0%} of selected validation cost, with precision {threshold_selection["near_optimal_precision_min"]:.4f}-{threshold_selection["near_optimal_precision_max"]:.4f} and recall {threshold_selection["near_optimal_recall_min"]:.4f}-{threshold_selection["near_optimal_recall_max"]:.4f}.

Cost alone does not fully determine the operating point when the cost curve is flat. The final threshold should be pinned with `--min-precision` or `--threshold-policy capacity` once fraud operations confirms the review budget and acceptable customer-friction level.

## What Drives Flags

For logistic models, driver values are standardized coefficients: positive values push a transaction toward a fraud flag and negative values push it away. For the boosted-tree challenger, driver values are validation PR-AUC drops from permutation importance. The `V*` fields are anonymized PCA components, so they are useful for model debugging but are not human business concepts like "merchant category" or "cardholder velocity."

{markdown_table(drivers.head(12)[driver_cols])}

For the selected linear operating model, local transaction-level `x_i * w_i` log-odds contributions are written to `interpretation/linear_log_odds_contributions_top_flags.csv`. These are not customer-facing reason codes; real reason codes require business features such as merchant category, cardholder velocity, device reputation, and chargeback history.

High-value false positives are written to `interpretation/high_value_false_positive_examples.csv`. These are useful monitoring cases because they turn false-positive-rate discussion into customer-friction and segment-risk review, not just a scalar metric.

## Artifacts

- `artifacts/fraud_model.json`: weights, preprocessing statistics, selected threshold, and metrics.
- `reports/metrics_summary.csv`: validation/test metrics for each model.
- `reports/precision_recall_curve.svg`: validation precision-recall curve for the champion.
- `interpretation/global_feature_drivers.csv`: global feature drivers, using signed standardized coefficients for logistic models or permutation PR-AUC drop for the boosted-tree model.
- `interpretation/linear_log_odds_contributions_top_flags.csv`: exact local log-odds contributions for the highest-risk test transactions scored by the selected linear operating model.
- `interpretation/high_value_false_positive_examples.csv`: flagged legitimate high-value transactions to inspect for customer-friction and segment-level false-positive monitoring.

Runtime: {elapsed_seconds:.1f} seconds.
"""
    report_path.write_text(report, encoding="utf-8")


def balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(int)
    positives = float(y.sum())
    negatives = float(len(y) - positives)
    if positives == 0.0 or negatives == 0.0:
        return np.ones_like(y, dtype=float)
    return np.where(y == 1, len(y) / (2.0 * positives), len(y) / (2.0 * negatives)).astype(float)


def fit_hist_gradient_boosting(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
) -> tuple[Any | None, str | None]:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except Exception as exc:
        return None, f"sklearn HistGradientBoostingClassifier unavailable: {exc}"

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.06,
        max_iter=180,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(x_train, y_train, sample_weight=balanced_sample_weight(y_train))
    return model, None


def train_and_evaluate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    val_positive_cost: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_positive_cost: np.ndarray,
    *,
    epochs: int,
    seed: int,
    threshold_policy: str,
    false_positive_cost: float,
    false_negative_handling_cost: float,
    max_alert_rate: float,
    min_precision: float | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], list[str]]:
    models: dict[str, Any] = {
        "unweighted_logistic": WeightedLogisticRegression(
            learning_rate=0.045,
            l2=1e-3,
            epochs=epochs,
            class_weight=None,
            seed=seed,
        ),
        "class_weighted_logistic": WeightedLogisticRegression(
            learning_rate=0.045,
            l2=1e-3,
            epochs=epochs,
            class_weight="balanced",
            seed=seed + 1,
        ),
    }
    notes: list[str] = [
        "Logistic models are kept as interpretable baselines; the optional boosted-tree challenger reflects a common fraud-team production pattern.",
        "HistGradientBoostingClassifier internal early stopping is disabled so the chronological validation split remains the only validation boundary.",
    ]

    results: list[dict[str, Any]] = []
    for name, model in models.items():
        model.fit(x_train, y_train, x_val, y_val)

    hgb_model, hgb_note = fit_hist_gradient_boosting(x_train, y_train, seed=seed + 2)
    if hgb_model is None:
        notes.append(hgb_note or "HistGradientBoostingClassifier unavailable.")
    else:
        models["hist_gradient_boosting"] = hgb_model

    for name, model in models.items():
        results.append(
            evaluate_model(
                name,
                model,
                x_val,
                y_val,
                val_positive_cost,
                x_test,
                y_test,
                test_positive_cost,
                threshold_policy=threshold_policy,
                false_positive_cost=false_positive_cost,
                false_negative_handling_cost=false_negative_handling_cost,
                max_alert_rate=max_alert_rate,
                min_precision=min_precision,
            )
        )

    if threshold_policy == "cost":
        validation_costs = [float(item["validation"]["expected_cost"]) for item in results]
        best_validation_cost = min(validation_costs)
        model_cost_tolerance = max(
            best_validation_cost * MODEL_SELECTION_COST_TOLERANCE_PCT,
            false_positive_cost * 5.0,
        )
        near_tied_results = [
            item
            for item in results
            if float(item["validation"]["expected_cost"]) <= best_validation_cost + model_cost_tolerance
        ]
        preferred_order = ["unweighted_logistic", "class_weighted_logistic", "hist_gradient_boosting"]
        champion_result = min(
            near_tied_results,
            key=lambda item: preferred_order.index(item["model_name"])
            if item["model_name"] in preferred_order
            else len(preferred_order),
        )
        argmin_validation_result = min(
            results,
            key=lambda item: (
                item["validation"]["expected_cost"],
                -item["validation"]["average_precision"],
                -item["validation"]["recall"],
            ),
        )
        tied_names = ", ".join(item["model_name"] for item in near_tied_results)
        notes.append(
            f"Validation expected-cost spread is treated as a model-selection tie when models are within {MODEL_SELECTION_COST_TOLERANCE_PCT:.0%} of the best validation cost. Near-tied models here: {tied_names}."
        )
        if champion_result["model_name"] != argmin_validation_result["model_name"]:
            notes.append(
                f"`{argmin_validation_result['model_name']}` has the numerical validation-cost minimum, but `{champion_result['model_name']}` is selected as the operating model because validation cannot separate the near-tied candidates and the baseline is easier to audit."
            )
    else:
        champion_result = max(
            results,
            key=lambda item: (
                item["validation"]["f2"],
                item["validation"]["average_precision"],
                item["validation"]["precision"],
            ),
        )
    cheapest_test_result = min(results, key=lambda item: item["test"]["expected_cost"])
    notes.append(
        f"Model selection uses validation expected cost over {int(y_val.sum())} validation frauds and {int(y_test.sum())} test frauds; close rankings should be treated as noisy rather than proof of model superiority."
    )
    notes.append(
        "Amount-weighted missed-fraud cost is economically better than a flat false-negative penalty, but it increases variance because a few high-value frauds can move the validation cost ranking."
    )
    if cheapest_test_result["model_name"] != champion_result["model_name"]:
        notes.append(
            f"Validation selected `{champion_result['model_name']}`, but `{cheapest_test_result['model_name']}` has the lowest illustrative test cost. This is reported as rank instability on a small fraud count, not hidden."
        )
    return models, results, champion_result, notes


def run_pipeline(
    *,
    raw_path: Path,
    output_dir: Path,
    data_url: str = DATA_URL,
    download: bool = True,
    force_download: bool = False,
    split: str = "chronological",
    epochs: int = 80,
    seed: int = 42,
    sample_rows: int | None = None,
    threshold_policy: str = "cost",
    false_positive_cost: float = DEFAULT_FALSE_POSITIVE_COST,
    false_negative_handling_cost: float = DEFAULT_FALSE_NEGATIVE_HANDLING_COST,
    max_alert_rate: float = DEFAULT_MAX_ALERT_RATE,
    min_precision: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    raw_path = raw_path.resolve()
    output_dir = output_dir.resolve()
    reports_dir = output_dir / "reports"
    artifacts_dir = output_dir / "artifacts"
    interpretation_dir = output_dir / "interpretation"
    for directory in (reports_dir, artifacts_dir, interpretation_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if download:
        download_dataset(raw_path, data_url=data_url, force=force_download)
    if not raw_path.exists():
        raise FileNotFoundError(f"Dataset not found at {raw_path}. Run without --no-download or place the CSV there.")

    df = load_dataset(raw_path)
    if sample_rows is not None and sample_rows < len(df):
        df = df.sample(n=sample_rows, random_state=seed).sort_values("Time", kind="mergesort").reset_index(drop=True)
    df = engineer_features(df)
    train_df, val_df, test_df = split_dataset(df, split=split, seed=seed)

    distribution = pd.DataFrame(
        [
            class_distribution(df, "all"),
            class_distribution(train_df, "train"),
            class_distribution(val_df, "validation"),
            class_distribution(test_df, "test"),
        ]
    )

    for split_name, split_frame in (("train", train_df), ("validation", val_df), ("test", test_df)):
        if split_frame[TARGET].nunique() < 2:
            raise ValueError(
                f"The {split_name} split has only one class. Use --split stratified or provide more data."
            )

    scaler = Standardizer.fit(train_df, FEATURE_NAMES)
    x_train = scaler.transform(train_df)
    x_val = scaler.transform(val_df)
    x_test = scaler.transform(test_df)
    y_train = train_df[TARGET].to_numpy(dtype=int)
    y_val = val_df[TARGET].to_numpy(dtype=int)
    y_test = test_df[TARGET].to_numpy(dtype=int)
    val_positive_cost = fraud_loss_amount(val_df, false_negative_handling_cost)
    test_positive_cost = fraud_loss_amount(test_df, false_negative_handling_cost)

    models, results, champion_result, model_notes = train_and_evaluate(
        x_train,
        y_train,
        x_val,
        y_val,
        val_positive_cost,
        x_test,
        y_test,
        test_positive_cost,
        epochs=epochs,
        seed=seed,
        threshold_policy=threshold_policy,
        false_positive_cost=false_positive_cost,
        false_negative_handling_cost=false_negative_handling_cost,
        max_alert_rate=max_alert_rate,
        min_precision=min_precision,
    )

    metrics = flatten_metrics(results)
    champion_name = champion_result["model_name"]
    champion_model = models[champion_name]
    drivers = feature_drivers(
        champion_model,
        FEATURE_NAMES,
        x_reference=x_val,
        y_reference=y_val,
        seed=seed,
    )
    result_by_name = {result["model_name"]: result for result in results}
    if has_linear_weights(champion_model):
        contribution_model_name = champion_name
    else:
        contribution_model_name = "class_weighted_logistic" if "class_weighted_logistic" in models else champion_name
    contribution_model = models[contribution_model_name]
    contribution_result = result_by_name[contribution_model_name]
    contributions = linear_log_odds_contributions(
        test_df,
        x_test,
        y_test,
        contribution_result["test_scores"],
        contribution_model,
        FEATURE_NAMES,
        contribution_result["threshold"],
    )
    high_value_fps = high_value_false_positive_examples(
        test_df,
        y_test,
        champion_result["test_scores"],
        champion_result["threshold"],
    )

    baseline = metrics_at_threshold(
        y_test,
        np.zeros_like(y_test, dtype=float),
        threshold=1.0,
        false_positive_cost=false_positive_cost,
        positive_cost=test_positive_cost,
    )
    elapsed = time.perf_counter() - started

    distribution.to_csv(reports_dir / "class_distribution.csv", index=False)
    metrics.to_csv(reports_dir / "metrics_summary.csv", index=False)
    drivers.to_csv(interpretation_dir / "global_feature_drivers.csv", index=False)
    contributions.to_csv(interpretation_dir / "linear_log_odds_contributions_top_flags.csv", index=False)
    high_value_fps.to_csv(interpretation_dir / "high_value_false_positive_examples.csv", index=False)
    history_frames = []
    for result in results:
        history = pd.DataFrame(result["training_history"])
        if not history.empty:
            history.insert(0, "model", result["model_name"])
            history_frames.append(history)
    if history_frames:
        pd.concat(history_frames, ignore_index=True).to_csv(reports_dir / "training_history.csv", index=False)
    else:
        pd.DataFrame([{"note": "Selected model does not expose iterative training history."}]).to_csv(
            reports_dir / "training_history.csv",
            index=False,
        )
    champion_result["validation_pr_curve"].to_csv(reports_dir / "validation_precision_recall_curve.csv", index=False)

    write_precision_recall_svg(
        champion_result["validation_pr_curve"],
        reports_dir / "precision_recall_curve.svg",
        f"Validation Precision-Recall: {champion_name}",
    )
    write_feature_driver_svg(drivers, reports_dir / "feature_drivers.svg", f"Top Drivers: {champion_name}")

    model_artifact = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_url": data_url,
        "model_name": champion_name,
        "feature_names": FEATURE_NAMES,
        "standardizer": {
            "mean": scaler.mean,
            "scale": scaler.scale,
        },
        "weights": getattr(champion_model, "weights", None),
        "intercept": getattr(champion_model, "intercept", None),
        "threshold": champion_result["threshold"],
        "threshold_strategy": f"validation_{threshold_policy}",
        "threshold_selection": champion_result["threshold_selection"],
        "model_selection": {
            "selection_metric": "validation_expected_cost",
            "near_tie_tolerance_pct": MODEL_SELECTION_COST_TOLERANCE_PCT,
            "near_tie_policy": "prefer_interpretable_baseline_then_class_weighted_then_boosted_tree",
        },
        "metrics": {
            "validation": champion_result["validation"],
            "test": champion_result["test"],
            "always_legitimate_baseline": baseline,
        },
    }
    (artifacts_dir / "fraud_model.json").write_text(json.dumps(to_builtin(model_artifact), indent=2), encoding="utf-8")
    champion_joblib_path = artifacts_dir / "champion_model.joblib"
    if not has_linear_weights(champion_model):
        try:
            import joblib

            joblib.dump(champion_model, champion_joblib_path)
        except Exception as exc:
            model_notes.append(f"Champion model serialization skipped: {exc}")
    elif champion_joblib_path.exists():
        champion_joblib_path.unlink()

    write_report(
        reports_dir / "model_report.md",
        data_url=data_url,
        split=split,
        distribution=distribution,
        metrics=metrics,
        champion_name=champion_name,
        champion_result=champion_result,
        drivers=drivers,
        baseline_metrics=baseline,
        model_notes=model_notes,
        elapsed_seconds=elapsed,
    )

    return {
        "output_dir": output_dir,
        "raw_path": raw_path,
        "champion_name": champion_name,
        "metrics": metrics,
        "champion_result": champion_result,
        "report_path": reports_dir / "model_report.md",
        "artifact_path": artifacts_dir / "fraud_model.json",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an imbalance-aware credit card fraud detector.")
    parser.add_argument("--raw-path", type=Path, default=PROJECT_ROOT / "data" / "raw" / "creditcard.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--data-url", default=DATA_URL)
    parser.add_argument("--no-download", action="store_true", help="Use an existing CSV at --raw-path.")
    parser.add_argument("--force-download", action="store_true", help="Redownload even if --raw-path exists.")
    parser.add_argument("--split", choices=["chronological", "stratified"], default="chronological")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rows", type=int, default=None, help="Optional row sample for quick experiments.")
    parser.add_argument("--threshold-policy", choices=["cost", "capacity", "f2", "f1"], default="cost")
    parser.add_argument("--false-positive-cost", type=float, default=DEFAULT_FALSE_POSITIVE_COST)
    parser.add_argument(
        "--false-negative-handling-cost",
        "--false-negative-cost",
        dest="false_negative_handling_cost",
        type=float,
        default=DEFAULT_FALSE_NEGATIVE_HANDLING_COST,
        help="Handling/investigation cost added to Amount for each missed fraud.",
    )
    parser.add_argument("--max-alert-rate", type=float, default=DEFAULT_MAX_ALERT_RATE)
    parser.add_argument("--min-precision", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = run_pipeline(
            raw_path=args.raw_path,
            output_dir=args.output_dir,
            data_url=args.data_url,
            download=not args.no_download,
            force_download=args.force_download,
            split=args.split,
            epochs=args.epochs,
            seed=args.seed,
            sample_rows=args.sample_rows,
            threshold_policy=args.threshold_policy,
            false_positive_cost=args.false_positive_cost,
            false_negative_handling_cost=args.false_negative_handling_cost,
            max_alert_rate=args.max_alert_rate,
            min_precision=args.min_precision,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    champion = result["champion_result"]
    test_metrics = champion["test"]
    print(f"Champion: {result['champion_name']}")
    print(f"Test precision: {test_metrics['precision']:.4f}")
    print(f"Test recall: {test_metrics['recall']:.4f}")
    print(f"Test average precision: {test_metrics['average_precision']:.4f}")
    print(f"Report: {result['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
