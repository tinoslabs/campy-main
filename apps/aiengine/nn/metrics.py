"""Evaluation metrics for classification, regression and detection."""
from __future__ import annotations

import numpy as np


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    pred = np.argmax(predictions, axis=-1)
    truth = np.asarray(targets)
    if truth.ndim == 2:
        truth = np.argmax(truth, axis=-1)
    return float(np.mean(pred == truth.ravel()))


def confusion_matrix(predictions: np.ndarray, targets: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    pred = np.argmax(predictions, axis=-1).ravel()
    truth = np.asarray(targets)
    if truth.ndim == 2:
        truth = np.argmax(truth, axis=-1)
    truth = truth.ravel()
    k = int(num_classes or max(pred.max(initial=0), truth.max(initial=0)) + 1)
    matrix = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(truth, pred):
        if 0 <= t < k and 0 <= p < k:
            matrix[t, p] += 1
    return matrix


def precision_recall_f1(predictions: np.ndarray, targets: np.ndarray, average: str = "macro") -> dict:
    matrix = confusion_matrix(predictions, targets)
    k = matrix.shape[0]
    precisions, recalls, f1s, supports = [], [], [], []
    for i in range(k):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(matrix[i, :].sum())

    if average == "weighted" and sum(supports):
        weights = np.asarray(supports, dtype=np.float64) / sum(supports)
        return {
            "precision": float(np.dot(precisions, weights)),
            "recall": float(np.dot(recalls, weights)),
            "f1": float(np.dot(f1s, weights)),
        }
    return {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
    }


def per_class_report(predictions, targets, class_names: list[str] | None = None) -> list[dict]:
    matrix = confusion_matrix(predictions, targets, num_classes=len(class_names) if class_names else None)
    rows = []
    for i in range(matrix.shape[0]):
        tp = int(matrix[i, i])
        fp = int(matrix[:, i].sum() - tp)
        fn = int(matrix[i, :].sum() - tp)
        support = int(matrix[i, :].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        rows.append(
            {
                "class": class_names[i] if class_names and i < len(class_names) else str(i),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0,
                "support": support,
            }
        )
    return rows


def mean_absolute_error(predictions, targets) -> float:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64).reshape(predictions.shape)
    return float(np.mean(np.abs(predictions - targets)))


def root_mean_squared_error(predictions, targets) -> float:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64).reshape(predictions.shape)
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def r_squared(predictions, targets) -> float:
    predictions = np.asarray(predictions, dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.float64).ravel()
    ss_res = float(np.sum((targets - predictions) ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else 0.0


def embedding_retrieval_accuracy(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Rank-1 accuracy: does the nearest *other* embedding share the identity?"""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels).ravel()
    if embeddings.shape[0] < 2:
        return 0.0
    sq = np.sum(embeddings ** 2, axis=1)
    dist = sq[:, None] - 2 * embeddings @ embeddings.T + sq[None, :]
    np.fill_diagonal(dist, np.inf)
    nearest = np.argmin(dist, axis=1)
    return float(np.mean(labels[nearest] == labels))


def evaluate_metrics(predictions: np.ndarray, targets, task: str = "classification") -> dict:
    """Dispatch to the metric set that matches the head being trained."""
    predictions = np.asarray(predictions)
    try:
        if task == "classification":
            result = {"accuracy": round(accuracy(predictions, targets), 4)}
            result.update({k: round(v, 4) for k, v in precision_recall_f1(predictions, targets).items()})
            return result
        if task == "embedding":
            return {"rank1": round(embedding_retrieval_accuracy(predictions, targets), 4)}
        if task == "regression":
            return {
                "mae": round(mean_absolute_error(predictions, targets), 4),
                "rmse": round(root_mean_squared_error(predictions, targets), 4),
                "r2": round(r_squared(predictions, targets), 4),
            }
        if task == "multilabel":
            probs = 1.0 / (1.0 + np.exp(-predictions))
            truth = np.asarray(targets, dtype=np.float32).reshape(probs.shape)
            pred_binary = (probs >= 0.5).astype(np.float32)
            tp = float(np.sum((pred_binary == 1) & (truth == 1)))
            fp = float(np.sum((pred_binary == 1) & (truth == 0)))
            fn = float(np.sum((pred_binary == 0) & (truth == 1)))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            return {
                "accuracy": round(float(np.mean(pred_binary == truth)), 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0,
            }
    except Exception:  # noqa: BLE001 - metrics must never break a training run
        return {}
    return {}
