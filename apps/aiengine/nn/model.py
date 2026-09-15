"""The Campy AI model container: a sequential stack with train / predict / save."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .config import as_array, default_dtype
from .layers import LAYER_REGISTRY, Layer
from .losses import Loss, build_loss
from .metrics import evaluate_metrics
from .optim import Optimizer, Schedule, build_optimizer, build_schedule

SCHEMA_VERSION = 1


class Sequential:
    """An ordered stack of :class:`~apps.aiengine.nn.layers.Layer` objects.

    The container owns the full lifecycle we need in production:
    forward/backward, mini-batch training with validation, early stopping,
    metric history, and a self-describing ``.npz`` checkpoint that carries the
    architecture alongside the weights — so a deployed model version can always
    be rebuilt from its file alone.
    """

    def __init__(self, layers: Iterable[Layer] | None = None, name: str = "campynet", meta: dict | None = None):
        self.layers: list[Layer] = list(layers or [])
        self.name = name
        self.meta: dict = dict(meta or {})
        self.history: list[dict] = []

    # -- composition -------------------------------------------------------
    def add(self, layer: Layer) -> "Sequential":
        self.layers.append(layer)
        return self

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    @property
    def trainable_layers(self) -> list[Layer]:
        return [layer for layer in self.layers if layer.params]

    @property
    def param_count(self) -> int:
        return int(sum(layer.param_count for layer in self.layers))

    # -- inference ---------------------------------------------------------
    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        out = as_array(x)
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def __call__(self, x, training=False):
        return self.forward(x, training=training)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        grad = dout
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def predict(self, x: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Batched inference — never allocates more than ``batch_size`` at once."""
        x = as_array(x)
        if x.shape[0] <= batch_size:
            return self.forward(x, training=False)
        chunks = [
            self.forward(x[i : i + batch_size], training=False)
            for i in range(0, x.shape[0], batch_size)
        ]
        return np.concatenate(chunks, axis=0)

    def predict_proba(self, x: np.ndarray, batch_size: int = 64) -> np.ndarray:
        from .functional import softmax

        logits = self.predict(x, batch_size=batch_size)
        return softmax(logits, axis=-1) if logits.ndim == 2 and logits.shape[1] > 1 else logits

    # -- training ----------------------------------------------------------
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        *,
        loss: Loss | str = "cross_entropy",
        optimizer: Optimizer | str = "adam",
        epochs: int = 20,
        batch_size: int = 32,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        schedule: Schedule | str | None = None,
        task: str = "classification",
        shuffle: bool = True,
        seed: int | None = 1337,
        early_stopping_patience: int | None = None,
        on_epoch_end: Callable[[dict], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        verbose: bool = False,
    ) -> list[dict]:
        """Mini-batch training loop.

        ``on_epoch_end`` receives the per-epoch metric dict — the training
        service uses it to stream live progress into the dashboard.
        ``should_stop`` is polled each epoch so a user can cancel a job.
        """
        if isinstance(loss, str):
            loss = build_loss(loss)
        if isinstance(optimizer, str):
            optimizer = build_optimizer(optimizer)
        if isinstance(schedule, str):
            schedule = build_schedule(schedule, total_epochs=epochs)

        x_train = as_array(x_train)
        y_train = np.asarray(y_train)
        n = x_train.shape[0]
        if n == 0:
            raise ValueError("Cannot train on an empty dataset.")
        batch_size = max(1, min(int(batch_size), n))
        rng = np.random.default_rng(seed)

        best_score = -np.inf
        best_weights: dict | None = None
        patience_left = early_stopping_patience
        self.history = []

        for epoch in range(int(epochs)):
            if should_stop is not None and should_stop():
                break

            if schedule is not None:
                optimizer.lr = float(schedule(epoch, optimizer.base_lr))

            order = rng.permutation(n) if shuffle else np.arange(n)
            epoch_started = time.time()
            running_loss = 0.0
            batches = 0

            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                xb, yb = x_train[idx], y_train[idx]

                predictions = self.forward(xb, training=True)
                batch_loss, dout = loss(predictions, yb)
                optimizer.zero_grad(self.layers)
                self.backward(dout)
                optimizer.step(self.trainable_layers)

                running_loss += float(batch_loss)
                batches += 1

            record = {
                "epoch": epoch + 1,
                "loss": round(running_loss / max(batches, 1), 6),
                "lr": round(float(optimizer.lr), 8),
                "seconds": round(time.time() - epoch_started, 3),
            }

            train_pred = self.predict(x_train, batch_size=max(batch_size, 64))
            record.update(
                {f"train_{k}": v for k, v in evaluate_metrics(train_pred, y_train, task=task).items()}
            )

            if validation_data is not None:
                x_val, y_val = validation_data
                if len(x_val):
                    val_pred = self.predict(as_array(x_val), batch_size=max(batch_size, 64))
                    val_loss, _ = loss(val_pred, y_val)
                    record["val_loss"] = round(float(val_loss), 6)
                    record.update(
                        {f"val_{k}": v for k, v in evaluate_metrics(val_pred, y_val, task=task).items()}
                    )

            self.history.append(record)
            if verbose:
                print(f"[{self.name}] {record}")
            if on_epoch_end is not None:
                on_epoch_end(record)

            # -- early stopping on the best available score ------------------
            score = record.get("val_accuracy", record.get("val_f1"))
            if score is None:
                score = -record.get("val_loss", record["loss"])
            if score > best_score:
                best_score = score
                best_weights = self.get_weights()
                patience_left = early_stopping_patience
            elif early_stopping_patience is not None:
                patience_left -= 1
                if patience_left <= 0:
                    if verbose:
                        print(f"[{self.name}] early stopping at epoch {epoch + 1}")
                    break

        if best_weights is not None:
            self.set_weights(best_weights)
        return self.history

    def evaluate(self, x, y, *, loss: Loss | str = "cross_entropy", task: str = "classification") -> dict:
        if isinstance(loss, str):
            loss = build_loss(loss)
        predictions = self.predict(as_array(x))
        value, _ = loss(predictions, y)
        result = {"loss": round(float(value), 6)}
        result.update(evaluate_metrics(predictions, y, task=task))
        return result

    # -- weights -----------------------------------------------------------
    def get_weights(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for index, layer in enumerate(self.layers):
            for key, value in layer.params.items():
                out[f"{index}.p.{key}"] = value.copy()
            for key, value in layer.buffers.items():
                out[f"{index}.b.{key}"] = value.copy()
        return out

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        for index, layer in enumerate(self.layers):
            for key in layer.params:
                name = f"{index}.p.{key}"
                if name in weights:
                    layer.params[key] = np.asarray(weights[name], dtype=default_dtype()).copy()
            for key in layer.buffers:
                name = f"{index}.b.{key}"
                if name in weights:
                    layer.buffers[key] = np.asarray(weights[name], dtype=default_dtype()).copy()

    # -- persistence -------------------------------------------------------
    def architecture(self) -> list[dict]:
        return [{"type": layer.__class__.__name__, "config": layer.config()} for layer in self.layers]

    def save(self, path: str | Path) -> Path:
        """Write a self-contained ``.npz`` checkpoint (architecture + weights)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.get_weights()
        payload["__meta__"] = np.frombuffer(
            json.dumps(
                {
                    "schema": SCHEMA_VERSION,
                    "name": self.name,
                    "architecture": self.architecture(),
                    "meta": self.meta,
                    "history": self.history[-50:],
                    "param_count": self.param_count,
                }
            ).encode("utf-8"),
            dtype=np.uint8,
        )
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Sequential":
        path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(bytes(archive["__meta__"]).decode("utf-8"))
            model = cls.from_architecture(meta["architecture"], name=meta.get("name", "campynet"))
            model.meta = meta.get("meta", {})
            model.history = meta.get("history", [])
            model.set_weights({k: archive[k] for k in archive.files if k != "__meta__"})
        return model

    @classmethod
    def from_architecture(cls, architecture: list[dict], name: str = "campynet") -> "Sequential":
        layers: list[Layer] = []
        for spec in architecture:
            layer_cls = LAYER_REGISTRY.get(spec["type"])
            if layer_cls is None:
                raise ValueError(f"Unknown layer type '{spec['type']}' in checkpoint.")
            config = dict(spec.get("config") or {})
            if "shape" in config:
                config["shape"] = tuple(config["shape"])
            layers.append(layer_cls(**config))
        return cls(layers, name=name)

    # -- reporting ---------------------------------------------------------
    def summary(self, input_shape: tuple[int, ...] | None = None) -> str:
        lines = [f"Campy model '{self.name}'", "-" * 62]
        shape = tuple(input_shape) if input_shape else None
        for layer in self.layers:
            if shape is not None:
                try:
                    shape = layer.output_shape(shape)
                except Exception:  # noqa: BLE001 - shape tracking is best-effort
                    shape = None
            shape_text = f"→ {shape}" if shape else ""
            lines.append(f"{layer.describe():<40} {layer.param_count:>9,}  {shape_text}")
        lines.append("-" * 62)
        lines.append(f"{'Total parameters':<40} {self.param_count:>9,}")
        return "\n".join(lines)
