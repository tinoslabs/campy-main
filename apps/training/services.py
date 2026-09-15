"""The training service: turn a Dataset into a deployable ModelVersion."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from django.conf import settings
from django.utils import timezone

from apps.aiengine import architectures
from apps.aiengine.nn.losses import build_loss
from apps.aiengine.nn.metrics import per_class_report
from apps.aiengine.nn.optim import build_optimizer, build_schedule
from apps.aiengine.vision.image import normalise, resize_bilinear, to_chw

logger = logging.getLogger("campy.training")


# ---------------------------------------------------------------------------
# Data loading & augmentation
# ---------------------------------------------------------------------------
def augment_batch(images: np.ndarray, options: dict, rng: np.random.Generator) -> np.ndarray:
    """Apply the dataset's augmentation policy to an NCHW batch.

    Augmentation matters more here than in a research setting: customers train
    on a few hundred frames from a handful of cameras, so without it the model
    memorises one camera's lighting and fails on the next one.
    """
    output = images.copy()
    options = options or {}

    if options.get("flip"):
        flip = rng.random(output.shape[0]) < 0.5
        output[flip] = output[flip][:, :, :, ::-1]

    brightness = float(options.get("brightness", 0) or 0)
    if brightness > 0:
        scale = 1.0 + rng.uniform(-brightness, brightness, (output.shape[0], 1, 1, 1))
        output = output * scale

    contrast = float(options.get("contrast", 0) or 0)
    if contrast > 0:
        factor = 1.0 + rng.uniform(-contrast, contrast, (output.shape[0], 1, 1, 1))
        mean = output.mean(axis=(1, 2, 3), keepdims=True)
        output = (output - mean) * factor + mean

    noise = float(options.get("noise", 0) or 0)
    if noise > 0:
        output = output + rng.normal(0, noise, output.shape)

    shift = float(options.get("shift", 0) or 0)
    if shift > 0:
        max_x = max(int(output.shape[3] * shift), 1)
        max_y = max(int(output.shape[2] * shift), 1)
        for i in range(output.shape[0]):
            dx = int(rng.integers(-max_x, max_x + 1))
            dy = int(rng.integers(-max_y, max_y + 1))
            output[i] = np.roll(np.roll(output[i], dx, axis=2), dy, axis=1)

    return np.clip(output, -4.0, 4.0).astype(np.float32)


def load_dataset(dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Materialise a dataset into ``(x_train, y_train, x_val, y_val)``."""
    from .models import DatasetSample

    samples = list(
        DatasetSample.objects.filter(dataset=dataset).exclude(label="").order_by("pk")
    )
    if not samples:
        raise ValueError("This dataset has no labelled samples yet.")

    classes = dataset.classes or []
    size = int(dataset.image_size or 96)
    features, labels = [], []

    for sample in samples:
        index = classes.index(sample.label) if sample.label in classes else -1
        if index < 0:
            continue

        if sample.features:
            vector = np.asarray(sample.features, dtype=np.float32).ravel()
            features.append(vector)
        elif sample.image:
            try:
                from PIL import Image

                sample.image.open("rb")
                with Image.open(sample.image) as handle:
                    array = np.asarray(handle.convert("RGB"), dtype=np.float32)
            except Exception:  # noqa: BLE001 - skip an unreadable sample
                logger.warning("Could not read sample %s", sample.pk)
                continue
            resized = resize_bilinear(array, size, size)
            features.append(to_chw(normalise(resized)))
        else:
            continue
        labels.append(index)

    if not features:
        raise ValueError("None of the labelled samples could be loaded.")

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    # Stratified split so every class appears in validation — otherwise the
    # validation score is meaningless for the rarest and most important class.
    rng = np.random.default_rng(1337)
    validation_fraction = float(dataset.validation_split or 0.2)
    train_idx, val_idx = [], []
    for class_index in np.unique(y):
        members = np.where(y == class_index)[0]
        rng.shuffle(members)
        cut = max(int(round(len(members) * validation_fraction)), 1 if len(members) > 2 else 0)
        val_idx.extend(members[:cut].tolist())
        train_idx.extend(members[cut:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_model(job, dataset, input_shape):
    family = job.architecture
    channels = input_shape[0] if len(input_shape) == 3 else 3
    size = input_shape[1] if len(input_shape) == 3 else int(dataset.image_size or 96)

    if family == "campyface":
        return architectures.build_embedding_network(
            embedding_dim=int((job.dataset.augmentation or {}).get("embedding_dim", 128)),
            in_channels=channels, width=job.width, depth=job.depth,
            seed=job.seed, input_size=size,
        )
    if family == "campydet":
        return architectures.build_detector(
            num_classes=dataset.class_count, in_channels=channels,
            width=job.width, seed=job.seed, input_size=size,
        )
    if family == "campydense":
        return architectures.build_density_network(
            in_channels=channels, width=job.width, seed=job.seed, input_size=size
        )
    if family == "campyseq":
        return architectures.build_sequence_classifier(
            feature_dim=int(np.prod(input_shape)), num_classes=dataset.class_count,
            dropout=job.dropout, seed=job.seed,
        )
    return architectures.build_classifier(
        num_classes=dataset.class_count, in_channels=channels, width=job.width,
        depth=job.depth, dropout=job.dropout, seed=job.seed, input_size=size,
    )


TASK_FOR_FAMILY = {
    "campynet": "classification",
    "campyface": "embedding",
    "campyseq": "classification",
    "campydense": "regression",
    "campydet": "classification",
}

DEFAULT_LOSS_FOR_FAMILY = {
    "campynet": "cross_entropy",
    "campyface": "triplet",
    "campyseq": "cross_entropy",
    "campydense": "mse",
    "campydet": "cross_entropy",
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_training_job(job) -> "ModelVersion | None":  # noqa: F821
    """Execute a training job end to end, streaming progress into the DB."""
    from .models import TrainingJob

    job.status = TrainingJob.Status.RUNNING
    job.started_at = timezone.now()
    job.progress = 0.0
    job.history = []
    job.error_message = ""
    job.append_log(f"Starting '{job.name}' — {job.architecture} on '{job.dataset.name}'")
    job.save(
        update_fields=[
            "status", "started_at", "progress", "history", "error_message", "log", "updated_at",
        ]
    )

    started = time.time()
    try:
        dataset = job.dataset
        x_train, y_train, x_val, y_val = load_dataset(dataset)
        job.append_log(
            f"Loaded {len(x_train)} training and {len(x_val)} validation samples "
            f"across {dataset.class_count} classes"
        )
        if dataset.is_imbalanced:
            job.append_log(
                "WARNING: classes are imbalanced by more than 4:1 — "
                "accuracy will overstate performance on the rare classes"
            )

        model = build_model(job, dataset, x_train.shape[1:])
        job.append_log(f"Built {job.architecture} with {model.param_count:,} parameters")

        family = job.architecture
        task = TASK_FOR_FAMILY.get(family, "classification")
        loss_name = job.loss_function or DEFAULT_LOSS_FOR_FAMILY.get(family, "cross_entropy")

        # Class weighting is what stops a rare-but-critical class (fire, weapon)
        # from being ignored by a model that scores well by predicting "normal".
        loss_kwargs = {}
        if loss_name == "cross_entropy" and dataset.class_count:
            counts = np.bincount(y_train, minlength=dataset.class_count).astype(np.float64)
            if counts.min() > 0 and counts.max() / counts.min() > 1.5:
                weights = counts.sum() / (len(counts) * counts)
                loss_kwargs["class_weights"] = weights.astype(np.float32)
                job.append_log(
                    "Applied inverse-frequency class weights: "
                    + ", ".join(f"{c}={w:.2f}" for c, w in zip(dataset.classes, weights))
                )

        loss = build_loss(loss_name, **loss_kwargs)
        optimizer = build_optimizer(job.optimizer, lr=job.learning_rate)
        schedule = build_schedule(
            job.schedule, total_epochs=job.epochs, warmup_epochs=max(job.epochs // 10, 1)
        )
        augmentation = dataset.augmentation or {}
        rng = np.random.default_rng(job.seed)

        def on_epoch_end(record: dict) -> None:
            job.history = (job.history or []) + [record]
            job.current_epoch = record["epoch"]
            job.progress = round(record["epoch"] / max(job.epochs, 1) * 100, 1)
            summary = ", ".join(
                f"{k}={v}" for k, v in record.items() if k in ("loss", "val_loss", "train_accuracy", "val_accuracy", "val_rank1")
            )
            job.append_log(f"epoch {record['epoch']}/{job.epochs} — {summary}")
            job.save(update_fields=["history", "current_epoch", "progress", "log", "updated_at"])

        def should_stop() -> bool:
            job.refresh_from_db(fields=["cancel_requested"])
            return job.cancel_requested

        # Augmentation is applied once up front rather than per batch: our
        # datasets are small enough to hold in memory, and this keeps the
        # training loop itself allocation-free.
        if augmentation and x_train.ndim == 4:
            extra_x = augment_batch(x_train, augmentation, rng)
            x_train = np.concatenate([x_train, extra_x], axis=0)
            y_train = np.concatenate([y_train, y_train], axis=0)
            job.append_log(f"Augmentation expanded the training set to {len(x_train)} samples")

        model.fit(
            x_train, y_train,
            loss=loss, optimizer=optimizer, schedule=schedule,
            epochs=job.epochs, batch_size=job.batch_size,
            validation_data=(x_val, y_val) if len(x_val) else None,
            task=task, seed=job.seed,
            early_stopping_patience=job.early_stopping_patience or None,
            on_epoch_end=on_epoch_end,
            should_stop=should_stop,
        )

        if job.cancel_requested:
            job.status = TrainingJob.Status.CANCELLED
            job.append_log("Cancelled by user")
            job.finished_at = timezone.now()
            job.duration_seconds = time.time() - started
            job.save(
                update_fields=["status", "log", "finished_at", "duration_seconds", "updated_at"]
            )
            return None

        # -- evaluate ------------------------------------------------------
        evaluation = {}
        if len(x_val):
            evaluation = model.evaluate(x_val, y_val, loss=loss, task=task)
            if task == "classification":
                evaluation["per_class"] = per_class_report(
                    model.predict(x_val), y_val, dataset.classes
                )
        job.metrics = evaluation
        job.append_log(f"Validation: {evaluation}")

        # -- measure inference cost ---------------------------------------
        probe = x_val[:1] if len(x_val) else x_train[:1]
        model.predict(probe)                       # warm up
        timer = time.perf_counter()
        for _ in range(10):
            model.predict(probe)
        inference_ms = (time.perf_counter() - timer) / 10 * 1000

        # -- persist -------------------------------------------------------
        model.meta.update(
            {
                "classes": dataset.classes,
                "input_size": int(dataset.image_size or 96),
                "dataset": dataset.name,
                "job": str(job.uid),
                "trained_at": timezone.now().isoformat(),
            }
        )
        version = _save_version(job, model, evaluation, inference_ms)

        job.status = TrainingJob.Status.SUCCEEDED
        job.progress = 100.0
        job.append_log(f"Saved model version {version.version} ({version.size_label})")
    except Exception as exc:  # noqa: BLE001 - failures belong in the job record
        logger.exception("Training job %s failed", job.pk)
        job.status = TrainingJob.Status.FAILED
        job.error_message = str(exc)[:2000]
        job.append_log(f"FAILED: {exc}")
        version = None

    job.finished_at = timezone.now()
    job.duration_seconds = round(time.time() - started, 2)
    job.save(
        update_fields=[
            "status", "progress", "metrics", "error_message", "log",
            "finished_at", "duration_seconds", "updated_at",
        ]
    )
    return version


def _save_version(job, model, evaluation: dict, inference_ms: float):
    from .models import ModelVersion

    dataset = job.dataset
    root = Path(settings.CAMPY["MODEL_ROOT"]) / str(job.organization_id)
    root.mkdir(parents=True, exist_ok=True)

    previous = ModelVersion.objects.filter(
        organization=job.organization, slot=job.target_slot or job.architecture
    ).count()
    version_label = f"1.{previous}.0"
    path = root / f"{job.architecture}-{job.uid}.npz"
    model.save(path)

    return ModelVersion.objects.create(
        organization=job.organization,
        name=job.name,
        version=version_label,
        slot=job.target_slot or job.architecture,
        architecture=job.architecture,
        task=TASK_FOR_FAMILY.get(job.architecture, "classification"),
        job=job,
        dataset=dataset,
        weights_path=str(path),
        file_size_bytes=path.stat().st_size if path.exists() else 0,
        parameter_count=model.param_count,
        classes=dataset.classes,
        input_size=int(dataset.image_size or 96),
        meta=model.meta,
        accuracy=float(evaluation.get("accuracy", evaluation.get("rank1", 0)) or 0),
        precision=float(evaluation.get("precision", 0) or 0),
        recall=float(evaluation.get("recall", 0) or 0),
        f1_score=float(evaluation.get("f1", 0) or 0),
        evaluation=evaluation,
        inference_ms=round(inference_ms, 3),
        status=ModelVersion.Status.READY,
        created_by=job.created_by,
    )


# ---------------------------------------------------------------------------
# Dataset bootstrapping
# ---------------------------------------------------------------------------
def harvest_false_positives(dataset, limit: int = 200) -> int:
    """Pull frames operators marked as false positives into a dataset.

    This closes the loop that makes the product get better with use: every time
    somebody dismisses a bad alert, that frame becomes a hard negative for the
    next training run.
    """
    from apps.events.models import Event

    from .models import DatasetSample

    query = Event.objects.filter(
        organization=dataset.organization,
        status=Event.Status.FALSE_POSITIVE,
        snapshot__isnull=False,
    ).exclude(dataset_samples__dataset=dataset)
    if dataset.harvest_analytics:
        query = query.filter(analytic__in=dataset.harvest_analytics)

    added = 0
    negative_label = dataset.classes[0] if dataset.classes else "normal"
    for event in query.select_related("snapshot", "camera")[:limit]:
        DatasetSample.objects.create(
            dataset=dataset,
            image=event.snapshot.image,
            label=negative_label,
            source_event=event,
            source_camera=event.camera,
            notes=f"Harvested from false-positive event {event.uid}",
        )
        added += 1

    if added:
        dataset.refresh_counts()
    return added
