"""Train the Campy AI base models shipped to every workspace.

    python manage.py train_base_models --slot campynet_fire --epochs 25

Base models give a brand-new customer a working detector before they have
labelled anything of their own. A workspace's own trained model always takes
precedence over one of these.
"""
from __future__ import annotations

import time
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone

from apps.aiengine.architectures import build_classifier, build_embedding_network
from apps.aiengine.registry import SLOT_KEYS, clear_cache
from apps.aiengine.simulation import synthetic_classification_dataset
from apps.training.models import ModelVersion

SLOT_CLASSES = {
    "campynet_fire": ["normal", "fire", "smoke"],
    "campynet_face": ["not_face", "face"],
    "campynet_object": ["normal", "bag", "box", "knife"],
}


class Command(BaseCommand):
    help = "Train and register Campy AI base models."

    def add_arguments(self, parser):
        parser.add_argument("--slot", default="campynet_fire", choices=sorted(SLOT_CLASSES) + ["campyface"])
        parser.add_argument("--epochs", type=int, default=20)
        parser.add_argument("--samples", type=int, default=120, help="Samples per class")
        parser.add_argument("--size", type=int, default=64)
        parser.add_argument("--width", type=float, default=1.0)
        parser.add_argument("--deploy", action="store_true", help="Deploy it once trained")

    def handle(self, *args, **options):
        import numpy as np

        from apps.aiengine.vision.image import normalise, to_chw

        slot = options["slot"]
        size = options["size"]
        self.stdout.write(self.style.MIGRATE_HEADING(f"Training base model for slot '{slot}'"))
        self.stdout.write(
            self.style.WARNING(
                "  Note: this trains on generated data. It proves the pipeline and gives new "
                "workspaces a starting point — real footage produces a far better model."
            )
        )

        if slot == "campyface":
            classes = [f"identity_{index}" for index in range(8)]
            model = build_embedding_network(embedding_dim=64, width=options["width"], input_size=size)
            task, loss = "embedding", "triplet"
        else:
            classes = SLOT_CLASSES[slot]
            model = build_classifier(
                num_classes=len(classes), width=options["width"], depth=3, input_size=size
            )
            task, loss = "classification", "cross_entropy"

        images, labels = synthetic_classification_dataset(
            classes, samples_per_class=options["samples"], size=size, seed=7
        )
        features = np.stack([to_chw(normalise(image.astype(np.float32))) for image in images])

        split = int(len(features) * 0.8)
        x_train, y_train = features[:split], labels[:split]
        x_val, y_val = features[split:], labels[split:]
        self.stdout.write(f"  {len(x_train)} training / {len(x_val)} validation samples, "
                          f"{model.param_count:,} parameters")

        started = time.time()
        history = model.fit(
            x_train, y_train, loss=loss, optimizer="adam", schedule="cosine",
            epochs=options["epochs"], batch_size=32,
            validation_data=(x_val, y_val), task=task, verbose=True,
            early_stopping_patience=8,
        )
        elapsed = time.time() - started

        evaluation = model.evaluate(x_val, y_val, loss=loss, task=task)
        self.stdout.write(f"  trained in {elapsed:.1f}s — {evaluation}")

        model.meta.update({"classes": classes, "input_size": size, "base_model": True})
        root = Path(settings.CAMPY["MODEL_ROOT"]) / "base"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{slot}.npz"
        model.save(path)

        previous = ModelVersion.objects.filter(slot=slot, is_base_model=True).count()
        version = ModelVersion.objects.create(
            organization=None, name=f"Campy AI base — {slot}", version=f"1.{previous}.0",
            slot=slot, architecture="campyface" if slot == "campyface" else "campynet",
            task=task, weights_path=str(path),
            file_size_bytes=path.stat().st_size, parameter_count=model.param_count,
            classes=classes, input_size=size, meta=model.meta,
            accuracy=float(evaluation.get("accuracy", evaluation.get("rank1", 0)) or 0),
            precision=float(evaluation.get("precision", 0) or 0),
            recall=float(evaluation.get("recall", 0) or 0),
            f1_score=float(evaluation.get("f1", 0) or 0),
            evaluation=evaluation, status=ModelVersion.Status.READY, is_base_model=True,
        )
        self.stdout.write(self.style.SUCCESS(f"  registered {version.name} v{version.version}"))

        if options["deploy"]:
            version.status = ModelVersion.Status.DEPLOYED
            version.deployed_at = timezone.now()
            version.save(update_fields=["status", "deployed_at"])
            clear_cache()
            self.stdout.write(self.style.SUCCESS("  deployed to every workspace"))
        _ = history, SLOT_KEYS
