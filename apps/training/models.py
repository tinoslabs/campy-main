"""The Campy AI training platform: datasets, annotations, jobs, model registry.

This is what makes the AI *ours* in the way that matters commercially — a
customer can take their own footage, label it, and train a model that runs on
their own cameras, without leaving the product or sending a frame to a
third-party API.
"""
from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.db import models
from django.urls import reverse
from django.utils import timezone

from apps.aiengine.architectures import ARCHITECTURE_CHOICES
from apps.aiengine.registry import MODEL_SLOTS
from apps.core.models import (
    ActorStampedModel,
    OrganizationOwnedModel,
    SoftDeleteModel,
    TimeStampedModel,
    UUIDModel,
)

TASK_CHOICES = [
    ("classification", "Image classification"),
    ("embedding", "Identity embedding"),
    ("detection", "Object detection"),
    ("density", "Crowd density"),
    ("multilabel", "Multi-label classification"),
]

SLOT_CHOICES = MODEL_SLOTS


class Dataset(UUIDModel, OrganizationOwnedModel, TimeStampedModel, SoftDeleteModel, ActorStampedModel):
    """A labelled collection of samples used to train one model."""

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        LABELLING = "labelling", "Labelling"
        READY = "ready", "Ready to train"
        ARCHIVED = "archived", "Archived"

    name = models.CharField(max_length=180)
    description = models.TextField(blank=True)
    task = models.CharField(max_length=20, choices=TASK_CHOICES, default="classification")
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.DRAFT, db_index=True)

    #: Ordered class names — index position *is* the training label.
    classes = models.JSONField(default=list, blank=True)
    image_size = models.PositiveIntegerField(default=96)
    channels = models.PositiveIntegerField(default=3)

    validation_split = models.FloatField(default=0.2)
    augmentation = models.JSONField(
        default=dict,
        blank=True,
        help_text="{'flip': true, 'brightness': 0.2, 'noise': 0.05, 'shift': 0.1}",
    )

    sample_count = models.PositiveIntegerField(default=0)
    labelled_count = models.PositiveIntegerField(default=0)
    #: Cache of {class_name: count} — powers the balance warning in the UI.
    class_distribution = models.JSONField(default=dict, blank=True)

    #: Auto-populate from operator feedback on real events.
    harvest_from_events = models.BooleanField(
        default=False,
        help_text="Add frames from events operators marked as false positives.",
    )
    harvest_analytics = models.JSONField(default=list, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["organization", "status"])]
        constraints = [
            models.UniqueConstraint(fields=["organization", "name"], name="uniq_dataset_name_per_org")
        ]

    def __str__(self) -> str:
        return self.name

    def get_absolute_url(self) -> str:
        return reverse("dashboard:dataset_detail", args=[self.uid])

    @property
    def class_count(self) -> int:
        return len(self.classes or [])

    @property
    def is_trainable(self) -> bool:
        if self.task == "embedding":
            # Triplet loss needs at least two identities and a few examples each.
            return self.class_count >= 2 and self.labelled_count >= self.class_count * 3
        return self.class_count >= 2 and self.labelled_count >= self.class_count * 5

    @property
    def readiness(self) -> dict:
        """Explains exactly what is missing before training can start."""
        needed = self.class_count * (3 if self.task == "embedding" else 5)
        return {
            "classes_ok": self.class_count >= 2,
            "samples_ok": self.labelled_count >= needed,
            "labelled": self.labelled_count,
            "required": needed,
            "balance_warning": self.is_imbalanced,
        }

    @property
    def is_imbalanced(self) -> bool:
        """True when the largest class is >4x the smallest — a model trained on
        this will look accurate while ignoring the rare class entirely."""
        counts = [v for v in (self.class_distribution or {}).values() if v]
        if len(counts) < 2:
            return False
        return max(counts) > min(counts) * 4

    @property
    def labelling_progress(self) -> float:
        if not self.sample_count:
            return 0.0
        return round(self.labelled_count / self.sample_count * 100, 1)

    def storage_path(self) -> Path:
        root = Path(settings.CAMPY["DATASET_ROOT"]) / str(self.organization_id) / str(self.uid)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def refresh_counts(self) -> None:
        samples = self.samples.all()
        self.sample_count = samples.count()
        labelled = samples.exclude(label="")
        self.labelled_count = labelled.count()
        distribution: dict[str, int] = {}
        for name in labelled.values_list("label", flat=True):
            distribution[name] = distribution.get(name, 0) + 1
        self.class_distribution = distribution
        if self.status == self.Status.DRAFT and self.labelled_count:
            self.status = self.Status.LABELLING
        if self.is_trainable and self.status == self.Status.LABELLING:
            self.status = self.Status.READY
        self.save(
            update_fields=[
                "sample_count", "labelled_count", "class_distribution", "status", "updated_at",
            ]
        )


class DatasetSample(UUIDModel, TimeStampedModel):
    """One labelled image (or feature vector) inside a dataset."""

    class Split(models.TextChoices):
        TRAIN = "train", "Train"
        VALIDATION = "validation", "Validation"
        TEST = "test", "Test"

    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="samples")
    image = models.ImageField(upload_to="datasets/%Y/%m/", blank=True, null=True)
    #: For non-image tasks (gesture windows), the raw feature vector.
    features = models.JSONField(default=list, blank=True)

    label = models.CharField(max_length=120, blank=True, db_index=True)
    #: Detection tasks: [{"box": [x1,y1,x2,y2], "label": "person"}, ...]
    annotations = models.JSONField(default=list, blank=True)
    split = models.CharField(max_length=12, choices=Split.choices, default=Split.TRAIN, db_index=True)

    source_event = models.ForeignKey(
        "events.Event", null=True, blank=True, on_delete=models.SET_NULL, related_name="dataset_samples"
    )
    source_camera = models.ForeignKey(
        "cameras.Camera", null=True, blank=True, on_delete=models.SET_NULL, related_name="dataset_samples"
    )
    labelled_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="labelled_samples"
    )
    labelled_at = models.DateTimeField(null=True, blank=True)
    is_verified = models.BooleanField(default=False)
    notes = models.CharField(max_length=300, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["dataset", "label"]),
            models.Index(fields=["dataset", "split"]),
        ]

    def __str__(self) -> str:
        return f"{self.dataset.name}: {self.label or 'unlabelled'}"

    @property
    def organization(self):
        return self.dataset.organization

    @property
    def label_index(self) -> int:
        try:
            return (self.dataset.classes or []).index(self.label)
        except ValueError:
            return -1

    def set_label(self, label: str, user=None) -> None:
        self.label = label
        self.labelled_by = user
        self.labelled_at = timezone.now()
        self.save(update_fields=["label", "labelled_by", "labelled_at", "updated_at"])


class TrainingJob(UUIDModel, OrganizationOwnedModel, TimeStampedModel, ActorStampedModel):
    """One training run. Progress streams into the dashboard live."""

    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        SUCCEEDED = "succeeded", "Succeeded"
        FAILED = "failed", "Failed"
        CANCELLED = "cancelled", "Cancelled"

    name = models.CharField(max_length=180)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="jobs")
    architecture = models.CharField(max_length=20, choices=ARCHITECTURE_CHOICES, default="campynet")
    target_slot = models.CharField(max_length=32, choices=SLOT_CHOICES, blank=True)

    # -- hyper-parameters -------------------------------------------------
    epochs = models.PositiveIntegerField(default=25)
    batch_size = models.PositiveIntegerField(default=32)
    learning_rate = models.FloatField(default=0.001)
    optimizer = models.CharField(
        max_length=12,
        choices=[("adam", "Adam"), ("adamw", "AdamW"), ("sgd", "SGD with momentum")],
        default="adam",
    )
    loss_function = models.CharField(max_length=24, default="cross_entropy")
    schedule = models.CharField(
        max_length=12,
        choices=[("constant", "Constant"), ("step", "Step decay"), ("cosine", "Cosine annealing")],
        default="cosine",
    )
    width = models.FloatField(default=1.0, help_text="Channel multiplier — capacity vs speed")
    depth = models.PositiveIntegerField(default=3)
    dropout = models.FloatField(default=0.25)
    early_stopping_patience = models.PositiveIntegerField(default=6, help_text="0 = disabled")
    seed = models.IntegerField(default=1337)

    # -- lifecycle --------------------------------------------------------
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.QUEUED, db_index=True)
    progress = models.FloatField(default=0.0)
    current_epoch = models.PositiveIntegerField(default=0)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    duration_seconds = models.FloatField(default=0.0)
    error_message = models.TextField(blank=True)
    cancel_requested = models.BooleanField(default=False)

    # -- results ----------------------------------------------------------
    #: Per-epoch metric rows, appended live during the run.
    history = models.JSONField(default=list, blank=True)
    metrics = models.JSONField(default=dict, blank=True)
    log = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["organization", "status", "-created_at"])]

    def __str__(self) -> str:
        return f"{self.name} ({self.status})"

    def get_absolute_url(self) -> str:
        return reverse("dashboard:training_job_detail", args=[self.uid])

    @property
    def is_running(self) -> bool:
        return self.status in {self.Status.QUEUED, self.Status.RUNNING}

    @property
    def is_finished(self) -> bool:
        return self.status in {self.Status.SUCCEEDED, self.Status.FAILED, self.Status.CANCELLED}

    @property
    def best_metric(self) -> float:
        if not self.history:
            return 0.0
        key = "val_accuracy" if any("val_accuracy" in row for row in self.history) else "train_accuracy"
        return max((row.get(key, 0) or 0) for row in self.history)

    @property
    def final_loss(self) -> float:
        return (self.history[-1].get("loss", 0.0) if self.history else 0.0)

    @property
    def chart_data(self) -> dict:
        """Series for the live training chart."""
        return {
            "epochs": [row.get("epoch") for row in self.history],
            "loss": [row.get("loss") for row in self.history],
            "val_loss": [row.get("val_loss") for row in self.history],
            "train_accuracy": [row.get("train_accuracy") for row in self.history],
            "val_accuracy": [row.get("val_accuracy") for row in self.history],
        }

    def append_log(self, message: str) -> None:
        stamp = timezone.now().strftime("%H:%M:%S")
        self.log = f"{self.log}\n[{stamp}] {message}".strip()[-20000:]


class ModelVersion(UUIDModel, OrganizationOwnedModel, TimeStampedModel, ActorStampedModel):
    """A trained checkpoint that can be deployed to a model slot.

    ``organization=None`` marks a Campy AI **base model** available to every
    customer; a workspace's own version always takes precedence over it.
    """

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        READY = "ready", "Ready"
        DEPLOYED = "deployed", "Deployed"
        ARCHIVED = "archived", "Archived"
        FAILED = "failed", "Failed"

    name = models.CharField(max_length=180)
    version = models.CharField(max_length=32, default="1.0.0")
    slot = models.CharField(max_length=32, choices=SLOT_CHOICES, db_index=True)
    architecture = models.CharField(max_length=20, choices=ARCHITECTURE_CHOICES, default="campynet")
    task = models.CharField(max_length=20, choices=TASK_CHOICES, default="classification")

    job = models.ForeignKey(
        TrainingJob, null=True, blank=True, on_delete=models.SET_NULL, related_name="versions"
    )
    dataset = models.ForeignKey(
        Dataset, null=True, blank=True, on_delete=models.SET_NULL, related_name="versions"
    )

    weights_path = models.CharField(max_length=500, blank=True)
    file_size_bytes = models.BigIntegerField(default=0)
    parameter_count = models.BigIntegerField(default=0)
    classes = models.JSONField(default=list, blank=True)
    input_size = models.PositiveIntegerField(default=96)
    meta = models.JSONField(default=dict, blank=True)

    accuracy = models.FloatField(default=0.0)
    precision = models.FloatField(default=0.0)
    recall = models.FloatField(default=0.0)
    f1_score = models.FloatField(default=0.0)
    evaluation = models.JSONField(default=dict, blank=True)
    inference_ms = models.FloatField(default=0.0)

    status = models.CharField(max_length=12, choices=Status.choices, default=Status.DRAFT, db_index=True)
    is_base_model = models.BooleanField(default=False, help_text="Provided by Campy AI to all workspaces")
    deployed_at = models.DateTimeField(null=True, blank=True)
    deployed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="deployed_models"
    )
    notes = models.TextField(blank=True)

    # ``organization`` is nullable here, unlike the abstract base.
    organization = models.ForeignKey(
        "accounts.Organization",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="model_versions",
    )

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["organization", "slot", "status"]),
            models.Index(fields=["slot", "is_base_model"]),
        ]

    def __str__(self) -> str:
        return f"{self.name} v{self.version} [{self.slot}]"

    def get_absolute_url(self) -> str:
        return reverse("dashboard:model_detail", args=[self.uid])

    @property
    def is_deployed(self) -> bool:
        return self.status == self.Status.DEPLOYED

    @property
    def size_label(self) -> str:
        size = self.file_size_bytes
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.0f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    @property
    def slot_label(self) -> str:
        return dict(SLOT_CHOICES).get(self.slot, self.slot)

    @property
    def exists_on_disk(self) -> bool:
        return bool(self.weights_path) and Path(self.weights_path).exists()

    @classmethod
    def deployed_for(cls, slot: str, organization=None):
        """Resolve a slot: workspace model first, then the Campy AI base model."""
        if organization is not None:
            own = cls.objects.filter(
                organization=organization, slot=slot, status=cls.Status.DEPLOYED
            ).order_by("-deployed_at").first()
            if own is not None:
                return own
        return (
            cls.objects.filter(
                organization__isnull=True, slot=slot, status=cls.Status.DEPLOYED, is_base_model=True
            )
            .order_by("-deployed_at")
            .first()
        )

    def deploy(self, user=None) -> None:
        """Promote this version, retiring whatever held the slot before."""
        from apps.aiengine.registry import clear_cache

        siblings = ModelVersion.objects.filter(
            organization=self.organization, slot=self.slot, status=self.Status.DEPLOYED
        ).exclude(pk=self.pk)
        siblings.update(status=self.Status.READY, updated_at=timezone.now())

        self.status = self.Status.DEPLOYED
        self.deployed_at = timezone.now()
        self.deployed_by = user
        self.save(update_fields=["status", "deployed_at", "deployed_by", "updated_at"])
        clear_cache()

    def retire(self) -> None:
        from apps.aiengine.registry import clear_cache

        self.status = self.Status.READY
        self.save(update_fields=["status", "updated_at"])
        clear_cache()

    def load(self):
        from apps.aiengine.registry import load_checkpoint

        return load_checkpoint(self.weights_path) if self.weights_path else None
