"""Packages, subscriptions, invoices and payments (Razorpay + Stripe)."""
from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from django.conf import settings
from django.db import models
from django.urls import reverse
from django.utils import timezone

from apps.core.models import (
    OrganizationOwnedModel,
    SluggedModel,
    TimeStampedModel,
    UUIDModel,
)
from apps.core.utils import money

# ---------------------------------------------------------------------------
# Feature catalogue — what a package can switch on
# ---------------------------------------------------------------------------
FEATURES = [
    ("gesture_tracking", "Employee gesture tracking"),
    ("face_recognition", "Face recognition & attendance"),
    ("geofencing", "Geofencing / restricted areas"),
    ("crowd_management", "Crowd management"),
    ("theft_detection", "Theft detection"),
    ("object_detection", "Dangerous & unattended objects"),
    ("fire_detection", "Fire & smoke detection"),
    ("custom_training", "Train your own AI models"),
    ("reports", "Scheduled reports & exports"),
    ("api_access", "REST API & webhooks"),
    ("sso", "Single sign-on"),
    ("priority_support", "Priority support"),
    ("white_label", "White labelling"),
    ("multi_site", "Multiple sites"),
    ("evidence_clips", "Video evidence clips"),
    ("audit_log", "Compliance audit log"),
]

FEATURE_KEYS = [key for key, _ in FEATURES]

QUOTAS = [
    ("cameras", "Cameras"),
    ("sites", "Sites"),
    ("users", "Team members"),
    ("employees", "Enrolled employees"),
    ("storage_gb", "Evidence storage (GB)"),
    ("retention_days", "Event retention (days)"),
    ("training_jobs_month", "Training jobs per month"),
    ("api_calls_day", "API calls per day"),
]

QUOTA_KEYS = [key for key, _ in QUOTAS]


class Package(UUIDModel, SluggedModel, TimeStampedModel):
    """A subscription plan Campy AI sells. Managed by the super admin."""

    class Interval(models.TextChoices):
        MONTHLY = "monthly", "Monthly"
        QUARTERLY = "quarterly", "Quarterly"
        YEARLY = "yearly", "Yearly"
        ONE_TIME = "one_time", "One-time"

    class Audience(models.TextChoices):
        PUBLIC = "public", "Publicly listed"
        PRIVATE = "private", "Private / by invitation"
        LEGACY = "legacy", "Legacy (existing customers only)"

    name = models.CharField(max_length=120)
    slug_source_field = "name"
    tagline = models.CharField(max_length=200, blank=True)
    description = models.TextField(blank=True)

    price_inr = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal("0.00"))
    price_usd = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal("0.00"))
    interval = models.CharField(max_length=12, choices=Interval.choices, default=Interval.MONTHLY)
    trial_days = models.PositiveIntegerField(default=14)
    setup_fee_inr = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal("0.00"))
    setup_fee_usd = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal("0.00"))

    #: {"face_recognition": true, ...}
    features = models.JSONField(default=dict, blank=True)
    #: {"cameras": 10, "sites": 2, ...} — ``-1`` means unlimited.
    quotas = models.JSONField(default=dict, blank=True)

    #: Metered overage, e.g. per extra camera per month.
    per_camera_inr = models.DecimalField(max_digits=8, decimal_places=2, default=Decimal("0.00"))
    per_camera_usd = models.DecimalField(max_digits=8, decimal_places=2, default=Decimal("0.00"))

    audience = models.CharField(max_length=10, choices=Audience.choices, default=Audience.PUBLIC)
    is_active = models.BooleanField(default=True)
    is_featured = models.BooleanField(default=False, help_text="Highlighted on the pricing page")
    sort_order = models.PositiveIntegerField(default=100)
    badge = models.CharField(max_length=40, blank=True, help_text="e.g. 'Most popular'")
    colour = models.CharField(max_length=7, default="#4f46e5")

    # Gateway product identifiers (optional — we can also charge ad-hoc).
    razorpay_plan_id = models.CharField(max_length=120, blank=True)
    stripe_price_id = models.CharField(max_length=120, blank=True)

    subscriber_count = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["sort_order", "price_inr"]
        indexes = [models.Index(fields=["is_active", "audience"])]

    def __str__(self) -> str:
        return self.name

    def get_absolute_url(self) -> str:
        return reverse("billing:package_detail", args=[self.slug])

    # -- capability lookups ------------------------------------------------
    def has_feature(self, key: str) -> bool:
        return bool((self.features or {}).get(key))

    def quota(self, key: str, default: int = 0) -> int:
        value = (self.quotas or {}).get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def is_unlimited(self, key: str) -> bool:
        return self.quota(key, 0) < 0

    @property
    def feature_list(self) -> list[str]:
        labels = dict(FEATURES)
        return [labels[key] for key in FEATURE_KEYS if self.has_feature(key)]

    @property
    def missing_features(self) -> list[str]:
        labels = dict(FEATURES)
        return [labels[key] for key in FEATURE_KEYS if not self.has_feature(key)]

    @property
    def analytics_count(self) -> int:
        analytic_features = {
            "gesture_tracking", "face_recognition", "geofencing", "crowd_management",
            "theft_detection", "object_detection", "fire_detection",
        }
        return sum(1 for key in analytic_features if self.has_feature(key))

    def price_for(self, currency: str) -> Decimal:
        return self.price_usd if str(currency).upper() == "USD" else self.price_inr

    def setup_fee_for(self, currency: str) -> Decimal:
        return self.setup_fee_usd if str(currency).upper() == "USD" else self.setup_fee_inr

    def quota_label(self, key: str) -> str:
        value = self.quota(key, 0)
        return "Unlimited" if value < 0 else f"{value:,}"

    @property
    def interval_months(self) -> int:
        return {"monthly": 1, "quarterly": 3, "yearly": 12, "one_time": 0}.get(self.interval, 1)


class Subscription(UUIDModel, OrganizationOwnedModel, TimeStampedModel):
    """What a workspace is currently paying for."""

    organization = models.ForeignKey(
        "accounts.Organization", on_delete=models.CASCADE, related_name="subscriptions"
    )

    class Status(models.TextChoices):
        TRIALING = "trialing", "Trialing"
        ACTIVE = "active", "Active"
        PAST_DUE = "past_due", "Past due"
        CANCELLED = "cancelled", "Cancelled"
        EXPIRED = "expired", "Expired"
        PENDING = "pending", "Pending payment"

    class Gateway(models.TextChoices):
        RAZORPAY = "razorpay", "Razorpay"
        STRIPE = "stripe", "Stripe"
        MANUAL = "manual", "Manual / offline"

    package = models.ForeignKey(Package, on_delete=models.PROTECT, related_name="subscriptions")
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.TRIALING, db_index=True)
    gateway = models.CharField(max_length=10, choices=Gateway.choices, default=Gateway.MANUAL)
    currency = models.CharField(max_length=3, default="INR")

    started_at = models.DateTimeField(default=timezone.now)
    trial_ends_at = models.DateTimeField(null=True, blank=True)
    current_period_start = models.DateTimeField(default=timezone.now)
    current_period_end = models.DateTimeField(null=True, blank=True)
    cancelled_at = models.DateTimeField(null=True, blank=True)
    cancel_at_period_end = models.BooleanField(default=False)
    ended_at = models.DateTimeField(null=True, blank=True)

    #: Snapshot of the package at purchase time, so a later plan edit never
    #: silently changes what an existing customer is entitled to.
    feature_overrides = models.JSONField(default=dict, blank=True)
    quota_overrides = models.JSONField(default=dict, blank=True)

    gateway_customer_id = models.CharField(max_length=120, blank=True)
    gateway_subscription_id = models.CharField(max_length=120, blank=True, db_index=True)

    seats = models.PositiveIntegerField(default=1)
    extra_cameras = models.PositiveIntegerField(default=0)
    discount_percent = models.FloatField(default=0.0)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["organization", "status"])]

    def __str__(self) -> str:
        return f"{self.organization.name} → {self.package.name} ({self.status})"

    # -- entitlement -------------------------------------------------------
    def has_feature(self, key: str) -> bool:
        if key in (self.feature_overrides or {}):
            return bool(self.feature_overrides[key])
        if not self.is_entitled:
            return False
        return self.package.has_feature(key)

    def quota(self, key: str, default: int = 0) -> int:
        if key in (self.quota_overrides or {}):
            try:
                return int(self.quota_overrides[key])
            except (TypeError, ValueError):
                return default
        value = self.package.quota(key, default)
        if key == "cameras" and value >= 0:
            value += self.extra_cameras
        return value

    @property
    def is_entitled(self) -> bool:
        """Is this subscription currently good for service?

        ``past_due`` deliberately still counts: cutting a security system off
        the moment a card fails is the wrong behaviour. The dashboard warns
        instead, and :class:`SubscriptionGuardMiddleware` enforces the grace.
        """
        return self.status in {self.Status.ACTIVE, self.Status.TRIALING, self.Status.PAST_DUE}

    @property
    def is_trialing(self) -> bool:
        return self.status == self.Status.TRIALING

    @property
    def days_remaining(self) -> int:
        end = self.trial_ends_at if self.is_trialing else self.current_period_end
        if not end:
            return 0
        return max((end - timezone.now()).days, 0)

    @property
    def is_expiring_soon(self) -> bool:
        return self.is_entitled and 0 < self.days_remaining <= 7

    @property
    def in_grace_period(self) -> bool:
        """Past-due subscriptions keep working for 7 days."""
        if self.status != self.Status.PAST_DUE:
            return False
        reference = self.current_period_end or self.updated_at
        return (timezone.now() - reference) < timedelta(days=7)

    @property
    def amount(self) -> Decimal:
        base = self.package.price_for(self.currency)
        extra = (
            self.package.per_camera_usd if self.currency.upper() == "USD"
            else self.package.per_camera_inr
        ) * self.extra_cameras
        total = base + extra
        if self.discount_percent:
            total = total * Decimal(1 - self.discount_percent / 100)
        return money(total)

    def renew(self, months: int | None = None) -> None:
        months = months if months is not None else self.package.interval_months
        start = timezone.now()
        self.current_period_start = start
        self.current_period_end = start + timedelta(days=30 * max(months, 1))
        self.status = self.Status.ACTIVE
        self.save(
            update_fields=["current_period_start", "current_period_end", "status", "updated_at"]
        )

    def cancel(self, at_period_end: bool = True) -> None:
        self.cancel_at_period_end = at_period_end
        self.cancelled_at = timezone.now()
        if not at_period_end:
            self.status = self.Status.CANCELLED
            self.ended_at = timezone.now()
        self.save(
            update_fields=["cancel_at_period_end", "cancelled_at", "status", "ended_at", "updated_at"]
        )


class Invoice(UUIDModel, OrganizationOwnedModel, TimeStampedModel):
    organization = models.ForeignKey(
        "accounts.Organization", on_delete=models.CASCADE, related_name="invoices"
    )

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        OPEN = "open", "Open"
        PAID = "paid", "Paid"
        VOID = "void", "Void"
        REFUNDED = "refunded", "Refunded"
        FAILED = "failed", "Failed"

    number = models.CharField(max_length=32, unique=True, db_index=True)
    subscription = models.ForeignKey(
        Subscription, null=True, blank=True, on_delete=models.SET_NULL, related_name="invoices"
    )
    package = models.ForeignKey(Package, null=True, blank=True, on_delete=models.SET_NULL, related_name="invoices")

    status = models.CharField(max_length=10, choices=Status.choices, default=Status.DRAFT, db_index=True)
    currency = models.CharField(max_length=3, default="INR")
    subtotal = models.DecimalField(max_digits=12, decimal_places=2, default=Decimal("0.00"))
    tax_rate = models.FloatField(default=18.0, help_text="GST / VAT percentage")
    tax_amount = models.DecimalField(max_digits=12, decimal_places=2, default=Decimal("0.00"))
    discount = models.DecimalField(max_digits=12, decimal_places=2, default=Decimal("0.00"))
    total = models.DecimalField(max_digits=12, decimal_places=2, default=Decimal("0.00"))
    amount_paid = models.DecimalField(max_digits=12, decimal_places=2, default=Decimal("0.00"))

    #: [{"description": ..., "quantity": 1, "unit_price": "…", "amount": "…"}]
    line_items = models.JSONField(default=list, blank=True)

    issued_at = models.DateTimeField(default=timezone.now)
    due_at = models.DateTimeField(null=True, blank=True)
    paid_at = models.DateTimeField(null=True, blank=True)
    period_start = models.DateTimeField(null=True, blank=True)
    period_end = models.DateTimeField(null=True, blank=True)

    billing_name = models.CharField(max_length=200, blank=True)
    billing_email = models.EmailField(blank=True)
    billing_address = models.TextField(blank=True)
    tax_id = models.CharField(max_length=64, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["-issued_at"]
        indexes = [models.Index(fields=["organization", "status", "-issued_at"])]

    def __str__(self) -> str:
        return self.number

    def get_absolute_url(self) -> str:
        return reverse("billing:invoice_detail", args=[self.uid])

    def save(self, *args, **kwargs):
        if not self.number:
            self.number = self._next_number()
        self.tax_amount = money(Decimal(self.subtotal - self.discount) * Decimal(self.tax_rate) / 100)
        self.total = money(Decimal(self.subtotal) - Decimal(self.discount) + Decimal(self.tax_amount))
        super().save(*args, **kwargs)

    def _next_number(self) -> str:
        prefix = f"CAI-{timezone.now():%Y%m}"
        last = (
            Invoice.objects.filter(number__startswith=prefix)
            .order_by("-number")
            .values_list("number", flat=True)
            .first()
        )
        sequence = int(last.rsplit("-", 1)[-1]) + 1 if last else 1
        return f"{prefix}-{sequence:05d}"

    @property
    def is_paid(self) -> bool:
        return self.status == self.Status.PAID

    @property
    def balance_due(self) -> Decimal:
        return money(Decimal(self.total) - Decimal(self.amount_paid))

    @property
    def is_overdue(self) -> bool:
        return bool(self.due_at and not self.is_paid and self.due_at < timezone.now())

    @property
    def currency_symbol(self) -> str:
        return {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}.get(self.currency.upper(), "")

    def mark_paid(self, amount=None) -> None:
        self.amount_paid = money(amount if amount is not None else self.total)
        self.status = self.Status.PAID
        self.paid_at = timezone.now()
        self.save(update_fields=["amount_paid", "status", "paid_at", "updated_at"])


class Payment(UUIDModel, OrganizationOwnedModel, TimeStampedModel):
    """A single charge attempt against a gateway."""

    organization = models.ForeignKey(
        "accounts.Organization", on_delete=models.CASCADE, related_name="payments"
    )

    class Status(models.TextChoices):
        CREATED = "created", "Created"
        PENDING = "pending", "Pending"
        AUTHORIZED = "authorized", "Authorized"
        CAPTURED = "captured", "Captured"
        FAILED = "failed", "Failed"
        REFUNDED = "refunded", "Refunded"
        CANCELLED = "cancelled", "Cancelled"

    class Gateway(models.TextChoices):
        RAZORPAY = "razorpay", "Razorpay"
        STRIPE = "stripe", "Stripe"
        MANUAL = "manual", "Manual"

    invoice = models.ForeignKey(Invoice, null=True, blank=True, on_delete=models.SET_NULL, related_name="payments")
    subscription = models.ForeignKey(
        Subscription, null=True, blank=True, on_delete=models.SET_NULL, related_name="payments"
    )
    package = models.ForeignKey(Package, null=True, blank=True, on_delete=models.SET_NULL, related_name="payments")

    gateway = models.CharField(max_length=10, choices=Gateway.choices)
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.CREATED, db_index=True)
    currency = models.CharField(max_length=3, default="INR")
    amount = models.DecimalField(max_digits=12, decimal_places=2, default=Decimal("0.00"))
    amount_refunded = models.DecimalField(max_digits=12, decimal_places=2, default=Decimal("0.00"))

    # Gateway identifiers — Razorpay order/payment, Stripe session/intent.
    gateway_order_id = models.CharField(max_length=140, blank=True, db_index=True)
    gateway_payment_id = models.CharField(max_length=140, blank=True, db_index=True)
    gateway_signature = models.CharField(max_length=300, blank=True)
    receipt = models.CharField(max_length=64, blank=True)

    method = models.CharField(max_length=40, blank=True, help_text="card, upi, netbanking, …")
    card_brand = models.CharField(max_length=30, blank=True)
    card_last4 = models.CharField(max_length=4, blank=True)

    error_code = models.CharField(max_length=80, blank=True)
    error_message = models.CharField(max_length=400, blank=True)
    raw_response = models.JSONField(default=dict, blank=True)

    initiated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="payments"
    )
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["organization", "status", "-created_at"]),
            models.Index(fields=["gateway", "gateway_order_id"]),
        ]

    def __str__(self) -> str:
        return f"{self.gateway} {self.amount} {self.currency} ({self.status})"

    @property
    def is_successful(self) -> bool:
        return self.status in {self.Status.CAPTURED, self.Status.AUTHORIZED}

    @property
    def currency_symbol(self) -> str:
        return {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}.get(self.currency.upper(), "")


class WebhookEvent(TimeStampedModel):
    """Every gateway callback, stored verbatim.

    Kept for two reasons: replaying a missed event, and proving what a gateway
    actually told us when a payment is disputed. ``event_id`` is unique so a
    retried delivery is processed exactly once.
    """

    gateway = models.CharField(max_length=10, db_index=True)
    event_id = models.CharField(max_length=200, unique=True)
    event_type = models.CharField(max_length=120, db_index=True)
    payload = models.JSONField(default=dict)
    signature_valid = models.BooleanField(default=False)
    processed = models.BooleanField(default=False, db_index=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    error = models.CharField(max_length=500, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["gateway", "processed"])]

    def __str__(self) -> str:
        return f"{self.gateway}:{self.event_type}"


class UsageRecord(TimeStampedModel):
    """Daily metering per workspace — powers quota enforcement and overage."""

    organization = models.ForeignKey(
        "accounts.Organization", on_delete=models.CASCADE, related_name="usage_records"
    )
    date = models.DateField(db_index=True)
    cameras_active = models.PositiveIntegerField(default=0)
    frames_processed = models.BigIntegerField(default=0)
    events_generated = models.PositiveIntegerField(default=0)
    api_calls = models.PositiveIntegerField(default=0)
    training_minutes = models.FloatField(default=0.0)
    storage_bytes = models.BigIntegerField(default=0)

    class Meta:
        ordering = ["-date"]
        constraints = [
            models.UniqueConstraint(fields=["organization", "date"], name="uniq_usage_per_day")
        ]

    def __str__(self) -> str:
        return f"{self.organization.name} {self.date}"


class Coupon(TimeStampedModel):
    """Discount code applied at checkout."""

    code = models.CharField(max_length=40, unique=True, db_index=True)
    description = models.CharField(max_length=200, blank=True)
    percent_off = models.FloatField(default=0.0)
    amount_off_inr = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal("0.00"))
    amount_off_usd = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal("0.00"))
    packages = models.ManyToManyField(Package, blank=True, related_name="coupons")
    max_redemptions = models.PositiveIntegerField(default=0, help_text="0 = unlimited")
    redemptions = models.PositiveIntegerField(default=0)
    valid_from = models.DateTimeField(default=timezone.now)
    valid_until = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.code

    @property
    def is_redeemable(self) -> bool:
        if not self.is_active:
            return False
        now = timezone.now()
        if self.valid_from > now:
            return False
        if self.valid_until and self.valid_until < now:
            return False
        if self.max_redemptions and self.redemptions >= self.max_redemptions:
            return False
        return True

    def discount_for(self, amount: Decimal, currency: str = "INR") -> Decimal:
        if not self.is_redeemable:
            return Decimal("0.00")
        if self.percent_off:
            return money(Decimal(amount) * Decimal(self.percent_off) / 100)
        flat = self.amount_off_usd if currency.upper() == "USD" else self.amount_off_inr
        return money(min(Decimal(flat), Decimal(amount)))
