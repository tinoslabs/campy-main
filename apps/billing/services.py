"""Subscription lifecycle, invoicing and webhook processing."""
from __future__ import annotations

import logging
from datetime import timedelta
from decimal import Decimal

from django.db import transaction
from django.utils import timezone

from apps.core.utils import money

from .gateways import GatewayError, WebhookResult, get_gateway
from .models import Coupon, Invoice, Package, Payment, Subscription, WebhookEvent

logger = logging.getLogger("campy.billing")

TAX_RATES = {"IN": 18.0}          # GST; other countries default to 0 unless configured


def tax_rate_for(organization) -> float:
    return TAX_RATES.get((organization.country or "").upper(), 0.0)


# ---------------------------------------------------------------------------
# Subscriptions
# ---------------------------------------------------------------------------
@transaction.atomic
def start_trial(organization, package: Package, user=None) -> Subscription:
    """Put a workspace on a package's trial without taking payment."""
    organization.refresh_subscription()
    existing = organization.active_subscription
    if existing is not None:
        existing.status = Subscription.Status.CANCELLED
        existing.ended_at = timezone.now()
        existing.save(update_fields=["status", "ended_at", "updated_at"])

    trial_days = package.trial_days or 14
    now = timezone.now()
    subscription = Subscription.objects.create(
        organization=organization,
        package=package,
        status=Subscription.Status.TRIALING,
        gateway=Subscription.Gateway.MANUAL,
        currency="USD" if (organization.country or "IN").upper() != "IN" else "INR",
        started_at=now,
        trial_ends_at=now + timedelta(days=trial_days),
        current_period_start=now,
        current_period_end=now + timedelta(days=trial_days),
    )
    organization.status = organization.Status.TRIAL
    organization.trial_ends_at = subscription.trial_ends_at
    organization.save(update_fields=["status", "trial_ends_at", "updated_at"])

    Package.objects.filter(pk=package.pk).update(subscriber_count=package.subscriber_count + 1)
    organization.refresh_subscription()
    _log(organization, user, f"Started {trial_days}-day trial on {package.name}")
    return subscription


@transaction.atomic
def create_invoice_for_package(
    organization, package: Package, *, currency: str | None = None,
    coupon: Coupon | None = None, extra_cameras: int = 0, user=None,
) -> Invoice:
    """Build the invoice a customer is about to pay."""
    currency = (currency or ("INR" if (organization.country or "IN").upper() == "IN" else "USD")).upper()
    unit_price = package.price_for(currency)
    per_camera = package.per_camera_usd if currency == "USD" else package.per_camera_inr

    line_items = [
        {
            "description": f"{package.name} — {package.get_interval_display()}",
            "quantity": 1,
            "unit_price": str(money(unit_price)),
            "amount": str(money(unit_price)),
        }
    ]
    subtotal = Decimal(unit_price)

    if extra_cameras and per_camera:
        amount = money(Decimal(per_camera) * extra_cameras)
        line_items.append(
            {
                "description": f"Additional cameras × {extra_cameras}",
                "quantity": extra_cameras,
                "unit_price": str(money(per_camera)),
                "amount": str(amount),
            }
        )
        subtotal += amount

    setup_fee = package.setup_fee_for(currency)
    if setup_fee:
        line_items.append(
            {
                "description": "One-time setup & onboarding",
                "quantity": 1,
                "unit_price": str(money(setup_fee)),
                "amount": str(money(setup_fee)),
            }
        )
        subtotal += Decimal(setup_fee)

    discount = coupon.discount_for(subtotal, currency) if coupon else Decimal("0.00")

    invoice = Invoice.objects.create(
        organization=organization,
        package=package,
        subscription=organization.active_subscription,
        status=Invoice.Status.OPEN,
        currency=currency,
        subtotal=money(subtotal),
        discount=discount,
        tax_rate=tax_rate_for(organization),
        line_items=line_items,
        issued_at=timezone.now(),
        due_at=timezone.now() + timedelta(days=7),
        period_start=timezone.now(),
        period_end=timezone.now() + timedelta(days=30 * max(package.interval_months, 1)),
        billing_name=organization.legal_name or organization.name,
        billing_email=organization.contact_email,
        billing_address=organization.address,
        tax_id=organization.tax_id,
    )
    _log(organization, user, f"Issued invoice {invoice.number} for {package.name}")
    return invoice


def start_checkout(invoice: Invoice, gateway_key: str, request=None, user=None) -> tuple[Payment, object]:
    """Create a gateway order and the matching local Payment record."""
    gateway = get_gateway(gateway_key)
    if not gateway.is_configured:
        raise GatewayError(f"{gateway.label} is not configured on this deployment.")

    intent = gateway.create_checkout(invoice, request)
    payment = Payment.objects.create(
        organization=invoice.organization,
        invoice=invoice,
        subscription=invoice.subscription,
        package=invoice.package,
        gateway=gateway.key,
        status=Payment.Status.CREATED,
        currency=intent.currency,
        amount=invoice.total,
        gateway_order_id=intent.order_id,
        receipt=invoice.number,
        initiated_by=user,
    )
    return payment, intent


@transaction.atomic
def complete_payment(payment: Payment, gateway_payment_id: str = "", raw: dict | None = None, user=None) -> Subscription:
    """Mark a payment captured, settle the invoice and activate the plan."""
    payment.status = Payment.Status.CAPTURED
    payment.gateway_payment_id = gateway_payment_id or payment.gateway_payment_id
    payment.raw_response = raw or payment.raw_response
    payment.completed_at = timezone.now()
    payment.save(
        update_fields=[
            "status", "gateway_payment_id", "raw_response", "completed_at", "updated_at",
        ]
    )

    invoice = payment.invoice
    if invoice is not None and not invoice.is_paid:
        invoice.mark_paid(payment.amount)

    return activate_subscription(
        payment.organization,
        payment.package or (invoice.package if invoice else None),
        gateway=payment.gateway,
        currency=payment.currency,
        user=user,
    )


@transaction.atomic
def activate_subscription(organization, package: Package | None, *, gateway="manual",
                          currency="INR", user=None) -> Subscription | None:
    """Move a workspace onto a paid package."""
    if package is None:
        return None

    organization.refresh_subscription()
    subscription = organization.active_subscription
    now = timezone.now()
    period_end = now + timedelta(days=30 * max(package.interval_months, 1))

    if subscription is None:
        subscription = Subscription.objects.create(
            organization=organization,
            package=package,
            status=Subscription.Status.ACTIVE,
            gateway=gateway,
            currency=currency,
            started_at=now,
            current_period_start=now,
            current_period_end=period_end,
        )
    else:
        previous = subscription.package
        subscription.package = package
        subscription.status = Subscription.Status.ACTIVE
        subscription.gateway = gateway
        subscription.currency = currency
        subscription.current_period_start = now
        subscription.current_period_end = period_end
        subscription.cancel_at_period_end = False
        subscription.cancelled_at = None
        subscription.save(
            update_fields=[
                "package", "status", "gateway", "currency", "current_period_start",
                "current_period_end", "cancel_at_period_end", "cancelled_at", "updated_at",
            ]
        )
        if previous and previous.pk != package.pk:
            Package.objects.filter(pk=previous.pk).update(
                subscriber_count=max(previous.subscriber_count - 1, 0)
            )

    organization.status = organization.Status.ACTIVE
    organization.save(update_fields=["status", "updated_at"])
    Package.objects.filter(pk=package.pk).update(subscriber_count=package.subscriber_count + 1)
    organization.refresh_subscription()

    _log(organization, user, f"Activated {package.name}", action="payment")
    return subscription


def change_package(organization, package: Package, user=None) -> Invoice | Subscription:
    """Upgrade or downgrade. Free plans apply immediately; paid ones invoice."""
    currency = "INR" if (organization.country or "IN").upper() == "IN" else "USD"
    if package.price_for(currency) <= 0:
        return activate_subscription(organization, package, currency=currency, user=user)
    return create_invoice_for_package(organization, package, currency=currency, user=user)


def cancel_subscription(organization, at_period_end: bool = True, user=None) -> Subscription | None:
    organization.refresh_subscription()
    subscription = organization.active_subscription
    if subscription is None:
        return None
    subscription.cancel(at_period_end=at_period_end)
    organization.refresh_subscription()
    if not at_period_end:
        organization.status = organization.Status.SUSPENDED
        organization.save(update_fields=["status", "updated_at"])
    _log(
        organization, user,
        "Cancelled subscription" + (" at period end" if at_period_end else " immediately"),
    )
    return subscription


# ---------------------------------------------------------------------------
# Webhooks
# ---------------------------------------------------------------------------
SUCCESS_EVENTS = {
    "payment.captured", "order.paid", "subscription.charged",              # Razorpay
    "checkout.session.completed", "payment_intent.succeeded", "invoice.paid",  # Stripe
}
FAILURE_EVENTS = {
    "payment.failed", "subscription.halted",
    "payment_intent.payment_failed", "invoice.payment_failed",
}
CANCELLATION_EVENTS = {"subscription.cancelled", "customer.subscription.deleted"}


@transaction.atomic
def process_webhook(gateway_key: str, body: bytes, signature: str) -> WebhookEvent:
    """Verify, store and act on a gateway callback — exactly once."""
    gateway = get_gateway(gateway_key)
    result: WebhookResult = gateway.parse_webhook(body, signature)

    record, created = WebhookEvent.objects.get_or_create(
        event_id=result.event_id,
        defaults={
            "gateway": gateway.key,
            "event_type": result.event_type,
            "payload": result.payload,
            "signature_valid": result.signature_valid,
        },
    )
    if not created:
        # Gateways retry aggressively; replaying would double-credit a customer.
        logger.info("Ignoring duplicate %s webhook %s", gateway.key, result.event_id)
        return record

    if not result.signature_valid:
        record.error = "Signature verification failed — the payload was not acted upon."
        record.save(update_fields=["error", "updated_at"])
        logger.warning("Rejected %s webhook %s: bad signature", gateway.key, result.event_id)
        return record

    try:
        _apply_webhook(result)
        record.processed = True
        record.processed_at = timezone.now()
    except Exception as exc:  # noqa: BLE001 - keep the payload for replay
        logger.exception("Failed to process %s webhook %s", gateway.key, result.event_id)
        record.error = str(exc)[:500]
    record.save(update_fields=["processed", "processed_at", "error", "updated_at"])
    return record


def _apply_webhook(result: WebhookResult) -> None:
    payment = None
    if result.gateway_order_id:
        payment = Payment.objects.filter(gateway_order_id=result.gateway_order_id).first()
    if payment is None and result.gateway_payment_id:
        payment = Payment.objects.filter(gateway_payment_id=result.gateway_payment_id).first()

    if result.event_type in SUCCESS_EVENTS:
        if payment is None:
            logger.warning("Success webhook with no matching payment: %s", result.gateway_order_id)
            return
        if payment.status != Payment.Status.CAPTURED:
            complete_payment(payment, result.gateway_payment_id, result.payload)
        return

    if result.event_type in FAILURE_EVENTS and payment is not None:
        payment.status = Payment.Status.FAILED
        payment.error_message = str(result.payload.get("error_description", ""))[:400]
        payment.raw_response = result.payload
        payment.save(update_fields=["status", "error_message", "raw_response", "updated_at"])
        subscription = payment.subscription or payment.organization.active_subscription
        if subscription and subscription.status == Subscription.Status.ACTIVE:
            subscription.status = Subscription.Status.PAST_DUE
            subscription.save(update_fields=["status", "updated_at"])
        return

    if result.event_type in CANCELLATION_EVENTS and result.gateway_subscription_id:
        subscription = Subscription.objects.filter(
            gateway_subscription_id=result.gateway_subscription_id
        ).first()
        if subscription:
            subscription.cancel(at_period_end=False)


def _log(organization, user, summary: str, action: str = "update") -> None:
    from apps.accounts.models import AuditLog

    AuditLog.record(
        action=action, actor=user, organization=organization, summary=summary, sensitive=True
    )
