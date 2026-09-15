"""Payment gateway integrations: Razorpay (India) and Stripe (international).

Both gateways implement the same small interface so the checkout views and the
subscription service never branch on which one a customer is using:

    gateway = get_gateway("razorpay")
    intent = gateway.create_checkout(invoice, request)   -> CheckoutIntent
    gateway.verify(payload)                              -> bool
    gateway.parse_webhook(body, signature)               -> WebhookResult

Money is handled in the gateway's *minor* unit (paise / cents) at the boundary
and converted back to Decimal immediately — never float.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
from dataclasses import dataclass, field
from decimal import Decimal

from django.conf import settings
from django.urls import reverse
from django.utils import timezone

from apps.core.utils import from_minor_units, to_minor_units

logger = logging.getLogger("campy.billing")


class GatewayError(Exception):
    """Raised when a gateway rejects a request or is not configured."""


@dataclass
class CheckoutIntent:
    """Everything the front end needs to open a payment flow."""

    gateway: str
    order_id: str
    amount: Decimal
    currency: str
    #: Razorpay renders inline via its JS SDK; Stripe redirects to Checkout.
    mode: str = "inline"
    redirect_url: str = ""
    public_key: str = ""
    payload: dict = field(default_factory=dict)


@dataclass
class WebhookResult:
    event_id: str
    event_type: str
    payload: dict
    signature_valid: bool
    gateway_payment_id: str = ""
    gateway_order_id: str = ""
    gateway_subscription_id: str = ""
    amount: Decimal = Decimal("0.00")
    currency: str = "INR"
    status: str = ""


class BaseGateway:
    key = "base"
    label = "Base"

    @property
    def is_configured(self) -> bool:
        return False

    def create_checkout(self, invoice, request=None) -> CheckoutIntent:
        raise NotImplementedError

    def verify(self, payload: dict) -> bool:
        raise NotImplementedError

    def parse_webhook(self, body: bytes, signature: str) -> WebhookResult:
        raise NotImplementedError

    def refund(self, payment, amount=None) -> dict:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Razorpay
# ---------------------------------------------------------------------------
class RazorpayGateway(BaseGateway):
    key = "razorpay"
    label = "Razorpay"

    def __init__(self):
        self.config = settings.RAZORPAY
        self._client = None

    @property
    def is_configured(self) -> bool:
        return bool(self.config.get("KEY_ID") and self.config.get("KEY_SECRET"))

    @property
    def client(self):
        if self._client is None:
            if not self.is_configured:
                raise GatewayError("Razorpay is not configured. Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET.")
            import razorpay

            self._client = razorpay.Client(auth=(self.config["KEY_ID"], self.config["KEY_SECRET"]))
            self._client.set_app_details({"title": "Campy AI", "version": "1.0.0"})
        return self._client

    def create_checkout(self, invoice, request=None) -> CheckoutIntent:
        currency = (invoice.currency or self.config.get("CURRENCY") or "INR").upper()
        amount_minor = to_minor_units(invoice.total, currency)
        if amount_minor <= 0:
            raise GatewayError("Cannot start a payment for a zero-value invoice.")

        order = self.client.order.create(
            {
                "amount": amount_minor,
                "currency": currency,
                "receipt": invoice.number,
                "notes": {
                    "invoice": invoice.number,
                    "organization": invoice.organization.slug,
                    "package": invoice.package.name if invoice.package else "",
                },
            }
        )
        return CheckoutIntent(
            gateway=self.key,
            order_id=order["id"],
            amount=Decimal(invoice.total),
            currency=currency,
            mode="inline",
            public_key=self.config["KEY_ID"],
            payload={
                "key": self.config["KEY_ID"],
                "amount": amount_minor,
                "currency": currency,
                "name": settings.CAMPY["BRAND_NAME"],
                "description": f"Invoice {invoice.number}",
                "order_id": order["id"],
                "prefill": {
                    "name": invoice.billing_name or invoice.organization.name,
                    "email": invoice.billing_email or invoice.organization.contact_email,
                    "contact": invoice.organization.contact_phone,
                },
                "notes": {"invoice": invoice.number},
                "theme": {"color": "#4f46e5"},
            },
        )

    def verify(self, payload: dict) -> bool:
        """Verify the ``razorpay_signature`` returned by Checkout.

        The signature is HMAC-SHA256 of ``order_id|payment_id`` keyed by the
        API secret. Skipping this check is how sites get charged-back into
        oblivion: the browser cannot be trusted to report a real payment.
        """
        order_id = payload.get("razorpay_order_id", "")
        payment_id = payload.get("razorpay_payment_id", "")
        signature = payload.get("razorpay_signature", "")
        if not (order_id and payment_id and signature):
            return False
        expected = hmac.new(
            self.config["KEY_SECRET"].encode(),
            f"{order_id}|{payment_id}".encode(),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    def parse_webhook(self, body: bytes, signature: str) -> WebhookResult:
        secret = self.config.get("WEBHOOK_SECRET", "")
        valid = False
        if secret and signature:
            expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
            valid = hmac.compare_digest(expected, signature)

        data = json.loads(body.decode("utf-8") or "{}")
        event_type = data.get("event", "")
        entity = (
            data.get("payload", {}).get("payment", {}).get("entity")
            or data.get("payload", {}).get("subscription", {}).get("entity")
            or {}
        )
        currency = (entity.get("currency") or "INR").upper()
        return WebhookResult(
            event_id=str(data.get("id") or entity.get("id") or f"rzp-{timezone.now().timestamp()}"),
            event_type=event_type,
            payload=data,
            signature_valid=valid,
            gateway_payment_id=entity.get("id", "") if "payment" in event_type else "",
            gateway_order_id=entity.get("order_id", ""),
            gateway_subscription_id=entity.get("id", "") if "subscription" in event_type else "",
            amount=from_minor_units(int(entity.get("amount") or 0), currency),
            currency=currency,
            status=entity.get("status", ""),
        )

    def fetch_payment(self, payment_id: str) -> dict:
        return self.client.payment.fetch(payment_id)

    def refund(self, payment, amount=None) -> dict:
        if not payment.gateway_payment_id:
            raise GatewayError("This payment has no Razorpay payment id to refund.")
        body = {}
        if amount is not None:
            body["amount"] = to_minor_units(amount, payment.currency)
        return self.client.payment.refund(payment.gateway_payment_id, body)


# ---------------------------------------------------------------------------
# Stripe
# ---------------------------------------------------------------------------
class StripeGateway(BaseGateway):
    key = "stripe"
    label = "Stripe"

    def __init__(self):
        self.config = settings.STRIPE

    @property
    def is_configured(self) -> bool:
        return bool(self.config.get("SECRET_KEY"))

    @property
    def stripe(self):
        if not self.is_configured:
            raise GatewayError("Stripe is not configured. Set STRIPE_SECRET_KEY.")
        import stripe

        stripe.api_key = self.config["SECRET_KEY"]
        stripe.api_version = "2024-06-20"
        return stripe

    def create_checkout(self, invoice, request=None) -> CheckoutIntent:
        currency = (invoice.currency or self.config.get("CURRENCY") or "usd").lower()
        amount_minor = to_minor_units(invoice.total, currency)
        if amount_minor <= 0:
            raise GatewayError("Cannot start a payment for a zero-value invoice.")

        success_url = failure_url = ""
        if request is not None:
            success_url = request.build_absolute_uri(
                reverse("billing:checkout_success", args=[invoice.uid])
            ) + "?session_id={CHECKOUT_SESSION_ID}"
            failure_url = request.build_absolute_uri(
                reverse("billing:checkout_cancelled", args=[invoice.uid])
            )

        session = self.stripe.checkout.Session.create(
            mode="payment",
            line_items=[
                {
                    "price_data": {
                        "currency": currency,
                        "unit_amount": amount_minor,
                        "product_data": {
                            "name": invoice.package.name if invoice.package else "Campy AI subscription",
                            "description": f"Invoice {invoice.number}",
                        },
                    },
                    "quantity": 1,
                }
            ],
            customer_email=invoice.billing_email or None,
            client_reference_id=invoice.number,
            metadata={
                "invoice": invoice.number,
                "invoice_uid": str(invoice.uid),
                "organization": invoice.organization.slug,
            },
            success_url=success_url or "https://example.com/success",
            cancel_url=failure_url or "https://example.com/cancel",
        )
        return CheckoutIntent(
            gateway=self.key,
            order_id=session["id"],
            amount=Decimal(invoice.total),
            currency=currency.upper(),
            mode="redirect",
            redirect_url=session["url"],
            public_key=self.config.get("PUBLISHABLE_KEY", ""),
            payload={"session_id": session["id"]},
        )

    def verify(self, payload: dict) -> bool:
        """Confirm with Stripe directly that the session is actually paid.

        The browser's return to ``success_url`` proves nothing on its own.
        """
        session_id = payload.get("session_id")
        if not session_id:
            return False
        try:
            session = self.stripe.checkout.Session.retrieve(session_id)
        except Exception:  # noqa: BLE001
            logger.exception("Could not retrieve Stripe session %s", session_id)
            return False
        return session.get("payment_status") == "paid"

    def parse_webhook(self, body: bytes, signature: str) -> WebhookResult:
        secret = self.config.get("WEBHOOK_SECRET", "")
        data, valid = {}, False

        if secret and signature:
            try:
                event = self.stripe.Webhook.construct_event(body, signature, secret)
                data, valid = dict(event), True
            except Exception:  # noqa: BLE001 - fall through to unverified parse
                logger.warning("Stripe webhook signature verification failed")
        if not data:
            data = json.loads(body.decode("utf-8") or "{}")

        obj = (data.get("data") or {}).get("object") or {}
        currency = (obj.get("currency") or "usd").upper()
        amount_minor = int(obj.get("amount_total") or obj.get("amount") or obj.get("amount_paid") or 0)
        return WebhookResult(
            event_id=str(data.get("id") or f"stripe-{timezone.now().timestamp()}"),
            event_type=str(data.get("type", "")),
            payload=data,
            signature_valid=valid,
            gateway_payment_id=str(obj.get("payment_intent") or obj.get("id") or ""),
            gateway_order_id=str(obj.get("id") or ""),
            gateway_subscription_id=str(obj.get("subscription") or ""),
            amount=from_minor_units(amount_minor, currency),
            currency=currency,
            status=str(obj.get("payment_status") or obj.get("status") or ""),
        )

    def refund(self, payment, amount=None) -> dict:
        if not payment.gateway_payment_id:
            raise GatewayError("This payment has no Stripe payment intent to refund.")
        body = {"payment_intent": payment.gateway_payment_id}
        if amount is not None:
            body["amount"] = to_minor_units(amount, payment.currency)
        return self.stripe.Refund.create(**body)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
GATEWAYS = {
    RazorpayGateway.key: RazorpayGateway,
    StripeGateway.key: StripeGateway,
}


def get_gateway(key: str) -> BaseGateway:
    cls = GATEWAYS.get(str(key).lower())
    if cls is None:
        raise GatewayError(f"Unknown payment gateway '{key}'.")
    return cls()


def available_gateways() -> list[BaseGateway]:
    """Only gateways the deployment has actually configured."""
    return [gateway for gateway in (cls() for cls in GATEWAYS.values()) if gateway.is_configured]


def default_gateway_for(currency: str) -> str:
    """Razorpay for rupees, Stripe for everything else — falling back to
    whichever is actually configured."""
    available = {g.key for g in available_gateways()}
    preferred = "razorpay" if str(currency).upper() == "INR" else "stripe"
    if preferred in available:
        return preferred
    return next(iter(available), preferred)
