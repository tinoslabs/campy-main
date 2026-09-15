"""Packages, quotas, checkout, gateway signatures and webhook handling."""
from __future__ import annotations

import hashlib
import hmac
import json
from datetime import timedelta
from decimal import Decimal

from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from apps.accounts.models import Organization
from apps.billing.gateways import GatewayError, get_gateway
from apps.billing.middleware import can_add, quota_usage
from apps.billing.models import Coupon, Package, Payment, Subscription, WebhookEvent
from apps.billing.services import (
    activate_subscription,
    cancel_subscription,
    complete_payment,
    create_invoice_for_package,
    process_webhook,
    start_trial,
)
from apps.cameras.models import Camera, Site

RAZORPAY_TEST = {
    "KEY_ID": "rzp_test_key", "KEY_SECRET": "rzp_test_secret",
    "WEBHOOK_SECRET": "hook_secret", "CURRENCY": "INR",
}


def make_org(name="Billing Co", country="IN") -> Organization:
    return Organization.objects.create(name=name, country=country, status="trial")


def make_package(**overrides) -> Package:
    defaults = {
        "name": "Test Plan", "price_inr": Decimal("10000"), "price_usd": Decimal("120"),
        "trial_days": 14,
        "features": {"geofencing": True, "fire_detection": True, "custom_training": False},
        "quotas": {"cameras": 3, "sites": 1, "users": 2, "employees": 10, "retention_days": 30},
    }
    defaults.update(overrides)
    return Package.objects.create(**defaults)


class PackageEntitlementTests(TestCase):
    def test_feature_lookup(self):
        package = make_package()
        self.assertTrue(package.has_feature("geofencing"))
        self.assertFalse(package.has_feature("custom_training"))
        self.assertFalse(package.has_feature("nonexistent_feature"))

    def test_unlimited_quota_is_negative_one(self):
        package = make_package(quotas={"cameras": -1, "sites": 2})
        self.assertTrue(package.is_unlimited("cameras"))
        self.assertFalse(package.is_unlimited("sites"))
        self.assertEqual(package.quota_label("cameras"), "Unlimited")
        self.assertEqual(package.quota_label("sites"), "2")

    def test_price_by_currency(self):
        package = make_package()
        self.assertEqual(package.price_for("INR"), Decimal("10000"))
        self.assertEqual(package.price_for("USD"), Decimal("120"))


class SubscriptionLifecycleTests(TestCase):
    def setUp(self):
        self.org = make_org()
        self.package = make_package()

    def test_trial_activates_the_workspace(self):
        subscription = start_trial(self.org, self.package)
        self.assertEqual(subscription.status, Subscription.Status.TRIALING)
        self.assertTrue(subscription.is_entitled)
        self.org.refresh_from_db()
        self.assertEqual(self.org.status, Organization.Status.TRIAL)

    def test_activation_is_visible_immediately_on_the_same_instance(self):
        """Regression: a cached_property left a freshly-activated workspace
        reporting no plan, closing every feature gate."""
        organization = make_org("Cache Co")
        self.assertIsNone(organization.active_subscription)   # primes the cache
        activate_subscription(organization, self.package, currency="INR")
        self.assertIsNotNone(organization.active_subscription)
        self.assertTrue(organization.has_feature("geofencing"))

    def test_past_due_keeps_working_through_its_grace_period(self):
        """A security system must not switch off the moment a card fails."""
        subscription = start_trial(self.org, self.package)
        subscription.status = Subscription.Status.PAST_DUE
        subscription.current_period_end = timezone.now() - timedelta(days=2)
        subscription.save()
        self.assertTrue(subscription.is_entitled)
        self.assertTrue(subscription.in_grace_period)

    def test_grace_period_eventually_expires(self):
        subscription = start_trial(self.org, self.package)
        subscription.status = Subscription.Status.PAST_DUE
        subscription.current_period_end = timezone.now() - timedelta(days=9)
        subscription.save()
        self.assertFalse(subscription.in_grace_period)

    def test_cancel_at_period_end_keeps_service_running(self):
        start_trial(self.org, self.package)
        subscription = cancel_subscription(self.org, at_period_end=True)
        self.assertTrue(subscription.cancel_at_period_end)
        self.assertTrue(subscription.is_entitled)

    def test_immediate_cancel_suspends_the_workspace(self):
        start_trial(self.org, self.package)
        subscription = cancel_subscription(self.org, at_period_end=False)
        self.assertEqual(subscription.status, Subscription.Status.CANCELLED)
        self.org.refresh_from_db()
        self.assertEqual(self.org.status, Organization.Status.SUSPENDED)

    def test_switching_package_keeps_one_active_subscription(self):
        start_trial(self.org, self.package)
        other = make_package(name="Bigger Plan", quotas={"cameras": 50})
        activate_subscription(self.org, other, currency="INR")
        self.assertEqual(
            Subscription.objects.filter(
                organization=self.org, status__in=["active", "trialing", "past_due"]
            ).count(),
            1,
        )
        self.assertEqual(self.org.refresh_subscription().package, other)


class QuotaTests(TestCase):
    def setUp(self):
        self.org = make_org()
        activate_subscription(self.org, make_package(), currency="INR")
        self.site = Site.objects.create(organization=self.org, name="HQ")

    def _add_cameras(self, count):
        for index in range(count):
            Camera.objects.create(
                organization=self.org, site=self.site, name=f"Cam {index}",
                protocol=Camera.Protocol.DEMO,
            )

    def test_within_quota_is_allowed(self):
        self._add_cameras(2)
        allowed, reason = can_add(self.org, "cameras")
        self.assertTrue(allowed, reason)

    def test_at_quota_blocks_the_next_one(self):
        self._add_cameras(3)
        allowed, reason = can_add(self.org, "cameras")
        self.assertFalse(allowed)
        self.assertIn("upgrade", reason.lower())

    def test_unlimited_never_blocks(self):
        activate_subscription(
            self.org, make_package(name="Unlimited", quotas={"cameras": -1}), currency="INR"
        )
        self._add_cameras(30)
        allowed, _ = can_add(self.org, "cameras")
        self.assertTrue(allowed)

    def test_usage_reporting_flags_the_limit(self):
        self._add_cameras(3)
        usage = quota_usage(self.org)["cameras"]
        self.assertEqual(usage["used"], 3)
        self.assertEqual(usage["limit"], 3)
        self.assertEqual(usage["percent"], 100)
        self.assertTrue(usage["near_limit"])

    def test_no_subscription_means_nothing_may_be_added(self):
        organization = make_org("Unpaid Co")
        allowed, reason = can_add(organization, "cameras")
        self.assertFalse(allowed)
        self.assertIn("no active package", reason.lower())


class InvoiceTests(TestCase):
    def setUp(self):
        self.org = make_org()
        self.package = make_package()

    def test_invoice_totals_include_tax(self):
        invoice = create_invoice_for_package(self.org, self.package, currency="INR")
        self.assertEqual(invoice.subtotal, Decimal("10000.00"))
        self.assertEqual(invoice.tax_rate, 18.0)          # India → GST
        self.assertEqual(invoice.tax_amount, Decimal("1800.00"))
        self.assertEqual(invoice.total, Decimal("11800.00"))

    def test_non_indian_workspace_has_no_gst(self):
        organization = make_org("US Co", country="US")
        invoice = create_invoice_for_package(organization, self.package, currency="USD")
        self.assertEqual(invoice.tax_rate, 0.0)
        self.assertEqual(invoice.total, invoice.subtotal)

    def test_invoice_numbers_are_sequential_and_unique(self):
        first = create_invoice_for_package(self.org, self.package)
        second = create_invoice_for_package(self.org, self.package)
        self.assertNotEqual(first.number, second.number)
        self.assertTrue(first.number.startswith("CAI-"))

    def test_coupon_reduces_the_total(self):
        coupon = Coupon.objects.create(code="HALFOFF", percent_off=50.0)
        invoice = create_invoice_for_package(self.org, self.package, coupon=coupon)
        self.assertEqual(invoice.discount, Decimal("5000.00"))
        self.assertEqual(invoice.total, Decimal("5900.00"))

    def test_expired_coupon_gives_no_discount(self):
        coupon = Coupon.objects.create(
            code="EXPIRED", percent_off=50.0, valid_until=timezone.now() - timedelta(days=1)
        )
        self.assertFalse(coupon.is_redeemable)
        self.assertEqual(coupon.discount_for(Decimal("100")), Decimal("0.00"))

    def test_extra_cameras_are_billed(self):
        package = make_package(name="Metered", per_camera_inr=Decimal("500"))
        invoice = create_invoice_for_package(self.org, package, extra_cameras=4)
        self.assertEqual(invoice.subtotal, Decimal("12000.00"))
        self.assertEqual(len(invoice.line_items), 2)


@override_settings(RAZORPAY=RAZORPAY_TEST)
class RazorpaySignatureTests(TestCase):
    def setUp(self):
        self.gateway = get_gateway("razorpay")

    def _signature(self, order_id: str, payment_id: str) -> str:
        return hmac.new(
            RAZORPAY_TEST["KEY_SECRET"].encode(), f"{order_id}|{payment_id}".encode(), hashlib.sha256
        ).hexdigest()

    def test_valid_signature_is_accepted(self):
        self.assertTrue(self.gateway.verify({
            "razorpay_order_id": "order_A", "razorpay_payment_id": "pay_A",
            "razorpay_signature": self._signature("order_A", "pay_A"),
        }))

    def test_signature_for_a_different_payment_is_rejected(self):
        """The browser cannot be trusted to report a real payment."""
        self.assertFalse(self.gateway.verify({
            "razorpay_order_id": "order_A", "razorpay_payment_id": "pay_ATTACKER",
            "razorpay_signature": self._signature("order_A", "pay_A"),
        }))

    def test_missing_signature_is_rejected(self):
        self.assertFalse(self.gateway.verify({}))
        self.assertFalse(self.gateway.verify({"razorpay_order_id": "order_A"}))

    def test_garbage_signature_is_rejected(self):
        self.assertFalse(self.gateway.verify({
            "razorpay_order_id": "order_A", "razorpay_payment_id": "pay_A",
            "razorpay_signature": "0" * 64,
        }))


@override_settings(RAZORPAY=RAZORPAY_TEST)
class WebhookTests(TestCase):
    def setUp(self):
        self.org = make_org()
        self.package = make_package()
        self.invoice = create_invoice_for_package(self.org, self.package)
        self.payment = Payment.objects.create(
            organization=self.org, invoice=self.invoice, package=self.package,
            gateway="razorpay", currency="INR", amount=self.invoice.total,
            gateway_order_id="order_HOOK",
        )

    def _body(self, event_id="evt_1", event="payment.captured"):
        return json.dumps({
            "id": event_id, "event": event,
            "payload": {"payment": {"entity": {
                "id": "pay_HOOK", "order_id": "order_HOOK",
                "amount": 1180000, "currency": "INR", "status": "captured",
            }}},
        }).encode()

    def _sign(self, body):
        return hmac.new(RAZORPAY_TEST["WEBHOOK_SECRET"].encode(), body, hashlib.sha256).hexdigest()

    def test_signed_webhook_captures_the_payment_and_activates_the_plan(self):
        body = self._body()
        record = process_webhook("razorpay", body, self._sign(body))
        self.assertTrue(record.signature_valid)
        self.assertTrue(record.processed)
        self.payment.refresh_from_db()
        self.assertEqual(self.payment.status, Payment.Status.CAPTURED)
        self.invoice.refresh_from_db()
        self.assertTrue(self.invoice.is_paid)
        self.assertIsNotNone(self.org.refresh_subscription())

    def test_unsigned_webhook_is_stored_but_never_acted_upon(self):
        body = self._body("evt_unsigned")
        record = process_webhook("razorpay", body, "not-a-signature")
        self.assertFalse(record.signature_valid)
        self.assertFalse(record.processed)
        self.payment.refresh_from_db()
        self.assertEqual(self.payment.status, Payment.Status.CREATED)

    def test_replayed_webhook_is_processed_exactly_once(self):
        """Gateways retry aggressively; replaying would double-credit."""
        body = self._body("evt_replay")
        first = process_webhook("razorpay", body, self._sign(body))
        second = process_webhook("razorpay", body, self._sign(body))
        self.assertEqual(first.pk, second.pk)
        self.assertEqual(WebhookEvent.objects.filter(event_id="evt_replay").count(), 1)

    def test_failed_payment_marks_the_subscription_past_due(self):
        activate_subscription(self.org, self.package, currency="INR")
        body = json.dumps({
            "id": "evt_fail", "event": "payment.failed",
            "payload": {"payment": {"entity": {
                "id": "pay_F", "order_id": "order_HOOK", "amount": 1180000,
                "currency": "INR", "status": "failed",
            }}},
        }).encode()
        process_webhook("razorpay", body, self._sign(body))
        self.payment.refresh_from_db()
        self.assertEqual(self.payment.status, Payment.Status.FAILED)
        self.assertEqual(self.org.refresh_subscription().status, Subscription.Status.PAST_DUE)

    def test_webhook_endpoint_rejects_a_bad_signature(self):
        body = self._body("evt_endpoint")
        response = self.client.post(
            reverse("razorpay_webhook"), data=body, content_type="application/json",
            HTTP_X_RAZORPAY_SIGNATURE="wrong",
        )
        self.assertEqual(response.status_code, 400)

    def test_webhook_endpoint_accepts_a_good_signature(self):
        body = self._body("evt_endpoint_ok")
        response = self.client.post(
            reverse("razorpay_webhook"), data=body, content_type="application/json",
            HTTP_X_RAZORPAY_SIGNATURE=self._sign(body),
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["received"])


class UnconfiguredGatewayTests(TestCase):
    @override_settings(RAZORPAY={"KEY_ID": "", "KEY_SECRET": "", "WEBHOOK_SECRET": ""})
    def test_unconfigured_gateway_reports_itself_clearly(self):
        gateway = get_gateway("razorpay")
        self.assertFalse(gateway.is_configured)
        with self.assertRaises(GatewayError):
            _ = gateway.client

    def test_unknown_gateway_name_raises(self):
        with self.assertRaises(GatewayError):
            get_gateway("paypal")


class PaymentCompletionTests(TestCase):
    def test_completing_a_payment_settles_the_invoice_and_starts_the_plan(self):
        org = make_org()
        package = make_package()
        invoice = create_invoice_for_package(org, package)
        payment = Payment.objects.create(
            organization=org, invoice=invoice, package=package, gateway="razorpay",
            currency="INR", amount=invoice.total, gateway_order_id="order_X",
        )
        subscription = complete_payment(payment, "pay_X", {"ok": True})
        payment.refresh_from_db()
        invoice.refresh_from_db()
        self.assertEqual(payment.status, Payment.Status.CAPTURED)
        self.assertTrue(invoice.is_paid)
        self.assertEqual(subscription.status, Subscription.Status.ACTIVE)
        org.refresh_from_db()
        self.assertEqual(org.status, Organization.Status.ACTIVE)
