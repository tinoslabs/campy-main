"""Billing, checkout and gateway webhooks."""
from __future__ import annotations

import json
import logging

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseBadRequest, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from apps.accounts.access import require_organization, require_perm
from apps.accounts.models import AuditLog

from .forms import CheckoutForm
from .gateways import GatewayError, available_gateways, get_gateway
from .middleware import quota_usage
from .models import FEATURES, QUOTAS, Invoice, Package, Payment
from .services import (
    cancel_subscription,
    complete_payment,
    create_invoice_for_package,
    start_checkout,
    start_trial,
)

logger = logging.getLogger("campy.billing")


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------
@login_required
@require_organization
@require_perm("billing.view")
def overview(request):
    organization = request.organization
    subscription = organization.active_subscription
    return render(
        request,
        "billing/overview.html",
        {
            "active": "billing",
            "subscription": subscription,
            "package": subscription.package if subscription else None,
            "quotas": quota_usage(organization),
            "quota_labels": dict(QUOTAS),
            "feature_labels": dict(FEATURES),
            "invoices": Invoice.objects.filter(organization=organization)[:12],
            "payments": Payment.objects.filter(organization=organization).select_related("package")[:10],
            "can_manage": request.user.has_campy_perm("billing.manage", organization),
        },
    )


@login_required
@require_organization
@require_perm("billing.view")
def packages(request):
    organization = request.organization
    subscription = organization.active_subscription
    current = subscription.package if subscription else None

    return render(
        request,
        "billing/packages.html",
        {
            "active": "billing",
            "packages": Package.objects.filter(
                is_active=True, audience=Package.Audience.PUBLIC
            ).order_by("sort_order", "price_inr"),
            "current_package": current,
            "subscription": subscription,
            "feature_labels": dict(FEATURES),
            "quota_labels": dict(QUOTAS),
            "currency": "INR" if (organization.country or "IN").upper() == "IN" else "USD",
            "can_manage": request.user.has_campy_perm("billing.manage", organization),
        },
    )


@login_required
@require_organization
@require_perm("billing.manage")
def checkout(request, slug):
    organization = request.organization
    package = get_object_or_404(Package, slug=slug, is_active=True)
    form = CheckoutForm(organization, package, request.POST or None)

    if request.method == "POST" and form.is_valid():
        gateway_key = form.cleaned_data["gateway"]
        currency = form.cleaned_data["currency"]
        coupon = form.cleaned_data.get("coupon_code")
        extra = form.cleaned_data.get("extra_cameras") or 0

        invoice = create_invoice_for_package(
            organization, package, currency=currency, coupon=coupon,
            extra_cameras=extra, user=request.user,
        )

        # A free plan (or a coupon that zeroes the bill) activates immediately.
        if invoice.total <= 0:
            from .services import activate_subscription

            activate_subscription(organization, package, currency=currency, user=request.user)
            invoice.mark_paid(0)
            messages.success(request, f"You are now on the {package.name} package.")
            return redirect("billing:overview")

        if gateway_key == "manual":
            messages.success(
                request,
                f"Invoice {invoice.number} has been raised. Our team will be in touch to "
                "complete payment.",
            )
            return redirect("billing:invoice_detail", uid=invoice.uid)

        try:
            payment, intent = start_checkout(invoice, gateway_key, request, request.user)
        except GatewayError as exc:
            messages.error(request, str(exc))
            return redirect("billing:packages")

        if intent.mode == "redirect" and intent.redirect_url:
            return redirect(intent.redirect_url)

        return render(
            request,
            "billing/checkout_razorpay.html",
            {
                "active": "billing", "invoice": invoice, "payment": payment,
                "package": package, "options": json.dumps(intent.payload),
                "public_key": intent.public_key,
            },
        )

    from decimal import Decimal

    currency = "INR" if (organization.country or "IN").upper() == "IN" else "USD"
    return render(
        request,
        "billing/checkout.html",
        {
            "active": "billing", "package": package, "form": form,
            "gateways": available_gateways(),
            "price": package.price_for(currency),
            "setup_fee": package.setup_fee_for(currency),
            "currency": currency,
            "symbol": "₹" if currency == "INR" else "$",
            "tax_rate": Decimal("18.0") if (organization.country or "IN").upper() == "IN" else Decimal("0"),
            "feature_labels": dict(FEATURES),
        },
    )


@login_required
@require_organization
@require_perm("billing.manage")
@require_POST
def razorpay_confirm(request, uid):
    """Browser callback from Razorpay Checkout — verified server-side."""
    organization = request.organization
    payment = get_object_or_404(Payment, uid=uid, organization=organization)
    gateway = get_gateway("razorpay")

    payload = {
        "razorpay_order_id": request.POST.get("razorpay_order_id", ""),
        "razorpay_payment_id": request.POST.get("razorpay_payment_id", ""),
        "razorpay_signature": request.POST.get("razorpay_signature", ""),
    }
    if not gateway.verify(payload):
        payment.status = Payment.Status.FAILED
        payment.error_message = "Signature verification failed"
        payment.raw_response = payload
        payment.save(update_fields=["status", "error_message", "raw_response", "updated_at"])
        AuditLog.record(
            action=AuditLog.Action.SECURITY, actor=request.user, organization=organization,
            summary=f"Razorpay signature verification failed for payment {payment.uid}",
            request=request, sensitive=True,
        )
        messages.error(
            request, "We could not verify that payment with Razorpay. No charge has been applied."
        )
        return redirect("billing:packages")

    payment.gateway_signature = payload["razorpay_signature"]
    payment.save(update_fields=["gateway_signature", "updated_at"])
    complete_payment(payment, payload["razorpay_payment_id"], payload, request.user)
    messages.success(request, "Payment received — your package is active. Thank you.")
    return redirect("billing:overview")


@login_required
@require_organization
def checkout_success(request, uid):
    """Stripe's return URL. The session is confirmed with Stripe, not trusted."""
    invoice = get_object_or_404(Invoice, uid=uid, organization=request.organization)
    session_id = request.GET.get("session_id")
    payment = Payment.objects.filter(invoice=invoice).order_by("-created_at").first()

    if payment and payment.status != Payment.Status.CAPTURED and session_id:
        gateway = get_gateway("stripe")
        if gateway.verify({"session_id": session_id}):
            complete_payment(payment, session_id, {"session_id": session_id}, request.user)
            messages.success(request, "Payment received — your package is active. Thank you.")
        else:
            messages.warning(
                request,
                "Stripe has not confirmed this payment yet. It will activate automatically "
                "once confirmation arrives.",
            )
    elif payment and payment.status == Payment.Status.CAPTURED:
        messages.success(request, "This payment is already confirmed.")

    return redirect("billing:overview")


@login_required
@require_organization
def checkout_cancelled(request, uid):
    invoice = get_object_or_404(Invoice, uid=uid, organization=request.organization)
    messages.info(request, f"Checkout was cancelled. Invoice {invoice.number} is still open.")
    return redirect("billing:packages")


@login_required
@require_organization
@require_perm("billing.view")
def invoices(request):
    return render(
        request,
        "billing/invoices.html",
        {
            "active": "billing",
            "invoices": Invoice.objects.filter(organization=request.organization),
        },
    )


@login_required
@require_organization
@require_perm("billing.invoices")
def invoice_detail(request, uid):
    invoice = get_object_or_404(
        Invoice.objects.select_related("package", "subscription"),
        uid=uid, organization=request.organization,
    )
    return render(
        request,
        "billing/invoice_detail.html",
        {
            "active": "billing", "invoice": invoice,
            "payments": invoice.payments.all(),
            "organization": request.organization,
        },
    )


@login_required
@require_organization
@require_perm("billing.manage")
@require_POST
def cancel(request):
    at_period_end = request.POST.get("immediately") != "1"
    subscription = cancel_subscription(request.organization, at_period_end, request.user)
    if subscription is None:
        messages.error(request, "There is no active subscription to cancel.")
    elif at_period_end:
        messages.success(
            request,
            f"Your plan will not renew. Monitoring continues until "
            f"{subscription.current_period_end:%d %b %Y}.",
        )
    else:
        messages.success(request, "Your subscription has been cancelled.")
    return redirect("billing:overview")


@login_required
@require_organization
@require_perm("billing.manage")
@require_POST
def start_trial_view(request, slug):
    package = get_object_or_404(Package, slug=slug, is_active=True)
    if request.organization.subscriptions.exists():
        messages.error(request, "This workspace has already used its trial.")
        return redirect("billing:packages")
    start_trial(request.organization, package, request.user)
    messages.success(request, f"Your {package.trial_days}-day trial of {package.name} has started.")
    return redirect("dashboard:onboarding")


# ---------------------------------------------------------------------------
# Webhooks
# ---------------------------------------------------------------------------
@csrf_exempt
@require_POST
def razorpay_webhook(request):
    """Razorpay server-to-server callback."""
    from .services import process_webhook

    signature = request.headers.get("X-Razorpay-Signature", "")
    try:
        record = process_webhook("razorpay", request.body, signature)
    except Exception:  # noqa: BLE001
        logger.exception("Razorpay webhook processing failed")
        return HttpResponse(status=500)

    if not record.signature_valid:
        return HttpResponseBadRequest("invalid signature")
    return JsonResponse({"received": True, "processed": record.processed})


@csrf_exempt
@require_POST
def stripe_webhook(request):
    """Stripe server-to-server callback."""
    from .services import process_webhook

    signature = request.headers.get("Stripe-Signature", "")
    try:
        record = process_webhook("stripe", request.body, signature)
    except Exception:  # noqa: BLE001
        logger.exception("Stripe webhook processing failed")
        return HttpResponse(status=500)

    if not record.signature_valid:
        return HttpResponseBadRequest("invalid signature")
    return JsonResponse({"received": True, "processed": record.processed})


def package_detail(request, slug):
    """Public package page — also reachable from the marketing site."""
    package = get_object_or_404(Package, slug=slug, is_active=True)
    return render(
        request,
        "billing/package_detail.html",
        {
            "package": package,
            "feature_labels": dict(FEATURES),
            "quota_labels": dict(QUOTAS),
        },
    )
