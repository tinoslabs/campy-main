"""Billing forms — package editor and checkout."""
from __future__ import annotations

from django import forms

from .models import FEATURE_KEYS, FEATURES, QUOTA_KEYS, QUOTAS, Coupon, Package


def _input(placeholder: str = "", **attrs):
    base = {"class": "input"}
    if placeholder:
        base["placeholder"] = placeholder
    base.update(attrs)
    return base


class PackageForm(forms.ModelForm):
    """Super-admin package editor. Features and quotas come from the POST body
    rather than declared fields, because both lists are data-driven."""

    class Meta:
        model = Package
        fields = [
            "name", "tagline", "description",
            "price_inr", "price_usd", "interval", "trial_days",
            "setup_fee_inr", "setup_fee_usd", "per_camera_inr", "per_camera_usd",
            "audience", "is_active", "is_featured", "sort_order", "badge", "colour",
            "razorpay_plan_id", "stripe_price_id",
        ]
        widgets = {
            "name": forms.TextInput(attrs=_input("Professional", autofocus=True)),
            "tagline": forms.TextInput(attrs=_input("For growing multi-site operations")),
            "description": forms.Textarea(attrs=_input(rows=3)),
            "price_inr": forms.NumberInput(attrs=_input(step="0.01", min="0")),
            "price_usd": forms.NumberInput(attrs=_input(step="0.01", min="0")),
            "interval": forms.Select(attrs={"class": "input"}),
            "trial_days": forms.NumberInput(attrs=_input(min="0", max="90")),
            "setup_fee_inr": forms.NumberInput(attrs=_input(step="0.01", min="0")),
            "setup_fee_usd": forms.NumberInput(attrs=_input(step="0.01", min="0")),
            "per_camera_inr": forms.NumberInput(attrs=_input(step="0.01", min="0")),
            "per_camera_usd": forms.NumberInput(attrs=_input(step="0.01", min="0")),
            "audience": forms.Select(attrs={"class": "input"}),
            "sort_order": forms.NumberInput(attrs=_input(min="0")),
            "badge": forms.TextInput(attrs=_input("Most popular")),
            "colour": forms.TextInput(attrs={"class": "input", "type": "color"}),
            "razorpay_plan_id": forms.TextInput(attrs=_input("plan_XXXXXXXXXXXX")),
            "stripe_price_id": forms.TextInput(attrs=_input("price_XXXXXXXXXXXX")),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_choices = FEATURES
        self.quota_choices = QUOTAS
        self.fields["per_camera_inr"].help_text = "Charged per camera above the included quota."
        self.fields["badge"].help_text = "Optional ribbon on the pricing page."

    def save(self, commit=True):
        package = super().save(commit=False)
        data = self.data
        package.features = {key: (f"feature_{key}" in data) for key in FEATURE_KEYS}
        quotas = {}
        for key in QUOTA_KEYS:
            raw = data.get(f"quota_{key}", "")
            try:
                quotas[key] = int(raw)
            except (TypeError, ValueError):
                quotas[key] = 0
        package.quotas = quotas
        if commit:
            package.save()
        return package


class CheckoutForm(forms.Form):
    gateway = forms.ChoiceField(choices=[], widget=forms.RadioSelect)
    currency = forms.ChoiceField(
        choices=[("INR", "Indian Rupee (₹)"), ("USD", "US Dollar ($)")],
        widget=forms.Select(attrs={"class": "input"}),
    )
    extra_cameras = forms.IntegerField(
        required=False, min_value=0, max_value=500, initial=0,
        widget=forms.NumberInput(attrs=_input("0")),
        label="Additional cameras",
    )
    coupon_code = forms.CharField(
        required=False, widget=forms.TextInput(attrs=_input("Discount code")), label="Coupon"
    )

    def __init__(self, organization, package, *args, **kwargs):
        self.organization = organization
        self.package = package
        super().__init__(*args, **kwargs)

        from .gateways import available_gateways, default_gateway_for

        gateways = available_gateways()
        self.fields["gateway"].choices = [(g.key, g.label) for g in gateways]
        if not gateways:
            # Nothing configured — offer an offline/invoice route rather than a dead end.
            self.fields["gateway"].choices = [("manual", "Request an invoice")]

        default_currency = "INR" if (organization.country or "IN").upper() == "IN" else "USD"
        self.fields["currency"].initial = default_currency
        self.fields["gateway"].initial = default_gateway_for(default_currency)
        if not package.per_camera_inr and not package.per_camera_usd:
            del self.fields["extra_cameras"]

    def clean_coupon_code(self):
        code = (self.cleaned_data.get("coupon_code") or "").strip().upper()
        if not code:
            return None
        coupon = Coupon.objects.filter(code__iexact=code).first()
        if coupon is None or not coupon.is_redeemable:
            raise forms.ValidationError("That coupon code is not valid.")
        if coupon.packages.exists() and not coupon.packages.filter(pk=self.package.pk).exists():
            raise forms.ValidationError("That coupon does not apply to this package.")
        return coupon
