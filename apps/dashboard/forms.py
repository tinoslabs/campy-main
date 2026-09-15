"""Forms for the operator dashboard."""
from __future__ import annotations

from django import forms

from apps.aiengine.detectors import ANALYTIC_CHOICES, analytic_metadata
from apps.aiengine.registry import MODEL_SLOTS
from apps.analytics.models import Report
from apps.cameras.models import Camera, Employee, Site, Zone
from apps.events.models import AlertRule, NotificationChannel
from apps.training.models import Dataset, TrainingJob
from apps.cameras.validators import clean_zone_polygon


def _input(placeholder: str = "", **attrs):
    base = {"class": "input"}
    if placeholder:
        base["placeholder"] = placeholder
    base.update(attrs)
    return base


class SiteForm(forms.ModelForm):
    class Meta:
        model = Site
        fields = [
            "name", "code", "description", "address", "city", "country",
            "latitude", "longitude", "timezone", "contact_name", "contact_phone", "is_active",
        ]
        widgets = {
            "name": forms.TextInput(attrs=_input("Bengaluru HQ — 3rd Floor", autofocus=True)),
            "code": forms.TextInput(attrs=_input("BLR-01")),
            "description": forms.Textarea(attrs=_input("What this location is used for", rows=2)),
            "address": forms.TextInput(attrs=_input()),
            "city": forms.TextInput(attrs=_input()),
            "country": forms.TextInput(attrs=_input("IN", maxlength="2")),
            "latitude": forms.NumberInput(attrs=_input(step="0.000001")),
            "longitude": forms.NumberInput(attrs=_input(step="0.000001")),
            "timezone": forms.TextInput(attrs=_input("Asia/Kolkata")),
            "contact_name": forms.TextInput(attrs=_input("Site manager")),
            "contact_phone": forms.TextInput(attrs=_input()),
        }


class CameraForm(forms.ModelForm):
    """Camera setup, including which analytics run on it."""

    enabled_analytics = forms.MultipleChoiceField(
        choices=ANALYTIC_CHOICES,
        required=False,
        widget=forms.CheckboxSelectMultiple,
        label="Analytics to run on this camera",
    )
    demo_scenario = forms.ChoiceField(
        required=False,
        choices=[
            ("walk", "Person walking through"),
            ("idle", "Person standing idle (loitering)"),
            ("crowd", "Crowd gathering"),
            ("fire", "Fire outbreak"),
            ("abandoned", "Object left unattended"),
            ("restricted", "Entry into a restricted zone"),
            ("running", "Person running"),
        ],
        widget=forms.Select(attrs={"class": "input"}),
        help_text="Which scene a simulated camera plays. Useful for demos and testing alert rules.",
    )

    class Meta:
        model = Camera
        fields = [
            "site", "name", "location_note", "description",
            "protocol", "stream_url", "username", "password", "onvif_host", "onvif_port",
            "resolution_width", "resolution_height", "source_fps", "target_fps", "rotation",
            "record_evidence", "evidence_seconds", "privacy_blur_faces", "is_active", "thumbnail",
        ]
        widgets = {
            "site": forms.Select(attrs={"class": "input"}),
            "name": forms.TextInput(attrs=_input("Main entrance", autofocus=True)),
            "location_note": forms.TextInput(attrs=_input("Above the reception desk, facing the door")),
            "description": forms.Textarea(attrs=_input(rows=2)),
            "protocol": forms.Select(attrs={"class": "input"}),
            "stream_url": forms.TextInput(attrs=_input("rtsp://192.168.1.50:554/stream1")),
            "username": forms.TextInput(attrs=_input("Camera login", autocomplete="off")),
            "password": forms.PasswordInput(
                attrs=_input("Camera password", autocomplete="new-password"), render_value=True
            ),
            "onvif_host": forms.TextInput(attrs=_input("192.168.1.50")),
            "onvif_port": forms.NumberInput(attrs=_input()),
            "resolution_width": forms.NumberInput(attrs=_input()),
            "resolution_height": forms.NumberInput(attrs=_input()),
            "source_fps": forms.NumberInput(attrs=_input(step="0.1")),
            "target_fps": forms.NumberInput(attrs=_input(step="0.5", min="0.5", max="30")),
            "rotation": forms.Select(attrs={"class": "input"}),
            "evidence_seconds": forms.NumberInput(attrs=_input(min="0", max="120")),
        }

    def __init__(self, organization, *args, **kwargs):
        self.organization = organization
        super().__init__(*args, **kwargs)
        self.fields["site"].queryset = Site.objects.filter(organization=organization, is_active=True)
        self.fields["target_fps"].help_text = (
            "Frames analysed per second. 4–8 suits most indoor scenes; higher costs more CPU."
        )
        self.fields["stream_url"].required = False
        self.analytic_meta = {item["key"]: item for item in analytic_metadata()}

        # Show which analytics the current package actually allows.
        from apps.aiengine.detectors import ANALYTIC_FEATURES

        self.blocked = {
            key for key, feature in ANALYTIC_FEATURES.items()
            if feature and not organization.has_feature(feature)
        }
        if self.instance and self.instance.pk:
            self.fields["enabled_analytics"].initial = self.instance.enabled_analytics
            self.fields["demo_scenario"].initial = (self.instance.analytics_config or {}).get(
                "demo_scenario", "walk"
            )

    def clean(self):
        cleaned = super().clean()
        protocol = cleaned.get("protocol")
        url = (cleaned.get("stream_url") or "").strip()
        if protocol != Camera.Protocol.DEMO and not url:
            self.add_error("stream_url", "A stream URL is required for this protocol.")
        if protocol == Camera.Protocol.RTSP and url and not url.startswith("rtsp"):
            self.add_error("stream_url", "An RTSP stream URL should start with rtsp:// or rtsps://.")

        cleaned["stream_url"] = self.tidy_credentials(cleaned, url)

        chosen = set(cleaned.get("enabled_analytics") or [])
        unavailable = chosen & self.blocked
        if unavailable:
            labels = dict(ANALYTIC_CHOICES)
            self.add_error(
                "enabled_analytics",
                "Your current package does not include: "
                + ", ".join(labels.get(key, key) for key in sorted(unavailable))
                + ". Upgrade to enable them.",
            )
        return cleaned

    def tidy_credentials(self, cleaned, url: str) -> str:
        """Keep the login in the credential fields rather than inside the URL.

        Vendors hand out the address as one string —
        ``rtsp://admin:Tinos@1122@192.168.1.50:554/Streaming/Channels/501`` — so
        that is what gets pasted, and then the username and password boxes get
        filled in too. Injecting a second copy at connect time sends
        ``admin:pass@admin:pass`` as the password and the camera answers 401,
        with nothing on screen to say which half was wrong.

        Splitting them here means the stored URL has one meaning, the password
        lives in the password field, and no page ever renders it.
        """
        from apps.cameras.urls_util import has_credentials, split_credentials

        if not url or not has_credentials(url):
            return url

        clean_url, username, password = split_credentials(url)
        # Anything typed into the fields is the more deliberate answer; fall
        # back to whatever the pasted URL carried.
        cleaned["username"] = (cleaned.get("username") or "").strip() or username
        if not (cleaned.get("password") or "").strip():
            cleaned["password"] = password
        return clean_url

    def save(self, commit=True):
        camera = super().save(commit=False)
        camera.organization = self.organization
        camera.enabled_analytics = list(self.cleaned_data.get("enabled_analytics") or [])
        config = dict(camera.analytics_config or {})
        if camera.protocol == Camera.Protocol.DEMO:
            config["demo_scenario"] = self.cleaned_data.get("demo_scenario") or "walk"
        camera.analytics_config = config
        if commit:
            camera.save()
        return camera


class ZoneForm(forms.ModelForm):
    schedule_days = forms.MultipleChoiceField(
        required=False,
        choices=[(d, d.title()) for d in ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]],
        widget=forms.CheckboxSelectMultiple,
        label="Active on days",
    )
    schedule_from = forms.CharField(
        required=False, widget=forms.TimeInput(attrs=_input(type="time")), label="Armed from"
    )
    schedule_to = forms.CharField(
        required=False, widget=forms.TimeInput(attrs=_input(type="time")), label="Armed until"
    )

    class Meta:
        model = Zone
        fields = [
            "name", "kind", "severity", "polygon", "colour",
            "max_occupancy", "min_dwell_seconds", "is_active", "notes",
        ]
        widgets = {
            "name": forms.TextInput(attrs=_input("Server room doorway", autofocus=True)),
            "kind": forms.Select(attrs={"class": "input"}),
            "severity": forms.Select(attrs={"class": "input"}),
            "polygon": forms.HiddenInput(),
            "colour": forms.TextInput(attrs={"class": "input", "type": "color"}),
            "max_occupancy": forms.NumberInput(attrs=_input(min="0")),
            "min_dwell_seconds": forms.NumberInput(attrs=_input(min="0")),
            "notes": forms.Textarea(attrs=_input(rows=2)),
        }

    def __init__(self, camera, *args, **kwargs):
        self.camera = camera
        super().__init__(*args, **kwargs)
        self.fields["polygon"].required = False
        # The shape is drawn on the camera image and posted through a hidden
        # input, so nobody ever types it. Django's stock "Enter a valid JSON."
        # would be baffling on a control nobody can see.
        self.fields["polygon"].error_messages["invalid"] = (
            "The zone shape could not be read. Draw it again."
        )
        self.fields["max_occupancy"].help_text = "0 = no occupancy limit."
        self.fields["min_dwell_seconds"].help_text = "0 = alert as soon as somebody enters."
        if self.instance and self.instance.pk:
            schedule = self.instance.schedule or {}
            self.fields["schedule_days"].initial = schedule.get("days", [])
            self.fields["schedule_from"].initial = schedule.get("from", "")
            self.fields["schedule_to"].initial = schedule.get("to", "")

    def clean_polygon(self):
        polygon = self.cleaned_data.get("polygon")
        if isinstance(polygon, str):
            import json

            try:
                polygon = json.loads(polygon or "[]")
            except ValueError:
                raise forms.ValidationError("The zone shape could not be read. Redraw it.")
        return clean_zone_polygon(polygon, forms.ValidationError)

    def save(self, commit=True):
        zone = super().save(commit=False)
        zone.camera = self.camera
        days = self.cleaned_data.get("schedule_days") or []
        start = self.cleaned_data.get("schedule_from") or ""
        end = self.cleaned_data.get("schedule_to") or ""
        zone.schedule = {"days": days, "from": start, "to": end} if (days or (start and end)) else {}
        if commit:
            zone.save()
        return zone


class EmployeeForm(forms.ModelForm):
    class Meta:
        model = Employee
        fields = [
            "full_name", "employee_code", "role", "department", "email", "phone",
            "photo", "sites", "is_active", "consent_given", "notes",
        ]
        widgets = {
            "full_name": forms.TextInput(attrs=_input("Asha Menon", autofocus=True)),
            "employee_code": forms.TextInput(attrs=_input("EMP-0142")),
            "role": forms.TextInput(attrs=_input("Warehouse Supervisor")),
            "department": forms.TextInput(attrs=_input("Operations")),
            "email": forms.EmailInput(attrs=_input()),
            "phone": forms.TextInput(attrs=_input()),
            "sites": forms.SelectMultiple(attrs={"class": "input", "size": 5}),
            "notes": forms.Textarea(attrs=_input(rows=2)),
        }

    def __init__(self, organization, *args, **kwargs):
        self.organization = organization
        super().__init__(*args, **kwargs)
        self.fields["sites"].queryset = Site.objects.filter(organization=organization)
        self.fields["sites"].required = False
        self.fields["consent_given"].label = "This person has given written consent to face recognition"
        self.fields["consent_given"].help_text = (
            "Biometric processing requires explicit, documented consent in most jurisdictions. "
            "Face enrolment stays disabled until this is confirmed."
        )

    def save(self, commit=True):
        employee = super().save(commit=False)
        employee.organization = self.organization
        if employee.consent_given and not employee.consent_recorded_at:
            from django.utils import timezone

            employee.consent_recorded_at = timezone.now()
        if commit:
            employee.save()
            self.save_m2m()
        return employee


class AlertRuleForm(forms.ModelForm):
    analytics = forms.MultipleChoiceField(
        choices=ANALYTIC_CHOICES, required=False, widget=forms.CheckboxSelectMultiple,
        help_text="Leave all unchecked to match every analytic.",
    )
    schedule_days = forms.MultipleChoiceField(
        required=False,
        choices=[(d, d.title()) for d in ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]],
        widget=forms.CheckboxSelectMultiple, label="Only on days",
    )
    schedule_from = forms.CharField(required=False, widget=forms.TimeInput(attrs=_input(type="time")))
    schedule_to = forms.CharField(required=False, widget=forms.TimeInput(attrs=_input(type="time")))

    class Meta:
        model = AlertRule
        fields = [
            "name", "description", "min_severity", "min_confidence",
            "sites", "cameras", "zones", "channels", "recipients",
            "cooldown_seconds", "threshold_count", "threshold_window_seconds",
            "escalate_after_seconds", "escalation_channels",
            "create_incident", "auto_acknowledge", "is_active", "priority",
        ]
        widgets = {
            "name": forms.TextInput(attrs=_input("Critical fire alerts → security team", autofocus=True)),
            "description": forms.Textarea(attrs=_input(rows=2)),
            "min_severity": forms.Select(attrs={"class": "input"}),
            "min_confidence": forms.NumberInput(attrs=_input(step="0.05", min="0", max="1")),
            "sites": forms.SelectMultiple(attrs={"class": "input", "size": 4}),
            "cameras": forms.SelectMultiple(attrs={"class": "input", "size": 6}),
            "zones": forms.SelectMultiple(attrs={"class": "input", "size": 4}),
            "channels": forms.CheckboxSelectMultiple,
            "escalation_channels": forms.CheckboxSelectMultiple,
            "recipients": forms.SelectMultiple(attrs={"class": "input", "size": 5}),
            "cooldown_seconds": forms.NumberInput(attrs=_input(min="0")),
            "threshold_count": forms.NumberInput(attrs=_input(min="1")),
            "threshold_window_seconds": forms.NumberInput(attrs=_input(min="10")),
            "escalate_after_seconds": forms.NumberInput(attrs=_input(min="0")),
            "priority": forms.NumberInput(attrs=_input(min="0")),
        }

    def __init__(self, organization, *args, **kwargs):
        self.organization = organization
        super().__init__(*args, **kwargs)
        from apps.accounts.models import User

        self.fields["sites"].queryset = Site.objects.filter(organization=organization)
        self.fields["cameras"].queryset = Camera.objects.filter(organization=organization)
        self.fields["zones"].queryset = Zone.objects.filter(camera__organization=organization)
        self.fields["channels"].queryset = NotificationChannel.objects.filter(
            organization=organization, is_active=True
        )
        self.fields["escalation_channels"].queryset = self.fields["channels"].queryset
        self.fields["recipients"].queryset = User.objects.filter(
            memberships__organization=organization, memberships__status="active"
        ).distinct()
        for name in ("sites", "cameras", "zones", "channels", "recipients", "escalation_channels"):
            self.fields[name].required = False
        self.fields["cooldown_seconds"].help_text = "Suppress repeats of this rule for this long."
        self.fields["escalate_after_seconds"].help_text = (
            "0 disables escalation. Otherwise, unacknowledged events escalate after this long."
        )
        if self.instance and self.instance.pk:
            self.fields["analytics"].initial = self.instance.analytics
            schedule = self.instance.active_schedule or {}
            self.fields["schedule_days"].initial = schedule.get("days", [])
            self.fields["schedule_from"].initial = schedule.get("from", "")
            self.fields["schedule_to"].initial = schedule.get("to", "")

    def save(self, commit=True):
        rule = super().save(commit=False)
        rule.organization = self.organization
        rule.analytics = list(self.cleaned_data.get("analytics") or [])
        days = self.cleaned_data.get("schedule_days") or []
        start = self.cleaned_data.get("schedule_from") or ""
        end = self.cleaned_data.get("schedule_to") or ""
        rule.active_schedule = {"days": days, "from": start, "to": end} if (days or (start and end)) else {}
        if commit:
            rule.save()
            self.save_m2m()
        return rule


class NotificationChannelForm(forms.ModelForm):
    emails = forms.CharField(
        required=False, widget=forms.Textarea(attrs=_input("ops@company.com, security@company.com", rows=2)),
        label="Email recipients", help_text="Comma or newline separated.",
    )
    url = forms.URLField(
        required=False, assume_scheme="https",
        widget=forms.URLInput(attrs=_input("https://hooks.example.com/…")),
    )
    secret = forms.CharField(
        required=False, widget=forms.TextInput(attrs=_input("Shared secret for HMAC signing")),
        help_text="If set, deliveries carry an X-Campy-Signature header.",
    )
    bot_token = forms.CharField(required=False, widget=forms.TextInput(attrs=_input()))
    chat_id = forms.CharField(required=False, widget=forms.TextInput(attrs=_input()))
    numbers = forms.CharField(
        required=False, widget=forms.TextInput(attrs=_input("+919876543210, +919876543211")),
        label="Phone numbers",
    )
    gateway_url = forms.URLField(
        required=False, assume_scheme="https", widget=forms.URLInput(attrs=_input())
    )

    class Meta:
        model = NotificationChannel
        fields = ["name", "kind", "include_snapshot", "is_active"]
        widgets = {
            "name": forms.TextInput(attrs=_input("Security team email", autofocus=True)),
            "kind": forms.Select(attrs={"class": "input"}),
        }

    def __init__(self, organization, *args, **kwargs):
        self.organization = organization
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.pk:
            config = self.instance.config or {}
            self.fields["emails"].initial = ", ".join(config.get("emails", []))
            self.fields["url"].initial = config.get("url", "")
            self.fields["secret"].initial = config.get("secret", "")
            self.fields["bot_token"].initial = config.get("bot_token", "")
            self.fields["chat_id"].initial = config.get("chat_id", "")
            self.fields["numbers"].initial = ", ".join(config.get("numbers", []))
            self.fields["gateway_url"].initial = config.get("gateway_url", "")

    def clean(self):
        cleaned = super().clean()
        kind = cleaned.get("kind")
        if kind == NotificationChannel.Kind.EMAIL and not cleaned.get("emails"):
            self.add_error("emails", "Add at least one email address.")
        if kind in {NotificationChannel.Kind.WEBHOOK, NotificationChannel.Kind.SLACK} and not cleaned.get("url"):
            self.add_error("url", "A delivery URL is required for this channel type.")
        if kind == NotificationChannel.Kind.TELEGRAM and not (
            cleaned.get("bot_token") and cleaned.get("chat_id")
        ):
            self.add_error("bot_token", "Telegram needs both a bot token and a chat id.")
        if kind == NotificationChannel.Kind.SMS and not (
            cleaned.get("gateway_url") and cleaned.get("numbers")
        ):
            self.add_error("gateway_url", "SMS needs a gateway URL and at least one number.")
        return cleaned

    def save(self, commit=True):
        channel = super().save(commit=False)
        channel.organization = self.organization
        data = self.cleaned_data

        def split(value):
            return [part.strip() for part in (value or "").replace("\n", ",").split(",") if part.strip()]

        channel.config = {
            "emails": split(data.get("emails")),
            "url": data.get("url", ""),
            "secret": data.get("secret", ""),
            "bot_token": data.get("bot_token", ""),
            "chat_id": data.get("chat_id", ""),
            "numbers": split(data.get("numbers")),
            "gateway_url": data.get("gateway_url", ""),
        }
        if commit:
            channel.save()
        return channel


class DatasetForm(forms.ModelForm):
    class_names = forms.CharField(
        label="Classes",
        widget=forms.TextInput(attrs=_input("normal, fire, smoke")),
        help_text="Comma separated, in order. The first class is treated as the negative/background class.",
    )
    augment_flip = forms.BooleanField(required=False, initial=True, label="Random horizontal flip")
    augment_brightness = forms.FloatField(
        required=False, initial=0.2, widget=forms.NumberInput(attrs=_input(step="0.05", min="0", max="0.6")),
        label="Brightness jitter",
    )
    augment_noise = forms.FloatField(
        required=False, initial=0.03, widget=forms.NumberInput(attrs=_input(step="0.01", min="0", max="0.3")),
        label="Sensor noise",
    )
    augment_shift = forms.FloatField(
        required=False, initial=0.08, widget=forms.NumberInput(attrs=_input(step="0.02", min="0", max="0.3")),
        label="Random shift",
    )

    class Meta:
        model = Dataset
        fields = [
            "name", "description", "task", "image_size", "validation_split",
            "harvest_from_events",
        ]
        widgets = {
            "name": forms.TextInput(attrs=_input("Warehouse fire detector v1", autofocus=True)),
            "description": forms.Textarea(attrs=_input(rows=2)),
            "task": forms.Select(attrs={"class": "input"}),
            "image_size": forms.NumberInput(attrs=_input(min="32", max="224", step="8")),
            "validation_split": forms.NumberInput(attrs=_input(step="0.05", min="0.1", max="0.4")),
        }

    def __init__(self, organization, *args, **kwargs):
        self.organization = organization
        super().__init__(*args, **kwargs)
        self.fields["image_size"].help_text = "Square input size. 96 is a good default; larger costs more CPU."
        self.fields["harvest_from_events"].label = "Automatically add frames from dismissed false alarms"
        self.fields["harvest_from_events"].help_text = (
            "Every alert an operator marks as a false positive becomes a hard negative for the next run."
        )
        if self.instance and self.instance.pk:
            self.fields["class_names"].initial = ", ".join(self.instance.classes or [])
            augmentation = self.instance.augmentation or {}
            self.fields["augment_flip"].initial = augmentation.get("flip", True)
            self.fields["augment_brightness"].initial = augmentation.get("brightness", 0.2)
            self.fields["augment_noise"].initial = augmentation.get("noise", 0.03)
            self.fields["augment_shift"].initial = augmentation.get("shift", 0.08)

    def clean_class_names(self):
        names = [n.strip() for n in (self.cleaned_data["class_names"] or "").split(",") if n.strip()]
        if len(names) < 2:
            raise forms.ValidationError("A dataset needs at least two classes to train on.")
        if len(set(names)) != len(names):
            raise forms.ValidationError("Class names must be unique.")
        return names

    def save(self, commit=True):
        dataset = super().save(commit=False)
        dataset.organization = self.organization
        dataset.classes = self.cleaned_data["class_names"]
        dataset.augmentation = {
            "flip": bool(self.cleaned_data.get("augment_flip")),
            "brightness": float(self.cleaned_data.get("augment_brightness") or 0),
            "noise": float(self.cleaned_data.get("augment_noise") or 0),
            "shift": float(self.cleaned_data.get("augment_shift") or 0),
        }
        if commit:
            dataset.save()
        return dataset


class TrainingJobForm(forms.ModelForm):
    class Meta:
        model = TrainingJob
        fields = [
            "name", "dataset", "architecture", "target_slot",
            "epochs", "batch_size", "learning_rate", "optimizer", "schedule",
            "width", "depth", "dropout", "early_stopping_patience", "seed",
        ]
        widgets = {
            "name": forms.TextInput(attrs=_input("Fire detector — run 3", autofocus=True)),
            "dataset": forms.Select(attrs={"class": "input"}),
            "architecture": forms.Select(attrs={"class": "input"}),
            "target_slot": forms.Select(attrs={"class": "input"}),
            "epochs": forms.NumberInput(attrs=_input(min="1", max="300")),
            "batch_size": forms.NumberInput(attrs=_input(min="1", max="512")),
            "learning_rate": forms.NumberInput(attrs=_input(step="0.0001", min="0.00001", max="1")),
            "optimizer": forms.Select(attrs={"class": "input"}),
            "schedule": forms.Select(attrs={"class": "input"}),
            "width": forms.NumberInput(attrs=_input(step="0.25", min="0.25", max="3")),
            "depth": forms.NumberInput(attrs=_input(min="1", max="6")),
            "dropout": forms.NumberInput(attrs=_input(step="0.05", min="0", max="0.7")),
            "early_stopping_patience": forms.NumberInput(attrs=_input(min="0", max="50")),
            "seed": forms.NumberInput(attrs=_input()),
        }

    def __init__(self, organization, *args, **kwargs):
        self.organization = organization
        super().__init__(*args, **kwargs)
        self.fields["dataset"].queryset = Dataset.objects.filter(
            organization=organization
        ).exclude(status=Dataset.Status.ARCHIVED)
        self.fields["target_slot"].choices = [("", "— Do not deploy automatically —")] + list(MODEL_SLOTS)
        self.fields["target_slot"].required = False
        self.fields["width"].help_text = "Channel multiplier: higher is more accurate but slower."
        self.fields["depth"].help_text = "Number of downsampling stages in the backbone."
        self.fields["early_stopping_patience"].help_text = (
            "Stop after this many epochs without improvement. 0 disables it."
        )

    def clean(self):
        cleaned = super().clean()
        dataset = cleaned.get("dataset")
        if dataset is not None and not dataset.is_trainable:
            readiness = dataset.readiness
            self.add_error(
                "dataset",
                f"'{dataset.name}' is not ready: {readiness['labelled']} labelled samples of "
                f"{readiness['required']} needed across at least 2 classes.",
            )
        return cleaned

    def save(self, commit=True):
        job = super().save(commit=False)
        job.organization = self.organization
        if commit:
            job.save()
        return job


class ReportForm(forms.ModelForm):
    analytics = forms.MultipleChoiceField(
        choices=ANALYTIC_CHOICES, required=False, widget=forms.CheckboxSelectMultiple
    )
    recipient_emails = forms.CharField(
        required=False, widget=forms.Textarea(attrs=_input("ops@company.com", rows=2)),
        label="Email the report to",
    )

    class Meta:
        model = Report
        fields = ["name", "kind", "frequency", "output_format", "sites", "cameras", "date_range_days", "is_active"]
        widgets = {
            "name": forms.TextInput(attrs=_input("Weekly safety summary", autofocus=True)),
            "kind": forms.Select(attrs={"class": "input"}),
            "frequency": forms.Select(attrs={"class": "input"}),
            "output_format": forms.Select(attrs={"class": "input"}),
            "sites": forms.SelectMultiple(attrs={"class": "input", "size": 4}),
            "cameras": forms.SelectMultiple(attrs={"class": "input", "size": 6}),
            "date_range_days": forms.NumberInput(attrs=_input(min="1", max="365")),
        }

    def __init__(self, organization, *args, **kwargs):
        self.organization = organization
        super().__init__(*args, **kwargs)
        self.fields["sites"].queryset = Site.objects.filter(organization=organization)
        self.fields["cameras"].queryset = Camera.objects.filter(organization=organization)
        self.fields["sites"].required = False
        self.fields["cameras"].required = False
        if self.instance and self.instance.pk:
            self.fields["analytics"].initial = self.instance.analytics
            self.fields["recipient_emails"].initial = ", ".join(self.instance.recipients or [])

    def save(self, commit=True):
        report = super().save(commit=False)
        report.organization = self.organization
        report.analytics = list(self.cleaned_data.get("analytics") or [])
        report.recipients = [
            part.strip()
            for part in (self.cleaned_data.get("recipient_emails") or "").replace("\n", ",").split(",")
            if part.strip()
        ]
        if commit:
            report.save()
            self.save_m2m()
        return report


class APIKeyForm(forms.Form):
    name = forms.CharField(
        max_length=120, widget=forms.TextInput(attrs=_input("Edge box — Bengaluru", autofocus=True))
    )
    scopes = forms.MultipleChoiceField(
        choices=[], widget=forms.CheckboxSelectMultiple, label="Permissions granted to this key"
    )
    expires_in_days = forms.IntegerField(
        required=False, min_value=1, max_value=3650,
        widget=forms.NumberInput(attrs=_input("365")),
        help_text="Leave blank for a key that never expires (not recommended).",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from apps.accounts.permissions import PERMISSIONS

        self.fields["scopes"].choices = [
            (p.code, f"{p.label} ({p.code})") for p in PERMISSIONS if not p.code.startswith("platform.")
        ]


class EventFilterForm(forms.Form):
    """Filter bar on the events list."""

    q = forms.CharField(required=False, widget=forms.TextInput(attrs=_input("Search events…")))
    severity = forms.ChoiceField(required=False, choices=[], widget=forms.Select(attrs={"class": "input"}))
    analytic = forms.ChoiceField(required=False, choices=[], widget=forms.Select(attrs={"class": "input"}))
    status = forms.ChoiceField(required=False, choices=[], widget=forms.Select(attrs={"class": "input"}))
    camera = forms.ChoiceField(required=False, choices=[], widget=forms.Select(attrs={"class": "input"}))
    days = forms.ChoiceField(
        required=False,
        choices=[("1", "Last 24 hours"), ("7", "Last 7 days"), ("30", "Last 30 days"), ("90", "Last 90 days"), ("", "All time")],
        initial="7",
        widget=forms.Select(attrs={"class": "input"}),
    )

    def __init__(self, organization, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from apps.events.models import SEVERITY_CHOICES, Event

        self.fields["severity"].choices = [("", "Any severity")] + list(SEVERITY_CHOICES)
        self.fields["analytic"].choices = [("", "All analytics")] + list(ANALYTIC_CHOICES)
        self.fields["status"].choices = [("", "Any status")] + list(Event.Status.choices)
        self.fields["camera"].choices = [("", "All cameras")] + [
            (str(c.pk), c.name) for c in Camera.objects.filter(organization=organization)[:200]
        ]
