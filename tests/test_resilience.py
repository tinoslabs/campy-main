"""Sweeps that walk the whole application instead of one view at a time.

Three bugs in this project reached a user because the suite tested views one at
a time, in the one state a freshly-seeded test produces: a form partial that
dropped every field, a page that died on a classifier's metrics, and a
last-seen write that only ran after five minutes. Each was invisible to a
targeted test and obvious to a sweep.

These are slower than the rest of the suite by design. They are the net that
catches the next one.
"""
from __future__ import annotations

import re
from datetime import timedelta
from decimal import Decimal

from django import forms as djforms
from django.test import TestCase
from django.urls import get_resolver
from django.utils import timezone

from apps.accounts.models import Membership, Organization, User
from apps.analytics.models import Report
from apps.billing.models import Package, Subscription
from apps.billing.services import activate_subscription
from apps.cameras.models import Camera, Employee, Site, Zone
from apps.events.models import AlertRule, Event, Incident, NotificationChannel
from apps.training.models import Dataset, ModelVersion, TrainingJob

SKIP = ("logout", "webhooks/", "admin/", "healthz", "jsi18n", "media/", "static/",
        "/delete/", "/remove/", "revoke")
ALL_FEATURES = {key: True for key in (
    "gesture_tracking face_recognition geofencing crowd_management theft_detection "
    "object_detection fire_detection custom_training reports api_access audit_log "
    "evidence_clips multi_site sso white_label priority_support").split()}


def all_routes():
    found = []

    def walk(resolver, prefix=""):
        for pattern in resolver.url_patterns:
            if hasattr(pattern, "url_patterns"):
                walk(pattern, prefix + str(pattern.pattern))
            else:
                found.append(prefix + str(pattern.pattern))

    walk(get_resolver())
    return sorted(set(found))


ROUTES = all_routes()


def workspace(name, *, features=None):
    package = Package.objects.create(
        name=f"{name} Plan", price_inr=Decimal("1000"), features=features or ALL_FEATURES,
        quotas={"cameras": 50, "sites": 10, "users": 20, "employees": 500,
                "retention_days": 90, "training_jobs_month": 10, "api_calls_day": 1000},
    )
    organization = Organization.objects.create(name=name, country="IN", status="active")
    activate_subscription(organization, package, currency="INR")
    owner = User.objects.create_user(
        email=f"owner@{re.sub(r'[^a-z0-9]', '', name.lower())}.test",
        password="TestPass!2024", full_name="Owner")
    Membership.objects.create(user=owner, organization=organization,
                              role=organization.roles.get(code="owner"))
    owner.active_organization = organization
    owner.save(update_fields=["active_organization"])
    return organization, owner


class SweepMixin:
    def identifiers(self, organization):
        def first(queryset):
            obj = queryset.first()
            return str(obj.uid) if obj is not None else ""

        return {
            "cameras/": first(Camera.objects.filter(organization=organization)),
            "events/": first(Event.objects.filter(organization=organization)),
            "incidents/": first(Incident.objects.filter(organization=organization)),
            "sites/": first(Site.objects.filter(organization=organization)),
            "zones/": first(Zone.objects.filter(camera__organization=organization)),
            "employees/": first(Employee.objects.filter(organization=organization)),
            "datasets/": first(Dataset.objects.filter(organization=organization)),
            "training/": first(TrainingJob.objects.filter(organization=organization)),
            "models/": first(ModelVersion.objects.filter(organization=organization)),
            "alerts/": first(AlertRule.objects.filter(organization=organization)),
            "channels/": first(NotificationChannel.objects.filter(organization=organization)),
            "reports/": first(Report.objects.filter(organization=organization)),
            "invoices/": first(organization.invoices.all()),
            "roles/": first(organization.roles.all()),
            "members/": first(Membership.objects.filter(organization=organization)),
            "packages/": getattr(Package.objects.first(), "slug", ""),
            "organizations/": organization.slug,
            "workspace/": organization.slug,
            "posts/": "", "pages/": "",
        }

    def concrete(self, pattern, table):
        url = "/" + pattern
        for match in list(re.finditer(r"<[^>]+>", url)):
            token, prefix, value = match.group(0), url[: match.start()], None
            if ":action>" in token:
                value = "acknowledge"
            elif ":token>" in token:
                value = "x" * 20
            else:
                for key, candidate in table.items():
                    if prefix.endswith(key) and candidate:
                        value = candidate
                        break
            if value is None:
                return None
            url = url[: match.start()] + value + url[match.end():]
        return url

    def urls_for(self, organization):
        table = self.identifiers(organization)
        for pattern in ROUTES:
            url = self.concrete(pattern, table)
            if url and not any(skip in url for skip in SKIP):
                yield url

    def assert_no_server_errors(self, organization, user, label):
        self.client.force_login(user)
        failures = []
        for url in self.urls_for(organization):
            try:
                response = self.client.get(url)
            except Exception as exc:                      # noqa: BLE001 - reported below
                failures.append(f"{url} raised {type(exc).__name__}: {exc}")
                continue
            if response.status_code >= 500:
                failures.append(f"{url} returned {response.status_code}")
        self.assertFalse(failures, f"{label}:\n  " + "\n  ".join(failures))


class ColdStateSweepTests(SweepMixin, TestCase):
    """States a freshly-seeded test never produces, but a real deployment does."""

    def test_a_workspace_with_nothing_in_it(self):
        organization, owner = workspace("Empty Co")
        self.assert_no_server_errors(organization, owner, "empty workspace")

    def test_objects_that_have_never_been_used(self):
        """Every "last happened at" column still null, hours after signing in."""
        organization, owner = workspace("Cold Co")
        site = Site.objects.create(organization=organization, name="Cold Site")
        camera = Camera.objects.create(
            organization=organization, site=site, name="Cold Cam",
            protocol=Camera.Protocol.DEMO, status=Camera.Status.PENDING,
            enabled_analytics=["gesture", "geofence"], last_seen_at=None, last_frame_at=None)
        Zone.objects.create(camera=camera, name="Cold Zone",
                            polygon=[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]])
        Employee.objects.create(organization=organization, full_name="Never Seen",
                                employee_code="E1", enrolled_at=None)
        AlertRule.objects.create(organization=organization, name="Never Fired",
                                 last_fired_at=None)
        dataset = Dataset.objects.create(organization=organization, name="Cold Set",
                                         task="classification", classes=["a", "b"])
        TrainingJob.objects.create(organization=organization, dataset=dataset,
                                   name="Queued", status=TrainingJob.Status.QUEUED,
                                   history=[])
        ModelVersion.objects.create(organization=organization, name="Cold Model",
                                    slot="campynet", version="0.0.1")
        Report.objects.create(organization=organization, name="Cold Report", last_run_at=None)
        NotificationChannel.objects.create(organization=organization, name="Cold Channel",
                                           kind=NotificationChannel.Kind.EMAIL)
        # Past the last-seen throttle, the state the SimpleLazyObject crash needed.
        User.objects.filter(pk=owner.pk).update(
            last_seen_at=timezone.now() - timedelta(hours=3))

        self.assert_no_server_errors(organization, owner, "cold objects")

    def test_every_subscription_state(self):
        now = timezone.now()
        states = {
            "trial expired": {"status": Subscription.Status.TRIALING,
                              "trial_ends_at": now - timedelta(days=3),
                              "current_period_end": now - timedelta(days=3)},
            "past due inside grace": {"status": Subscription.Status.PAST_DUE,
                                      "current_period_end": now - timedelta(days=2)},
            "past due grace expired": {"status": Subscription.Status.PAST_DUE,
                                       "current_period_end": now - timedelta(days=30)},
            "cancelled": {"status": Subscription.Status.CANCELLED,
                          "current_period_end": now - timedelta(days=1)},
        }
        for label, changes in states.items():
            with self.subTest(subscription=label):
                organization, owner = workspace(f"Sub {label}")
                Site.objects.create(organization=organization, name="S")
                Subscription.objects.filter(
                    pk=organization.subscriptions.first().pk).update(**changes)
                organization.refresh_subscription()
                self.assert_no_server_errors(organization, owner, label)

    def test_a_workspace_with_no_subscription_at_all(self):
        organization, owner = workspace("Unsubscribed Co")
        organization.subscriptions.all().delete()
        organization.refresh_subscription()
        self.assert_no_server_errors(organization, owner, "no subscription")

    def test_a_suspended_workspace(self):
        organization, owner = workspace("Suspended Co")
        Organization.objects.filter(pk=organization.pk).update(status="suspended")
        organization.refresh_from_db()
        self.assert_no_server_errors(organization, owner, "suspended")


class WriteSweepTests(SweepMixin, TestCase):
    """Submitting is a separate surface from rendering, and was untested.

    A rejected form re-rendering, or a redirect, is a pass here — only a server
    error is a failure. The point is that no write path dies on submit.
    """

    def payload_for(self, form):
        data = {}
        for name, field in form.fields.items():
            initial = form.get_initial_for_field(field, name)
            if isinstance(field, djforms.ModelMultipleChoiceField):
                data[name] = [obj.pk for obj in field.queryset[:1]]
            elif isinstance(field, djforms.ModelChoiceField):
                obj = field.queryset.first()
                data[name] = obj.pk if obj else ""
            elif isinstance(field, djforms.MultipleChoiceField):
                data[name] = [c[0] for c in field.choices if c[0]][:1]
            elif isinstance(field, djforms.ChoiceField):
                choices = [c[0] for c in field.choices if c[0]]
                data[name] = initial if initial in choices else (choices[0] if choices else "")
            elif isinstance(field, djforms.BooleanField):
                data[name] = bool(initial)
            elif isinstance(field, (djforms.IntegerField, djforms.FloatField,
                                    djforms.DecimalField)):
                data[name] = initial if initial not in (None, "") else 1
            elif isinstance(field, djforms.DateTimeField):
                data[name] = (initial or timezone.now()).strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(field, djforms.DateField):
                data[name] = (initial or timezone.now().date()).strftime("%Y-%m-%d")
            elif isinstance(field, djforms.EmailField):
                data[name] = initial or "someone@example.test"
            elif isinstance(field, djforms.URLField):
                data[name] = initial or "https://example.test/hook"
            elif isinstance(field, djforms.FileField):
                continue
            else:
                import json
                data[name] = (json.dumps(initial) if isinstance(initial, (list, dict))
                              else (initial if initial not in (None, "") else "Swept value"))
        return data

    def test_no_write_endpoint_dies_on_submit(self):
        organization, owner = workspace("Write Co")
        owner.platform_role = User.PlatformRole.SUPERADMIN
        owner.is_superuser = owner.is_staff = True
        owner.save()
        site = Site.objects.create(organization=organization, name="Site")
        camera = Camera.objects.create(organization=organization, site=site, name="Cam",
                                       protocol=Camera.Protocol.DEMO,
                                       enabled_analytics=["gesture"])
        Zone.objects.create(camera=camera, name="Z",
                            polygon=[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]])
        Employee.objects.create(organization=organization, full_name="E", employee_code="E1")
        AlertRule.objects.create(organization=organization, name="R")
        NotificationChannel.objects.create(organization=organization, name="C",
                                           kind=NotificationChannel.Kind.EMAIL,
                                           config={"to": "a@b.test"})
        dataset = Dataset.objects.create(organization=organization, name="D",
                                         task="classification", classes=["a", "b"])
        TrainingJob.objects.create(organization=organization, dataset=dataset, name="J",
                                   status=TrainingJob.Status.SUCCEEDED, history=[])
        ModelVersion.objects.create(organization=organization, name="M", slot="campynet",
                                    version="1")
        Report.objects.create(organization=organization, name="Rep")
        Event.objects.create(organization=organization, camera=camera, analytic="gesture",
                             event_type="running", severity="medium", title="E",
                             description="d", confidence=0.9, occurred_at=timezone.now())
        Incident.objects.create(organization=organization, primary_camera=camera, title="I",
                                reference="INC-1", severity="low", started_at=timezone.now())

        self.client.force_login(owner)
        failures, submitted = [], 0
        for url in self.urls_for(organization):
            data = {}
            try:
                page = self.client.get(url)
                form = page.context.get("form") if getattr(page, "context", None) else None
                if form is not None and hasattr(form, "fields"):
                    data = self.payload_for(form)
            except Exception:                             # noqa: BLE001 - the POST is the test
                pass
            data.setdefault("action", "acknowledge")
            try:
                response = self.client.post(url, data)
            except Exception as exc:                      # noqa: BLE001 - reported below
                failures.append(f"POST {url} raised {type(exc).__name__}: {exc}")
                continue
            if response.status_code >= 500:
                failures.append(f"POST {url} returned {response.status_code}")
            elif response.status_code < 400:
                submitted += 1

        self.assertFalse(failures, "write endpoints failed:\n  " + "\n  ".join(failures))
        self.assertGreater(submitted, 40, "the sweep should be reaching real endpoints")


class ApiSchemaTests(TestCase):
    def test_the_openapi_schema_generates_without_warnings(self):
        """A warning here means clients generate duplicate or misnamed types.

        Driven through the management command rather than the generator's
        internals, which move between releases.
        """
        import io

        from django.core.management import call_command

        out, err = io.StringIO(), io.StringIO()
        call_command("spectacular", "--fail-on-warn", stdout=out, stderr=err)

        self.assertIn("openapi", out.getvalue().lower())
        self.assertNotIn("Warning", err.getvalue())
