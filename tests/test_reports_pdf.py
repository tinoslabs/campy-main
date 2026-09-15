"""Analysis reports as PDF, per feature and combined, with evidence.

A finding is a claim; the frame behind it is what makes the claim checkable in
a safety meeting or an insurance file. These tests assert on the produced
document — its text and its embedded images — rather than on the fact that
bytes came back.
"""
from __future__ import annotations

import io
from datetime import timedelta
from decimal import Decimal

import numpy as np
import pymupdf
from django.core.files.base import ContentFile
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from apps.analytics.evidence import annotate_event
from apps.analytics.pdf import build_analysis_report, humanise
from apps.cameras.models import Camera, CameraSnapshot, Site, Zone
from apps.events.models import Event
from tests.test_platform import full_workspace


def a_jpeg(width=480, height=270, colour=90):
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.full((height, width, 3), colour, dtype=np.uint8)).save(
        buffer, format="JPEG")
    return buffer.getvalue()


def text_of(pdf: bytes) -> str:
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        return "\n".join(page.get_text("text") for page in document)


def image_count(pdf: bytes) -> int:
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        return sum(len(page.get_images()) for page in document)


class ReportTestCase(TestCase):
    def setUp(self):
        self.organization, self.owner = full_workspace("Report Co")
        self.site = Site.objects.create(organization=self.organization, name="Bengaluru Plant")
        self.camera = Camera.objects.create(
            organization=self.organization, site=self.site, name="Loading Bay",
            protocol=Camera.Protocol.DEMO)
        self.zone = Zone.objects.create(camera=self.camera, name="Machine Guard Zone",
                                        polygon=[[0.4, 0.3], [0.9, 0.3], [0.9, 0.9], [0.4, 0.9]])

    def an_event(self, *, analytic="geofence", severity="critical", with_snapshot=True,
                 event_type="unauthorised_zone_entry", title="Unauthorised entry", **kwargs):
        snapshot = None
        if with_snapshot:
            snapshot = CameraSnapshot(camera=self.camera, captured_at=timezone.now(),
                                      width=480, height=270, reason=event_type)
            snapshot.image.save("evidence.jpg", ContentFile(a_jpeg()), save=False)
            snapshot.save()
        return Event.objects.create(
            organization=self.organization, camera=self.camera, zone=self.zone,
            analytic=analytic, event_type=event_type, severity=severity, title=title,
            description="Somebody crossed into the guarded area.", confidence=0.91,
            occurred_at=timezone.now() - timedelta(hours=2),
            bounding_box=[120.0, 60.0, 190.0, 220.0], snapshot=snapshot, **kwargs)


class EvidenceTests(ReportTestCase):
    def test_the_detection_is_drawn_onto_the_stored_frame(self):
        event = self.an_event()

        annotated = annotate_event(event)

        self.assertIsNotNone(annotated)
        self.assertTrue(annotated.startswith(b"\xff\xd8"))
        from PIL import Image

        original = Image.open(io.BytesIO(a_jpeg())).convert("RGB")
        marked = Image.open(io.BytesIO(annotated)).convert("RGB")
        self.assertNotEqual(list(original.getdata()), list(marked.getdata()),
                            "nothing was drawn on the frame")

    def test_an_event_with_no_stored_frame_yields_nothing_rather_than_failing(self):
        self.assertIsNone(annotate_event(self.an_event(with_snapshot=False)))

    def test_a_malformed_box_does_not_stop_the_frame_being_used(self):
        event = self.an_event()
        event.bounding_box = ["a", "b", "c", "d"]
        self.assertIsNotNone(annotate_event(event), "the frame is still evidence")

    def test_an_empty_box_still_produces_the_frame(self):
        event = self.an_event()
        event.bounding_box = []
        self.assertIsNotNone(annotate_event(event))

    def test_labels_read_as_english(self):
        self.assertEqual(humanise("fire_detected"), "Fire detected")
        self.assertEqual(humanise(""), "Detection")


class SingleFeatureReportTests(ReportTestCase):
    def test_it_names_the_feature_and_explains_what_it_watches_for(self):
        self.an_event(analytic="fire", event_type="fire_detected", title="Possible fire")

        pdf = build_analysis_report(self.organization, analytics=["fire"], days=7)
        text = text_of(pdf)

        self.assertIn("Fire & Smoke Detection", text)
        self.assertIn("not a certified replacement", text, "the limits must travel with it")
        self.assertNotIn("Combined analysis report", text)

    def test_the_evidence_frame_is_embedded_with_its_context(self):
        self.an_event()

        pdf = build_analysis_report(self.organization, analytics=["geofence"], days=7)
        text = text_of(pdf)

        self.assertGreaterEqual(image_count(pdf), 1, "no evidence image was embedded")
        self.assertIn("Evidence", text)
        self.assertIn("Loading Bay", text)
        self.assertIn("Machine Guard Zone", text)
        self.assertIn("confidence 91%", text)

    def test_only_the_requested_feature_appears(self):
        self.an_event(analytic="geofence", title="Zone entry")
        self.an_event(analytic="theft", event_type="shelf_dwell", title="Lingering at stock")

        text = text_of(build_analysis_report(self.organization, analytics=["geofence"], days=7))

        self.assertIn("Zone entry", text)
        self.assertNotIn("Lingering at stock", text)


class CombinedReportTests(ReportTestCase):
    def test_it_compares_the_features_against_each_other(self):
        self.an_event(analytic="geofence", title="Zone entry")
        self.an_event(analytic="fire", event_type="fire_detected", title="Possible fire")
        self.an_event(analytic="theft", event_type="shelf_dwell", severity="medium",
                      title="Lingering at stock")

        text = text_of(build_analysis_report(self.organization, days=7))

        self.assertIn("Combined analysis report", text)
        self.assertIn("By feature", text)
        for label in ("Geofencing", "Fire & Smoke Detection", "Theft Detection"):
            self.assertIn(label, text)

    def test_the_headline_figures_are_the_real_counts(self):
        for _ in range(3):
            self.an_event(severity="critical")
        self.an_event(severity="low", status=Event.Status.FALSE_POSITIVE)

        text = text_of(build_analysis_report(self.organization, days=7))

        self.assertIn("What was found", text)
        self.assertIn("4", text)

    def test_the_caveats_a_reader_needs_are_present(self):
        self.an_event(analytic="theft", event_type="shelf_dwell", title="Lingering")

        text = text_of(build_analysis_report(self.organization, days=7))

        self.assertIn("prompts for a person to review", text)
        self.assertIn("never intent", text.replace("\n", " "))
        self.assertIn("data-protection", text)

    def test_a_period_with_nothing_in_it_still_produces_a_report(self):
        """An empty result is a finding: the cameras were watched."""
        text = text_of(build_analysis_report(self.organization, days=1))

        self.assertIn("Nothing was detected", text)
        self.assertIn("not the same as nobody having looked", text.replace("\n", " "))

    def test_the_period_and_scope_are_stated(self):
        pdf = build_analysis_report(self.organization, analytics=["fire"], days=30,
                                    generated_by="Priya Raman")
        text = text_of(pdf)

        self.assertIn("30 days", text)
        self.assertIn("Priya Raman", text)
        self.assertIn(self.organization.name, text)


class DownloadTests(ReportTestCase):
    def setUp(self):
        super().setUp()
        self.an_event()
        self.client.force_login(self.owner)
        self.url = reverse("dashboard:analysis_report")

    def test_the_combined_report_downloads_as_a_pdf(self):
        response = self.client.get(self.url, {"days": 7})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/pdf")
        self.assertIn("campy-combined-report", response["Content-Disposition"])
        self.assertTrue(response.content.startswith(b"%PDF"))

    def test_a_single_feature_downloads_under_its_own_name(self):
        response = self.client.get(self.url, {"analytics": "geofence", "days": 7})

        self.assertEqual(response.status_code, 200)
        self.assertIn("campy-geofence-report", response["Content-Disposition"])

    def test_a_camera_can_be_named(self):
        response = self.client.get(self.url, {"camera": str(self.camera.uid), "days": 7})
        self.assertEqual(response.status_code, 200)
        self.assertIn(self.camera.name, text_of(response.content))

    def test_junk_parameters_do_not_break_the_download(self):
        for params in ({"days": "banana"}, {"analytics": "nonsense,geofence"},
                       {"camera": "not-a-uuid"}, {"evidence": "-5"}, {"days": "99999"}):
            with self.subTest(params=params):
                response = self.client.get(self.url, params)
                self.assertEqual(response.status_code, 200, params)

    def test_a_feature_outside_the_package_is_not_reported_on(self):
        from apps.billing.models import Package
        from apps.billing.services import activate_subscription

        starter = Package.objects.create(
            name="Starter Only", price_inr=Decimal("100"),
            features={"geofencing": True, "fire_detection": False, "reports": True},
            quotas={"cameras": 5})
        activate_subscription(self.organization, starter, currency="INR")
        self.organization.refresh_subscription()

        text = text_of(self.client.get(self.url, {"days": 7}).content)

        self.assertIn("Geofencing", text)
        self.assertNotIn("Fire & Smoke Detection", text)

    def test_a_viewer_without_the_export_permission_is_refused(self):
        from apps.accounts.models import Membership, User

        viewer = User.objects.create_user(email="viewer@report.test", password="TestPass!2024",
                                          full_name="Viewer")
        Membership.objects.create(user=viewer, organization=self.organization,
                                  role=self.organization.roles.get(code="viewer"))
        viewer.active_organization = self.organization
        viewer.save(update_fields=["active_organization"])

        self.client.force_login(viewer)
        self.assertIn(self.client.get(self.url).status_code, (302, 403))

    def test_one_workspace_cannot_report_on_anothers_events(self):
        other, _ = full_workspace("Somebody Else")
        their_site = Site.objects.create(organization=other, name="Theirs")
        their_camera = Camera.objects.create(organization=other, site=their_site,
                                             name="Their Secret Camera",
                                             protocol=Camera.Protocol.DEMO)
        Event.objects.create(organization=other, camera=their_camera, analytic="geofence",
                             event_type="unauthorised_zone_entry", severity="critical",
                             title="Their private finding", description="x", confidence=0.9,
                             occurred_at=timezone.now())

        text = text_of(self.client.get(self.url, {"days": 7}).content)

        self.assertNotIn("Their private finding", text)
        self.assertNotIn("Their Secret Camera", text)


class SavedReportTests(ReportTestCase):
    def test_a_report_defined_as_pdf_exports_as_one(self):
        """The model has offered a PDF format all along; it only ever wrote CSV."""
        from apps.analytics.models import Report

        self.an_event()
        report = Report.objects.create(
            organization=self.organization, name="Weekly safety",
            output_format=Report.Format.PDF, analytics=["geofence"], date_range_days=7)
        self.client.force_login(self.owner)

        response = self.client.get(reverse("dashboard:report_export", args=[report.uid]))

        self.assertEqual(response["Content-Type"], "application/pdf")
        self.assertIn("weekly-safety", response["Content-Disposition"])
        self.assertIn("Weekly safety", text_of(response.content))

    def test_a_csv_report_is_still_a_csv(self):
        from apps.analytics.models import Report

        report = Report.objects.create(
            organization=self.organization, name="Weekly csv",
            output_format=Report.Format.CSV, date_range_days=7)
        self.client.force_login(self.owner)

        response = self.client.get(reverse("dashboard:report_export", args=[report.uid]))

        self.assertEqual(response["Content-Type"], "text/csv")
