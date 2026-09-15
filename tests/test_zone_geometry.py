"""Zone outlines: rejected at the door, and survivable if already stored.

A zone polygon lives in a JSON column and is read by every page that lists the
zone and by the worker analysing that camera. A coordinate that is not a number
used to raise out of `polygon_area`, so one bad zone took its camera's page down
permanently — there was no way to reach the editor to fix it.
"""
from __future__ import annotations

from django.test import TestCase
from django.urls import reverse

from apps.aiengine.vision.geometry import (
    as_polygon, box_polygon_overlap, point_in_polygon, polygon_area,
    polygon_bounds, polygon_centroid,
)
from apps.api.serializers import ZoneSerializer
from apps.cameras.models import Camera, Site, Zone
from apps.dashboard.forms import ZoneForm
from tests.test_platform import full_workspace

SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
UNUSABLE = [
    ("coordinates that are strings", [["a", "b"], ["c", "d"], ["e", "f"]]),
    ("nothing at all", []),
    ("a ragged point", [[1, 2], [3, 4, 5]]),
    ("an odd number of values", [1, 2, 3]),
    ("None", None),
    ("a NaN corner", [[float("nan"), 0.0], [1.0, 0.0], [1.0, 1.0]]),
    ("a string", "nonsense"),
    ("a mapping", {"x": 1}),
]


class GeometryTests(TestCase):
    def test_a_well_formed_polygon_is_unchanged(self):
        self.assertEqual(polygon_area(SQUARE), 1.0)
        self.assertTrue(point_in_polygon((0.5, 0.5), SQUARE))
        self.assertFalse(point_in_polygon((2.0, 2.0), SQUARE))
        self.assertEqual(polygon_centroid(SQUARE), (0.5, 0.5))
        self.assertEqual(polygon_bounds(SQUARE), (0.0, 0.0, 1.0, 1.0))
        self.assertEqual(as_polygon(SQUARE).tolist(), SQUARE)

    def test_unusable_shapes_read_as_covering_nothing(self):
        for label, polygon in UNUSABLE:
            with self.subTest(shape=label):
                self.assertEqual(polygon_area(polygon), 0.0)
                self.assertFalse(point_in_polygon((0.5, 0.5), polygon))
                self.assertEqual(box_polygon_overlap((0, 0, 1, 1), polygon), 0.0)
                self.assertEqual(as_polygon(polygon).shape, (0, 2))

    def test_too_few_points_is_valid_data_but_not_yet_a_polygon(self):
        """Numbers we can read, but nothing you can be inside of."""
        sliver = [[0.1, 0.1]]
        self.assertEqual(as_polygon(sliver).shape, (1, 2))
        self.assertEqual(polygon_area(sliver), 0.0)
        self.assertFalse(point_in_polygon((0.1, 0.1), sliver))


class ZoneInputTests(TestCase):
    def setUp(self):
        self.organization, self.owner = full_workspace("Zone Co")
        site = Site.objects.create(organization=self.organization, name="Site")
        self.camera = Camera.objects.create(organization=self.organization, site=site,
                                            name="Cam", protocol=Camera.Protocol.DEMO)

    def form(self, polygon):
        return ZoneForm(
            data={"name": "Z", "kind": Zone.Kind.RESTRICTED, "severity": Zone.Severity.HIGH,
                  "polygon": polygon, "colour": "#ef4444", "max_occupancy": 0,
                  "min_dwell_seconds": 0, "is_active": True},
            camera=self.camera,
        )

    def test_the_form_refuses_coordinates_that_are_not_numbers(self):
        form = self.form([["a", "b"], ["c", "d"], ["e", "f"]])
        self.assertFalse(form.is_valid())
        self.assertIn("numbers", " ".join(form.errors["polygon"]))

    def test_the_form_refuses_a_shape_with_too_few_points(self):
        self.assertFalse(self.form([[0.1, 0.1], [0.9, 0.9]]).is_valid())

    def test_the_form_accepts_a_real_outline(self):
        form = self.form(SQUARE)
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.cleaned_data["polygon"], SQUARE)

    def test_points_just_outside_the_frame_are_clamped_not_refused(self):
        """You cannot draw outside the image; that is rounding, not an error."""
        form = self.form([[-0.02, 0.0], [1.04, 0.0], [0.5, 1.2]])
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.cleaned_data["polygon"], [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])

    def test_the_shape_arrives_as_a_json_string_from_the_editor(self):
        """The real widget is a hidden input, so the browser posts a string."""
        import json

        form = self.form(json.dumps(SQUARE))
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.cleaned_data["polygon"], SQUARE)

        bad = self.form(json.dumps([["a", "b"]] * 3))
        self.assertFalse(bad.is_valid())

        unreadable = self.form("{not json")
        self.assertFalse(unreadable.is_valid())
        # Not Django's stock "Enter a valid JSON." — nobody typed this field.
        self.assertIn("Draw it again", " ".join(unreadable.errors["polygon"]))

    def test_a_drawn_zone_saves_through_the_real_view(self):
        """End to end: the editor's POST creates a zone with usable coordinates."""
        import json

        self.client.force_login(self.owner)
        response = self.client.post(
            reverse("dashboard:zone_create", args=[self.camera.uid]),
            {"name": "Drawn", "kind": Zone.Kind.RESTRICTED, "severity": Zone.Severity.HIGH,
             "polygon": json.dumps(SQUARE), "colour": "#ef4444", "max_occupancy": 0,
             "min_dwell_seconds": 0, "is_active": "on"},
        )
        self.assertIn(response.status_code, (200, 302), getattr(response, "context", None))
        zone = Zone.objects.filter(camera=self.camera, name="Drawn").first()
        self.assertIsNotNone(zone, "the drawn zone was not saved")
        self.assertEqual(zone.polygon, SQUARE)
        self.assertEqual(zone.area_fraction, 1.0)

    def test_the_api_applies_the_same_rules(self):
        serializer = ZoneSerializer(data={"name": "Z", "kind": "restricted",
                                          "severity": "high", "polygon": [["a", "b"]] * 3})
        self.assertFalse(serializer.is_valid())
        self.assertIn("polygon", serializer.errors)

        ok = ZoneSerializer(data={"name": "Z", "kind": "restricted",
                                  "severity": "high", "polygon": SQUARE})
        self.assertTrue(ok.is_valid(), ok.errors)

    def test_a_bad_polygon_already_in_the_database_still_renders(self):
        """Validation stops new ones; this is the escape hatch for old ones."""
        Zone.objects.create(camera=self.camera, name="Broken",
                            polygon=[["a", "b"], ["c", "d"], ["e", "f"]])
        self.client.force_login(self.owner)

        for url in (reverse("dashboard:camera_detail", args=[self.camera.uid]),
                    reverse("dashboard:zones")):
            with self.subTest(url=url):
                self.assertEqual(self.client.get(url).status_code, 200)

    def test_a_bad_polygon_does_not_stop_the_camera_being_analysed(self):
        """The worker reads the same column; it must not die on one bad zone."""
        from apps.aiengine.pipeline import build_pipeline_from_camera

        Zone.objects.create(camera=self.camera, name="Broken",
                            polygon=[["a", "b"], ["c", "d"], ["e", "f"]])
        self.camera.enabled_analytics = ["geofence"]
        self.camera.save(update_fields=["enabled_analytics"])

        import numpy as np
        pipeline = build_pipeline_from_camera(self.camera)
        result = pipeline.process(np.zeros((180, 320, 3), dtype=np.uint8))
        self.assertIsNotNone(result)
