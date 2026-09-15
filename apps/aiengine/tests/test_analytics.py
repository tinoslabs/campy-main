"""Acceptance tests for the seven Campy AI analytics.

Each test drives a deterministic synthetic scene through the real pipeline and
asserts that the expected event fires — and, just as importantly, that quiet
scenes stay quiet. A detector that never misses but alerts constantly is
useless in a control room.
"""
from __future__ import annotations

from django.test import TestCase

from apps.aiengine.detectors.base import Zone
from apps.aiengine.pipeline import CameraPipeline
from apps.aiengine.simulation import SceneSimulator


def drive(frames, analytics=None, zones=None, fps=6.0, config=None):
    """Run frames through a pipeline; return ``(events, last_result)``.

    Uses :class:`~django.test.TestCase` rather than ``SimpleTestCase`` because
    the model registry legitimately queries the database to resolve which
    checkpoint (if any) is deployed to each slot.
    """
    pipeline = CameraPipeline(
        camera_id=1, analytics=analytics, zones=zones or [], fps=fps, config=config or {}
    )
    events, last = [], None
    for frame in frames:
        result = pipeline.process(frame)
        events.extend(result.events)
        last = result
    return events, last


def event_types(events) -> set[str]:
    return {event.event_type for event in events}


def scene(seed: int = 3) -> SceneSimulator:
    return SceneSimulator(320, 180, seed=seed)


RESTRICTED = Zone(
    id=1, name="Server Room",
    polygon=[[0.62, 0.35], [1.0, 0.35], [1.0, 1.0], [0.62, 1.0]],
    kind="restricted", severity="high",
)
SHELF = Zone(
    id=2, name="Stock Shelf",
    polygon=[[0.70, 0.25], [1.0, 0.25], [1.0, 0.85], [0.70, 0.85]],
    kind="shelf",
)
CORRIDOR = Zone(
    id=3, name="Corridor",
    polygon=[[0.05, 0.4], [0.95, 0.4], [0.95, 1.0], [0.05, 1.0]],
    kind="counting", max_occupancy=5,
)


class GestureTrackingTests(TestCase):
    def test_walking_person_is_tracked_and_classified(self):
        events, last = drive(scene().walk_across(60), ["gesture"])
        postures = [
            d.label for r in [last.results["gesture"]] for d in r.detections
        ]
        _ = events
        # Somewhere in the clip the walker must have been seen walking.
        pipeline = CameraPipeline(camera_id=1, analytics=["gesture"], fps=6.0)
        seen = set()
        for frame in scene().walk_across(60):
            result = pipeline.process(frame)
            seen.update(d.label for d in result.results["gesture"].detections)
        self.assertIn("walking", seen, f"postures seen: {seen} / final {postures}")

    def test_prolonged_inactivity_and_loitering_are_flagged(self):
        events, _ = drive(
            scene().idle_person(400),
            ["gesture"],
            config={"gesture": {"idle_seconds": 25, "loiter_seconds": 30}},
        )
        types = event_types(events)
        self.assertIn("prolonged_inactivity", types, types)
        self.assertIn("loitering", types, types)

    def test_running_is_detected(self):
        events, _ = drive(scene().running_person(90), ["gesture"])
        self.assertIn("running_detected", event_types(events))

    def test_walking_alone_does_not_raise_running(self):
        """A normal walk must not be reported as running."""
        events, _ = drive(scene().walk_across(70), ["gesture"])
        self.assertNotIn("running_detected", event_types(events))


class GeofenceTests(TestCase):
    def test_entering_a_restricted_zone_raises_an_event(self):
        events, last = drive(scene().restricted_entry(70), ["geofence"], [RESTRICTED])
        self.assertIn("unauthorised_zone_entry", event_types(events))
        self.assertEqual(last.results["geofence"].metrics["zones"], 1)

    def test_severity_matches_the_zone(self):
        events, _ = drive(scene().restricted_entry(70), ["geofence"], [RESTRICTED])
        entry = next(e for e in events if e.event_type == "unauthorised_zone_entry")
        self.assertEqual(entry.severity, "high")
        self.assertEqual(entry.zone_id, RESTRICTED.id)

    def test_no_zones_means_no_events(self):
        events, _ = drive(scene().restricted_entry(70), ["geofence"], [])
        self.assertEqual(events, [])

    def test_a_disarmed_zone_stays_quiet(self):
        """A zone scheduled outside the current window must not fire."""
        night_only = Zone(
            id=9, name="After Hours", polygon=RESTRICTED.polygon, kind="restricted",
            schedule={"from": "23:30", "to": "23:31"},
        )
        from datetime import datetime

        # is_active_now is time-based; verify the predicate itself rather than
        # depending on when the suite happens to run.
        moment = datetime(2025, 1, 1, 12, 0)
        self.assertFalse(night_only.is_active_now(moment))
        self.assertTrue(RESTRICTED.is_active_now(moment))


class CrowdTests(TestCase):
    def test_crowd_formation_is_detected(self):
        events, last = drive(
            scene().crowd(90, people=16), ["crowd"], [CORRIDOR],
            config={"crowd": {"crowd_threshold": 4, "sustain_frames": 4}},
        )
        self.assertIn("crowd_formation", event_types(events))
        self.assertGreaterEqual(last.results["crowd"].metrics["peak"], 4)

    def test_single_person_is_not_a_crowd(self):
        events, last = drive(scene().walk_across(70), ["crowd"])
        self.assertNotIn("crowd_formation", event_types(events))
        self.assertFalse(last.results["crowd"].metrics["is_crowded"])

    def test_heatmap_is_produced(self):
        _, last = drive(scene().crowd(40, people=8), ["crowd"])
        heatmap = last.results["crowd"].metrics["heatmap"]
        self.assertEqual(len(heatmap), 4)
        self.assertEqual(len(heatmap[0]), 4)


class FireDetectionTests(TestCase):
    def test_fire_outbreak_raises_a_critical_event(self):
        events, _ = drive(scene().fire_outbreak(70, warmup=16), ["fire"])
        self.assertIn("fire_detected", event_types(events))
        fire = next(e for e in events if e.event_type == "fire_detected")
        self.assertEqual(fire.severity, "critical")
        self.assertGreater(fire.confidence, 0.5)

    def test_ordinary_scene_produces_no_fire_alert(self):
        events, _ = drive(scene().walk_across(70), ["fire"])
        self.assertEqual(event_types(events), set())

    def test_empty_scene_produces_no_fire_alert(self):
        simulator = scene()
        events, _ = drive([simulator.frame([]) for _ in range(40)], ["fire"])
        self.assertEqual(event_types(events), set())


class ObjectDetectionTests(TestCase):
    def test_unattended_object_is_detected(self):
        events, _ = drive(scene().abandoned_object(180), ["object"], fps=3.0)
        self.assertIn("unattended_object", event_types(events))

    def test_person_walking_through_leaves_nothing_behind(self):
        events, _ = drive(scene().walk_across(80), ["object"], fps=3.0)
        self.assertNotIn("unattended_object", event_types(events))


class TheftDetectionTests(TestCase):
    def test_shelf_interaction_raises_a_review_prompt(self):
        events, _ = drive(scene().restricted_entry(100), ["theft"], [SHELF])
        self.assertIn("suspicious_handling", event_types(events))

    def test_findings_are_phrased_as_review_not_accusation(self):
        """The spec is explicit: highlight risk, never accuse an individual."""
        events, _ = drive(scene().restricted_entry(100), ["theft"], [SHELF])
        finding = next(e for e in events if e.event_type == "suspicious_handling")
        self.assertTrue(finding.metadata.get("review_only"))
        self.assertIn("review", finding.description.lower())

    def test_no_shelf_zones_means_no_theft_events(self):
        events, _ = drive(scene().restricted_entry(100), ["theft"], [])
        self.assertEqual(event_types(events), set())


class FaceRecognitionTests(TestCase):
    def test_faces_are_located_on_people(self):
        _, last = drive(scene().walk_across(60), ["face"])
        self.assertGreaterEqual(last.results["face"].metrics["faces"], 1)

    def test_unknown_alerts_require_an_enrolled_gallery(self):
        """With nobody enrolled, everyone is 'unknown' — alerting would be noise."""
        events, last = drive(scene().walk_across(60), ["face"])
        self.assertEqual(last.results["face"].metrics["gallery_size"], 0)
        self.assertNotIn("unknown_person", event_types(events))

    def test_gallery_matches_by_cosine_similarity(self):
        import numpy as np

        from apps.aiengine.detectors.face import FaceGallery

        gallery = FaceGallery(
            [
                {"id": 1, "name": "Asha", "role": "engineer", "embedding": [1.0, 0.0, 0.0]},
                {"id": 2, "name": "Ravi", "role": "guard", "embedding": [0.0, 1.0, 0.0]},
            ]
        )
        self.assertEqual(len(gallery), 2)
        match = gallery.match(np.array([0.98, 0.04, 0.0]), threshold=0.62)
        self.assertIsNotNone(match)
        self.assertEqual(match[1], "Asha")
        # An orthogonal vector must not match anybody.
        self.assertIsNone(gallery.match(np.array([0.0, 0.0, 1.0]), threshold=0.62))


class PipelineTests(TestCase):
    def test_all_seven_analytics_run_together(self):
        _, last = drive(scene().walk_across(60), None, [RESTRICTED, SHELF, CORRIDOR])
        self.assertEqual(len(last.results), 7)
        for key, result in last.results.items():
            self.assertNotIn("error", result.metrics, f"{key} raised: {result.metrics.get('error')}")

    def test_a_failing_analytic_does_not_break_the_pipeline(self):
        """One bad analytic must never take a camera offline."""
        pipeline = CameraPipeline(camera_id=1, analytics=["gesture", "crowd"], fps=6.0)

        def explode(context):
            raise RuntimeError("boom")

        pipeline.analytics["gesture"].analyse = explode
        result = pipeline.process(scene().frame([]))
        self.assertIn("error", result.results["gesture"].metrics)
        self.assertNotIn("error", result.results["crowd"].metrics)

    def test_repeated_findings_are_deduplicated_into_one_event(self):
        events, _ = drive(scene().restricted_entry(70), ["geofence"], [RESTRICTED])
        entries = [e for e in events if e.event_type == "unauthorised_zone_entry"]
        self.assertEqual(len(entries), 1, "the same entry should not alert every frame")

    def test_ghosts_do_not_become_phantom_people(self):
        """A subject present in frame 0 is baked into the initial background.

        When they move off, the vacated patch differs from the model forever.
        Without ghost rejection this registers as a motionless second person.
        """
        _, last = drive(scene().walk_across(70), ["crowd"])
        self.assertLessEqual(
            last.results["crowd"].metrics["people_count"], 1,
            "a vacated background patch was counted as a person",
        )

    def test_pipeline_reports_health(self):
        pipeline = CameraPipeline(camera_id=7, analytics=["gesture"], fps=6.0)
        for frame in scene().walk_across(20):
            pipeline.process(frame)
        health = pipeline.health()
        self.assertEqual(health["camera_id"], 7)
        self.assertEqual(health["frames"], 20)
        self.assertTrue(health["background_ready"])
