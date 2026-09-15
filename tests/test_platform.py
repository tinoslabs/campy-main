"""End-to-end platform behaviour: worker, training, deployment and the API."""
from __future__ import annotations

from decimal import Decimal

from django.core import mail
from django.test import TestCase, override_settings

from apps.accounts.models import APIKey, Membership, Organization, User
from apps.aiengine.registry import ModelProvider, clear_cache
from apps.billing.models import Package
from apps.billing.services import activate_subscription
from apps.cameras.models import Camera, Site
from apps.events.models import AlertRule, Event, Incident, NotificationChannel
from apps.training.models import Dataset, DatasetSample, ModelVersion, TrainingJob

ALL_FEATURES = {
    "gesture_tracking": True, "face_recognition": True, "geofencing": True,
    "crowd_management": True, "theft_detection": True, "object_detection": True,
    "fire_detection": True, "custom_training": True, "reports": True,
    "api_access": True, "audit_log": True, "evidence_clips": True,
}


def full_workspace(name="Platform Co"):
    package = Package.objects.create(
        name=f"{name} Plan", price_inr=Decimal("1000"), features=ALL_FEATURES,
        quotas={"cameras": 50, "sites": 10, "users": 20, "employees": 500, "retention_days": 90},
    )
    organization = Organization.objects.create(name=name, country="IN", status="active")
    activate_subscription(organization, package, currency="INR")
    owner = User.objects.create_user(email=f"owner@{name.lower().replace(' ', '')}.test",
                                     password="TestPass!2024", full_name="Owner")
    Membership.objects.create(
        user=owner, organization=organization, role=organization.roles.get(code="owner")
    )
    owner.active_organization = organization
    owner.save(update_fields=["active_organization"])
    return organization, owner


def demo_camera(organization, analytics=None, scenario="walk"):
    site = Site.objects.create(organization=organization, name="Test Site")
    return Camera.objects.create(
        organization=organization, site=site, name="Test Camera",
        protocol=Camera.Protocol.DEMO, status=Camera.Status.ONLINE,
        enabled_analytics=analytics or ["gesture", "geofence", "crowd"],
        analytics_config={"demo_scenario": scenario},
        target_fps=6.0,
    )


class CameraWorkerTests(TestCase):
    def test_worker_processes_a_simulated_stream_and_records_events(self):
        organization, _ = full_workspace("Worker Co")
        camera = demo_camera(organization, ["gesture", "geofence"], "restricted")
        from apps.cameras.models import Zone
        Zone.objects.create(
            camera=camera, name="Restricted", kind=Zone.Kind.RESTRICTED,
            severity=Zone.Severity.HIGH,
            polygon=[[0.6, 0.35], [1.0, 0.35], [1.0, 1.0], [0.6, 1.0]],
        )

        from apps.cameras.services import CameraWorker

        worker = CameraWorker(camera, max_frames=70, save_snapshots=False)
        stats = worker.run()

        self.assertGreaterEqual(stats.frames, 60)
        self.assertGreater(stats.fps, 3, f"only {stats.fps:.1f} fps")
        self.assertGreater(
            Event.objects.filter(camera=camera).count(), 0,
            "a person walking into a restricted zone should produce an event",
        )
        camera.refresh_from_db()
        self.assertEqual(camera.status, Camera.Status.ONLINE)

    def test_worker_marks_a_broken_stream_offline(self):
        organization, _ = full_workspace("Broken Co")
        site = Site.objects.create(organization=organization, name="S")
        camera = Camera.objects.create(
            organization=organization, site=site, name="Dead Camera",
            protocol=Camera.Protocol.RTSP, stream_url="rtsp://127.0.0.1:1/nothing",
            enabled_analytics=["gesture"],
        )
        from apps.cameras.services import CameraWorker

        CameraWorker(camera, max_frames=3, save_snapshots=False).run()
        camera.refresh_from_db()
        self.assertIn(camera.status, {Camera.Status.OFFLINE, Camera.Status.DEGRADED})
        self.assertTrue(camera.status_detail)

    def test_analytics_blocked_by_the_plan_do_not_run(self):
        package = Package.objects.create(
            name="No Fire", price_inr=Decimal("100"),
            features={"gesture_tracking": True, "fire_detection": False},
            quotas={"cameras": 5, "sites": 1},
        )
        organization = Organization.objects.create(name="Gated Worker", country="IN", status="active")
        activate_subscription(organization, package, currency="INR")
        camera = demo_camera(organization, ["gesture", "fire"], "fire")

        from apps.aiengine.pipeline import build_pipeline_from_camera

        pipeline = build_pipeline_from_camera(camera)
        self.assertIn("gesture", pipeline.enabled_analytics)
        self.assertNotIn("fire", pipeline.enabled_analytics)


class EventPipelineTests(TestCase):
    def setUp(self):
        self.org, self.owner = full_workspace("Events Co")
        self.camera = demo_camera(self.org)

    def _candidate(self, **overrides):
        from apps.aiengine.detectors.base import EventCandidate

        defaults = {
            "analytic": "fire", "event_type": "fire_detected", "severity": "critical",
            "confidence": 0.9, "title": "Fire detected", "description": "Testing.",
        }
        defaults.update(overrides)
        return EventCandidate(**defaults)

    def test_repeat_findings_collapse_into_one_event(self):
        from apps.events.services import record_event

        first = record_event(self.camera, self._candidate())
        second = record_event(self.camera, self._candidate())
        self.assertIsNotNone(first)
        self.assertIsNone(second, "an identical finding should update, not duplicate")
        first.refresh_from_db()
        self.assertEqual(first.occurrence_count, 2)

    def test_different_findings_are_separate_events(self):
        from apps.events.services import record_event

        self.assertIsNotNone(record_event(self.camera, self._candidate()))
        self.assertIsNotNone(record_event(
            self.camera, self._candidate(event_type="smoke_detected", dedupe_key="fire:smoke")
        ))
        self.assertEqual(Event.objects.filter(camera=self.camera).count(), 2)

    def test_related_events_group_into_one_incident(self):
        from apps.events.services import record_event

        record_event(self.camera, self._candidate())
        record_event(self.camera, self._candidate(event_type="smoke_detected", dedupe_key="fire:smoke"))
        self.assertEqual(Incident.objects.filter(organization=self.org).count(), 1)
        incident = Incident.objects.get(organization=self.org)
        self.assertEqual(incident.event_count, 2)
        self.assertTrue(incident.reference.startswith("INC-"))

    def test_incident_severity_rises_to_its_worst_event(self):
        from apps.events.services import record_event

        record_event(self.camera, self._candidate(severity="low", event_type="a", dedupe_key="a"))
        record_event(self.camera, self._candidate(severity="critical", event_type="b", dedupe_key="b"))
        self.assertEqual(Incident.objects.get(organization=self.org).severity, "critical")


@override_settings(EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend")
class AlertRoutingTests(TestCase):
    def setUp(self):
        self.org, self.owner = full_workspace("Alert Co")
        self.camera = demo_camera(self.org)
        self.channel = NotificationChannel.objects.create(
            organization=self.org, name="Ops email", kind=NotificationChannel.Kind.EMAIL,
            config={"emails": ["ops@alert.test"]},
        )
        self.rule = AlertRule.objects.create(
            organization=self.org, name="Criticals", min_severity="high",
            min_confidence=0.5, cooldown_seconds=300,
        )
        self.rule.channels.set([self.channel])
        mail.outbox = []

    def _event(self, severity="critical", confidence=0.9, analytic="fire"):
        return Event.objects.create(
            organization=self.org, camera=self.camera, analytic=analytic,
            event_type="fire_detected", severity=severity, confidence=confidence,
            title=f"{severity} finding", description="Test.",
        )

    def test_a_matching_event_sends_an_email(self):
        from apps.events.services import dispatch_alerts

        sent = dispatch_alerts(self._event())
        self.assertEqual(len(sent), 1)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("CRITICAL", mail.outbox[0].subject)
        self.assertEqual(mail.outbox[0].to, ["ops@alert.test"])

    def test_an_event_below_the_severity_floor_is_ignored(self):
        from apps.events.services import dispatch_alerts

        self.assertEqual(dispatch_alerts(self._event(severity="low")), [])
        self.assertEqual(len(mail.outbox), 0)

    def test_an_event_below_the_confidence_floor_is_ignored(self):
        from apps.events.services import dispatch_alerts

        self.assertEqual(dispatch_alerts(self._event(confidence=0.2)), [])

    def test_the_cooldown_suppresses_a_repeat(self):
        """Alert fatigue is the failure mode that kills monitoring systems."""
        from apps.events.services import dispatch_alerts

        dispatch_alerts(self._event())
        mail.outbox = []
        self.assertEqual(dispatch_alerts(self._event(analytic="object")), [])
        self.assertEqual(len(mail.outbox), 0)

    def test_an_analytic_filter_is_respected(self):
        from apps.events.services import dispatch_alerts

        self.rule.analytics = ["theft"]
        self.rule.save()
        self.assertEqual(dispatch_alerts(self._event(analytic="fire")), [])

    def test_a_failing_channel_does_not_lose_the_event(self):
        from apps.events.services import dispatch_alerts

        self.channel.config = {"emails": []}          # misconfigured on purpose
        self.channel.save()
        sent = dispatch_alerts(self._event())
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].status, "failed")
        self.assertTrue(sent[0].error)


class TrainingTests(TestCase):
    def setUp(self):
        self.org, self.owner = full_workspace("Training Co")
        self.dataset = Dataset.objects.create(
            organization=self.org, name="Test dataset", task="classification",
            classes=["normal", "fire", "smoke"], image_size=32, validation_split=0.25,
        )
        self._populate()

    def _populate(self, per_class=18):
        import io

        import numpy as np
        from django.core.files.base import ContentFile
        from PIL import Image

        from apps.aiengine.simulation import synthetic_classification_dataset

        images, labels = synthetic_classification_dataset(
            self.dataset.classes, samples_per_class=per_class, size=32, seed=5
        )
        for index, (array, label_index) in enumerate(zip(images, labels)):
            buffer = io.BytesIO()
            Image.fromarray(np.asarray(array, dtype=np.uint8)).save(buffer, format="JPEG")
            sample = DatasetSample(dataset=self.dataset, label=self.dataset.classes[int(label_index)])
            sample.image.save(f"s{index}.jpg", ContentFile(buffer.getvalue()), save=False)
            sample.save()
        self.dataset.refresh_counts()

    def test_dataset_reports_readiness(self):
        self.assertTrue(self.dataset.is_trainable)
        self.assertEqual(self.dataset.labelled_count, 54)
        self.assertEqual(self.dataset.status, Dataset.Status.READY)

    def test_an_empty_dataset_is_not_trainable(self):
        empty = Dataset.objects.create(
            organization=self.org, name="Empty", classes=["a", "b"], image_size=32
        )
        self.assertFalse(empty.is_trainable)
        self.assertFalse(empty.readiness["samples_ok"])

    def test_imbalance_is_detected(self):
        self.dataset.class_distribution = {"normal": 400, "fire": 10, "smoke": 8}
        self.assertTrue(self.dataset.is_imbalanced)
        self.dataset.class_distribution = {"normal": 40, "fire": 30, "smoke": 35}
        self.assertFalse(self.dataset.is_imbalanced)

    def test_training_produces_a_model_that_actually_learned(self):
        from apps.training.services import run_training_job

        job = TrainingJob.objects.create(
            organization=self.org, name="Test run", dataset=self.dataset,
            architecture="campynet", target_slot="campynet_fire",
            epochs=10, batch_size=16, learning_rate=0.003, width=0.75, depth=2,
            created_by=self.owner,
        )
        version = run_training_job(job)
        job.refresh_from_db()

        self.assertEqual(job.status, TrainingJob.Status.SUCCEEDED, job.error_message)
        self.assertGreaterEqual(len(job.history), 5)
        self.assertLess(
            job.history[-1]["loss"], job.history[0]["loss"], "loss did not decrease"
        )
        self.assertIsNotNone(version)
        self.assertGreater(version.accuracy, 0.7, f"accuracy only {version.accuracy}")
        self.assertTrue(version.exists_on_disk)
        self.assertEqual(version.classes, self.dataset.classes)

    def test_a_cancelled_job_stops_and_produces_nothing(self):
        from apps.training.services import run_training_job

        job = TrainingJob.objects.create(
            organization=self.org, name="Cancelled run", dataset=self.dataset,
            architecture="campynet", epochs=30, batch_size=16, width=0.5, depth=2,
            cancel_requested=True, created_by=self.owner,
        )
        version = run_training_job(job)
        job.refresh_from_db()
        self.assertEqual(job.status, TrainingJob.Status.CANCELLED)
        self.assertIsNone(version)

    def test_training_failure_is_recorded_rather_than_raised(self):
        from apps.training.services import run_training_job

        empty = Dataset.objects.create(
            organization=self.org, name="No samples", classes=["a", "b"], image_size=32
        )
        job = TrainingJob.objects.create(
            organization=self.org, name="Doomed run", dataset=empty,
            architecture="campynet", epochs=2, created_by=self.owner,
        )
        version = run_training_job(job)
        job.refresh_from_db()
        self.assertEqual(job.status, TrainingJob.Status.FAILED)
        self.assertTrue(job.error_message)
        self.assertIsNone(version)


class ModelDeploymentTests(TestCase):
    def setUp(self):
        self.org, self.owner = full_workspace("Deploy Co")
        self.other, _ = full_workspace("Other Co")
        clear_cache()

    def _version(self, organization, slot="campynet_fire", version="1.0.0"):
        import tempfile
        from pathlib import Path

        from apps.aiengine.architectures import build_classifier

        model = build_classifier(num_classes=3, width=0.5, depth=2, input_size=32)
        model.meta["classes"] = ["normal", "fire", "smoke"]
        path = Path(tempfile.mkdtemp()) / "model.npz"
        model.save(path)
        return ModelVersion.objects.create(
            organization=organization, name="Test model", version=version, slot=slot,
            architecture="campynet", weights_path=str(path),
            parameter_count=model.param_count, classes=model.meta["classes"],
            status=ModelVersion.Status.READY,
        )

    def test_deploying_makes_the_model_resolve_for_its_slot(self):
        version = self._version(self.org)
        self.assertIsNone(ModelVersion.deployed_for("campynet_fire", self.org))
        version.deploy(self.owner)
        resolved = ModelVersion.deployed_for("campynet_fire", self.org)
        self.assertEqual(resolved.pk, version.pk)

    def test_a_deployed_model_loads_and_runs(self):
        version = self._version(self.org)
        version.deploy(self.owner)
        clear_cache()
        loaded = ModelProvider(self.org).get("campynet_fire")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.param_count, version.parameter_count)

        import numpy as np
        prediction = loaded.predict(np.zeros((1, 3, 32, 32), dtype=np.float32))
        self.assertEqual(prediction.shape, (1, 3))

    def test_one_workspaces_model_is_invisible_to_another(self):
        version = self._version(self.org)
        version.deploy(self.owner)
        clear_cache()
        self.assertIsNone(ModelProvider(self.other).get("campynet_fire"))

    def test_deploying_retires_the_previous_version_in_that_slot(self):
        first = self._version(self.org, version="1.0.0")
        first.deploy(self.owner)
        second = self._version(self.org, version="1.1.0")
        second.deploy(self.owner)
        first.refresh_from_db()
        self.assertEqual(first.status, ModelVersion.Status.READY)
        self.assertEqual(ModelVersion.deployed_for("campynet_fire", self.org).pk, second.pk)

    def test_rollback_falls_back_to_the_classical_detectors(self):
        version = self._version(self.org)
        version.deploy(self.owner)
        version.retire()
        clear_cache()
        self.assertIsNone(ModelVersion.deployed_for("campynet_fire", self.org))
        self.assertIsNone(ModelProvider(self.org).get("campynet_fire"))


class APITests(TestCase):
    def setUp(self):
        self.org, self.owner = full_workspace("API Co")
        self.camera = demo_camera(self.org)
        self.event = Event.objects.create(
            organization=self.org, camera=self.camera, analytic="fire",
            event_type="fire_detected", severity="critical", confidence=0.9,
            title="API test event",
        )

    def _key(self, scopes):
        _, plaintext = APIKey.issue(self.org, "Test key", scopes, created_by=self.owner)
        return {"HTTP_AUTHORIZATION": f"Api-Key {plaintext}"}

    def test_anonymous_requests_are_rejected(self):
        self.assertIn(self.client.get("/api/v1/cameras/").status_code, (401, 403))

    def test_a_scoped_key_can_read_what_it_was_granted(self):
        response = self.client.get("/api/v1/cameras/", **self._key(["cameras.view"]))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 1)

    def test_a_key_cannot_read_what_it_was_not_granted(self):
        response = self.client.get("/api/v1/employees/", **self._key(["cameras.view"]))
        self.assertEqual(response.status_code, 403)

    def test_a_forged_key_is_rejected(self):
        response = self.client.get(
            "/api/v1/cameras/", HTTP_AUTHORIZATION="Api-Key cai_00000000.nope"
        )
        self.assertIn(response.status_code, (401, 403))

    def test_a_revoked_key_stops_working(self):
        key, plaintext = APIKey.issue(self.org, "Doomed", ["cameras.view"], created_by=self.owner)
        headers = {"HTTP_AUTHORIZATION": f"Api-Key {plaintext}"}
        self.assertEqual(self.client.get("/api/v1/cameras/", **headers).status_code, 200)
        key.revoke()
        self.assertIn(self.client.get("/api/v1/cameras/", **headers).status_code, (401, 403))

    def test_face_embeddings_are_never_serialised(self):
        from apps.cameras.models import Employee

        Employee.objects.create(
            organization=self.org, full_name="Enrolled Person",
            consent_given=True, face_embedding=[0.1] * 128,
        )
        response = self.client.get("/api/v1/employees/", **self._key(["employees.view"]))
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("face_embedding", response.content.decode())

    def test_stream_credentials_are_never_serialised(self):
        self.camera.stream_url = "rtsp://user:hunter2@10.0.0.5/stream"
        self.camera.username = "user"
        self.camera.password = "hunter2"
        self.camera.save()
        response = self.client.get("/api/v1/cameras/", **self._key(["cameras.view"]))
        body = response.content.decode()
        self.assertNotIn("hunter2", body)

    def test_edge_devices_can_ingest_a_detection(self):
        payload = {
            "camera": str(self.camera.uid), "analytic": "fire",
            "event_type": "edge_fire", "severity": "critical",
            "title": "Edge detection", "confidence": 0.88,
            "bounding_box": [1, 2, 30, 40],
        }
        response = self.client.post(
            "/api/v1/events/ingest/", payload, content_type="application/json",
            **self._key(["events.view"]),
        )
        self.assertEqual(response.status_code, 201, response.content)
        self.assertTrue(Event.objects.filter(event_type="edge_fire").exists())

    def test_ingest_rejects_an_unknown_analytic(self):
        response = self.client.post(
            "/api/v1/events/ingest/",
            {"camera": str(self.camera.uid), "analytic": "telepathy",
             "event_type": "x", "title": "x"},
            content_type="application/json", **self._key(["events.view"]),
        )
        self.assertEqual(response.status_code, 400)

    def test_ingest_rejects_a_camera_from_another_workspace(self):
        other_org, other_owner = full_workspace("Foreign Co")
        foreign_camera = demo_camera(other_org)
        response = self.client.post(
            "/api/v1/events/ingest/",
            {"camera": str(foreign_camera.uid), "analytic": "fire",
             "event_type": "x", "title": "x"},
            content_type="application/json", **self._key(["events.view"]),
        )
        self.assertEqual(response.status_code, 400)

    def test_the_api_root_documents_authentication(self):
        response = self.client.get("/api/v1/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("api_key", response.json()["authentication"])

    def test_the_openapi_schema_generates(self):
        response = self.client.get("/api/v1/schema/", **self._key(["cameras.view"]))
        self.assertEqual(response.status_code, 200)
