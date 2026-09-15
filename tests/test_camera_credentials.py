"""Credentials, a stream URL, and the 401 that sits between them.

A camera's address arrives as one string from the vendor —
``rtsp://admin:Tinos@1122@192.168.29.209:554/Streaming/Channels/501`` — and the
form also offers a username and a password box. Filling in both used to inject
a second copy of the login, so the camera received ``admin:pass@admin:pass`` as
the password and answered 401 Unauthorized. Nothing on screen said which half
of the credentials was wrong, because neither was.
"""
from __future__ import annotations

from django.test import TestCase

from apps.cameras.models import Camera, Site
from apps.cameras.urls_util import has_credentials, inject_credentials, split_credentials
from apps.dashboard.forms import CameraForm
from tests.test_platform import full_workspace

PASTED = "rtsp://admin:Tinos@1122@192.168.29.209:554/Streaming/Channels/501"
CLEAN = "rtsp://192.168.29.209:554/Streaming/Channels/501"


class UrlCredentialTests(TestCase):
    def test_a_password_containing_an_at_sign_is_read_correctly(self):
        """The host is after the LAST @, which is how decoders read it too."""
        self.assertTrue(has_credentials(PASTED))
        self.assertEqual(split_credentials(PASTED), (CLEAN, "admin", "Tinos@1122"))

    def test_credentials_are_never_injected_twice(self):
        """The bug: a second copy becomes part of the password, and 401 follows."""
        self.assertEqual(inject_credentials(PASTED, "admin", "Tinos@1122"), PASTED)

    def test_characters_that_would_break_the_url_are_encoded(self):
        self.assertEqual(inject_credentials(CLEAN, "admin", "pa/ss"),
                         "rtsp://admin:pa%2Fss@192.168.29.209:554/Streaming/Channels/501")
        self.assertEqual(inject_credentials(CLEAN, "admin", "Tinos@1122"),
                         "rtsp://admin:Tinos%401122@192.168.29.209:554/Streaming/Channels/501")

    def test_a_url_with_no_credentials_to_add_is_untouched(self):
        self.assertEqual(inject_credentials(CLEAN, "", ""), CLEAN)
        self.assertEqual(inject_credentials("", "admin", "x"), "")
        self.assertEqual(inject_credentials("not a url", "admin", "x"), "not a url")

    def test_a_username_with_no_password_still_works(self):
        self.assertEqual(inject_credentials(CLEAN, "admin", ""),
                         "rtsp://admin@192.168.29.209:554/Streaming/Channels/501")


class CameraFormTests(TestCase):
    def setUp(self):
        self.organization, self.owner = full_workspace("Credential Co")
        self.site = Site.objects.create(organization=self.organization, name="Plant")

    def form(self, **overrides):
        data = {
            "site": self.site.pk, "name": "Gate", "protocol": Camera.Protocol.RTSP,
            "stream_url": PASTED, "username": "", "password": "",
            "target_fps": 6.0, "source_fps": 25.0, "rotation": 0,
            "resolution_width": 1920, "resolution_height": 1080,
            "enabled_analytics": [], "evidence_seconds": 10,
            "onvif_port": 80, "demo_scenario": "walk",
        }
        data.update(overrides)
        return CameraForm(self.organization, data=data)

    def test_a_pasted_url_has_its_login_moved_into_the_proper_fields(self):
        form = self.form()
        self.assertTrue(form.is_valid(), form.errors)

        self.assertEqual(form.cleaned_data["stream_url"], CLEAN)
        self.assertEqual(form.cleaned_data["username"], "admin")
        self.assertEqual(form.cleaned_data["password"], "Tinos@1122")

    def test_filling_in_both_does_not_produce_a_doubled_login(self):
        """The exact combination that produced the 401."""
        form = self.form(username="admin", password="Tinos@1122")
        self.assertTrue(form.is_valid(), form.errors)

        camera = form.save(commit=False)
        camera.organization = self.organization
        self.assertEqual(camera.stream_url, CLEAN)
        self.assertEqual(camera.connection_url,
                         "rtsp://admin:Tinos%401122@192.168.29.209:554/Streaming/Channels/501")
        self.assertNotIn("admin:Tinos@1122@admin", camera.connection_url)

    def test_a_typed_password_wins_over_the_one_in_the_url(self):
        """If someone corrects the password, the stale URL must not override it."""
        form = self.form(username="admin", password="Corrected!9")
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.cleaned_data["password"], "Corrected!9")

    def test_a_url_without_credentials_is_left_exactly_as_typed(self):
        form = self.form(stream_url=CLEAN, username="admin", password="Tinos@1122")
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.cleaned_data["stream_url"], CLEAN)

    def test_the_stored_url_no_longer_carries_the_password(self):
        form = self.form()
        self.assertTrue(form.is_valid(), form.errors)
        camera = form.save(commit=False)
        self.assertNotIn("Tinos", camera.stream_url)
        self.assertNotIn("Tinos", camera.masked_url)
