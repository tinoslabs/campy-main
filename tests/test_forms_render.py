"""Every form must actually reach the browser with its inputs.

A status-code sweep cannot catch a template that renders a form as an empty
``<form>`` with only a submit button, which is exactly what happened when the
shared field loop guarded on an undefined ``hidden_fields`` variable: Django's
``{% if %}`` swallows the resulting ``TypeError`` and silently skips every
field.  These tests assert on the rendered markup instead of the status code.
"""
from __future__ import annotations

import re

from django.template.loader import render_to_string
from django.test import TestCase
from django.urls import reverse

from apps.accounts.forms import LoginForm, SignupForm
from apps.accounts.models import User
from apps.cameras.models import Site
from tests.test_platform import full_workspace

CONTROL = re.compile(r"<(input|select|textarea)\b")


def controls(html: str) -> set[str]:
    return set(re.findall(r'<(?:input|select|textarea)[^>]*\bname="([^"]+)"', html))


def empty_choice_field(field) -> bool:
    """A multi-choice widget with nothing to choose from legitimately renders no input."""
    choices = getattr(field, "choices", None)
    return choices is not None and not list(choices)


class FormPartialTests(TestCase):
    def test_partial_renders_every_visible_field(self):
        for form_class in (LoginForm, SignupForm):
            with self.subTest(form=form_class.__name__):
                form = form_class()
                html = render_to_string("partials/form.html", {"form": form})
                self.assertEqual(controls(html), set(form.fields))

    def test_partial_still_honours_an_explicit_hidden_fields_list(self):
        html = render_to_string(
            "partials/form.html",
            {"form": LoginForm(), "hidden_fields": ["remember_me"]},
        )
        self.assertNotIn("remember_me", controls(html))
        self.assertIn("username", controls(html))


class FormPageTests(TestCase):
    """Real pages, fetched through the client, must carry usable inputs."""

    def setUp(self):
        self.organization, self.owner = full_workspace("Form Co")
        Site.objects.create(organization=self.organization, name="HQ")

    def assert_form_is_usable(self, url):
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200, url)
        html = response.content.decode()
        form = response.context.get("form") if response.context else None
        if form is None:
            self.fail(f"{url} rendered no form in its context")
        rendered = controls(html)
        missing = {
            name for name in set(form.fields) - rendered
            if not empty_choice_field(form.fields[name])
        }
        self.assertFalse(missing, f"{url} dropped fields: {sorted(missing)}")

    def test_public_auth_pages_render_their_inputs(self):
        for name in ("accounts:login", "accounts:signup", "accounts:password_reset"):
            with self.subTest(page=name):
                self.assert_form_is_usable(reverse(name))

    def test_workspace_create_pages_render_their_inputs(self):
        self.client.force_login(self.owner)
        for name in ("dashboard:site_create", "dashboard:camera_create",
                     "dashboard:alert_rule_create", "dashboard:dataset_create",
                     "dashboard:employee_create", "dashboard:report_create",
                     "accounts:invite_member", "accounts:role_create"):
            with self.subTest(page=name):
                self.assert_form_is_usable(reverse(name))

    def test_a_user_can_sign_in_through_the_rendered_form(self):
        User.objects.create_user(email="pilot@form.test", password="TestPass!2024",
                                 full_name="Pilot")
        page = self.client.get(reverse("accounts:login")).content.decode()
        self.assertIn('name="username"', page)
        self.assertIn('name="password"', page)
        response = self.client.post(
            reverse("accounts:login"),
            {"username": "pilot@form.test", "password": "TestPass!2024"},
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context["user"].is_authenticated)


class DetailPageRenderTests(TestCase):
    """Detail templates must survive the shape of the data they actually get."""

    def setUp(self):
        self.organization, self.owner = full_workspace("Detail Co")
        self.client.force_login(self.owner)

    def test_training_job_page_renders_for_a_classifier_run(self):
        """A classifier's epochs carry accuracy, never the embedding rank-1 keys.

        ``{{ a|default:b }}`` resolves ``b`` eagerly, and a missing dict key in
        a *filter argument* raises instead of falling back — so a template that
        offered rank-1 as the fallback crashed on every classification job.
        """
        from apps.training.models import Dataset, TrainingJob

        dataset = Dataset.objects.create(
            organization=self.organization, name="Fire set", task="classification",
            classes=["fire", "smoke", "none"],
        )
        job = TrainingJob.objects.create(
            organization=self.organization, dataset=dataset, name="Fire run",
            status=TrainingJob.Status.SUCCEEDED, epochs=2, current_epoch=2,
            history=[
                {"epoch": 1, "loss": 1.08, "lr": 0.002, "seconds": 0.44,
                 "train_accuracy": 0.407, "val_loss": 1.07, "val_accuracy": 0.370},
                {"epoch": 2, "loss": 0.56, "lr": 0.002, "seconds": 0.41,
                 "train_accuracy": 0.722, "val_loss": 0.61, "val_accuracy": 0.704},
            ],
        )
        response = self.client.get(job.get_absolute_url())
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "0.722")

    def test_training_job_page_renders_for_an_embedding_run(self):
        """The other branch: embeddings report rank-1, never accuracy."""
        from apps.training.models import Dataset, TrainingJob

        dataset = Dataset.objects.create(
            organization=self.organization, name="Face set", task="embedding",
            classes=["a", "b", "c"],
        )
        job = TrainingJob.objects.create(
            organization=self.organization, dataset=dataset, name="Face run",
            status=TrainingJob.Status.SUCCEEDED, epochs=1, current_epoch=1,
            history=[{"epoch": 1, "loss": 0.31, "lr": 0.001, "seconds": 0.9,
                      "train_rank1": 0.918, "val_rank1": 0.874}],
        )
        response = self.client.get(job.get_absolute_url())
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "0.918")
