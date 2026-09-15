"""Setting the project up for the first time.

Running `createsuperuser` before `migrate` produced a forty-line traceback
ending in "no such table: accounts_user". Django does warn first — and then
runs the command anyway — so the warning scrolls past and the traceback is what
the user is left holding.
"""
from __future__ import annotations

import io

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db.utils import OperationalError, ProgrammingError
from django.test import TestCase

from manage import unmigrated_database_advice


class UnmigratedDatabaseAdviceTests(TestCase):
    """The guard in manage.py, which turns the crash into an instruction."""

    def test_every_backend_phrasing_is_recognised(self):
        for message in (
            "no such table: accounts_user",                       # SQLite
            'relation "accounts_user" does not exist',            # PostgreSQL
            "Table 'campyai.accounts_user' doesn't exist",        # MySQL
            "UNDEFINED TABLE: accounts_user",                     # upper-cased
        ):
            with self.subTest(backend=message):
                advice = unmigrated_database_advice(OperationalError(message))
                self.assertIsNotNone(advice)
                self.assertIn("manage.py setup", advice)
                self.assertIn("manage.py migrate", advice)
                self.assertIn(message, advice, "the raw cause should still be shown")

    def test_it_reads_ProgrammingError_too(self):
        advice = unmigrated_database_advice(
            ProgrammingError('relation "cameras_camera" does not exist'))
        self.assertIsNotNone(advice)

    def test_other_database_failures_are_left_alone(self):
        """Only a missing table means "you have not migrated"."""
        for message in ("database is locked", "connection refused",
                        "disk I/O error", "too many connections"):
            with self.subTest(failure=message):
                self.assertIsNone(unmigrated_database_advice(OperationalError(message)))


class SuperuserTests(TestCase):
    """`createsuperuser` on an email-login project should ask for an email."""

    def test_it_asks_for_nothing_beyond_the_email_and_password(self):
        User = get_user_model()
        self.assertEqual(User.USERNAME_FIELD, "email")
        self.assertEqual(User.REQUIRED_FIELDS, [],
                         "the manager derives the username; asking for it "
                         "makes createsuperuser demand a discarded value")

    def test_a_superuser_can_be_created_without_naming_a_username(self):
        call_command("createsuperuser", email="admin@first.test", interactive=False,
                     stdout=io.StringIO())

        user = get_user_model().objects.get(email="admin@first.test")
        self.assertTrue(user.is_superuser)
        self.assertTrue(user.username, "a username should have been derived")
        self.assertEqual(user.platform_role, user.PlatformRole.SUPERADMIN)

    def test_two_addresses_with_the_same_local_part_both_work(self):
        """Why the manager leaves the username to be derived."""
        for email in ("chris@one.test", "chris@two.test"):
            call_command("createsuperuser", email=email, interactive=False,
                         stdout=io.StringIO())

        usernames = set(get_user_model().objects
                        .filter(email__startswith="chris@")
                        .values_list("username", flat=True))
        self.assertEqual(len(usernames), 2, f"usernames collided: {usernames}")

    def test_a_duplicate_address_is_still_refused(self):
        call_command("createsuperuser", email="dupe@first.test", interactive=False,
                     stdout=io.StringIO())
        with self.assertRaises(CommandError):
            call_command("createsuperuser", email="dupe@first.test", interactive=False,
                         stdout=io.StringIO())


class SetupCommandTests(TestCase):
    """One command for the first run, so the order cannot be got wrong."""

    def run_setup(self, *args):
        out = io.StringIO()
        call_command("setup", *args, stdout=out)
        return out.getvalue()

    def test_it_seeds_and_reports_the_accounts_to_sign_in_with(self):
        output = self.run_setup()

        self.assertIn("Database already up to date", output)
        self.assertIn("admin@campy.ai", output)
        self.assertIn("CampyDemo!2024", output)
        self.assertIn("runserver", output)
        self.assertTrue(get_user_model().objects.filter(email="admin@campy.ai").exists())

    def test_the_credentials_are_listed_once_not_twice(self):
        """The seed prints its own block; setup captures it and summarises."""
        self.assertEqual(self.run_setup().count("admin@campy.ai"), 1)

    def test_running_it_again_changes_nothing_and_does_not_fail(self):
        self.run_setup()
        before = get_user_model().objects.count()

        self.run_setup()

        self.assertEqual(get_user_model().objects.count(), before)

    def test_no_demo_leaves_an_empty_database_and_says_what_to_do_next(self):
        output = self.run_setup("--no-demo")

        self.assertIn("Skipping demo content", output)
        self.assertIn("createsuperuser", output)
        self.assertFalse(get_user_model().objects.exists())
