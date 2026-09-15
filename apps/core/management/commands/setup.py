"""First-run setup: create the database, then optionally fill and open it.

    python manage.py setup                  # tables + demo workspace
    python manage.py setup --no-demo        # tables only
    python manage.py setup --superuser      # tables, demo, then prompt for an admin

Everything here is safe to run again: migrations are skipped when applied and
the demo seed is idempotent.

It exists because the documented first run was four commands in an order
nothing enforced, and running them out of order gave a traceback rather than a
correction.
"""
from __future__ import annotations

import io

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import BaseCommand
from django.db import connection
from django.db.migrations.executor import MigrationExecutor


class Command(BaseCommand):
    help = "Prepare a fresh checkout: apply migrations and seed the demo workspace."

    def add_arguments(self, parser):
        parser.add_argument("--no-demo", action="store_true",
                            help="Create the tables but no demo content")
        parser.add_argument("--superuser", action="store_true",
                            help="Prompt for a superuser once the tables exist")

    def handle(self, *args, **options):
        pending = self.pending_migrations()
        self.stdout.write(self.style.MIGRATE_HEADING("Campy AI setup"))

        if pending:
            self.stdout.write(f"  Applying {pending} migration(s)…")
            call_command("migrate", verbosity=0)
            self.stdout.write(self.style.SUCCESS("  Database ready"))
        else:
            self.stdout.write("  Database already up to date")

        if options["no_demo"]:
            self.stdout.write("  Skipping demo content (--no-demo)")
        else:
            self.stdout.write("  Seeding packages, website content and a demo workspace…")
            # Capture rather than silence: the seed lists the demo credentials,
            # which belong in this command's own summary, printed once.
            call_command("seed_demo", stdout=io.StringIO())
            self.stdout.write(self.style.SUCCESS("  Demo workspace ready"))

        if options["superuser"]:
            self.stdout.write("")
            call_command("createsuperuser")

        self.report()

    @staticmethod
    def pending_migrations() -> int:
        executor = MigrationExecutor(connection)
        targets = executor.loader.graph.leaf_nodes()
        return len(executor.migration_plan(targets))

    def report(self) -> None:
        User = get_user_model()
        seeded = list(
            User.objects.filter(email__in=[
                "admin@campy.ai", "owner@acme-demo.com", "security@acme-demo.com",
                "operator@acme-demo.com", "ai@acme-demo.com", "viewer@acme-demo.com",
            ]).order_by("pk").values_list("email", flat=True)
        )

        self.stdout.write("")
        if seeded:
            self.stdout.write(self.style.MIGRATE_HEADING("Sign in at /accounts/login/"))
            for email in seeded:
                self.stdout.write(f"  {email:26s} CampyDemo!2024")
        elif not User.objects.exists():
            self.stdout.write(self.style.WARNING(
                "  No accounts yet — run: python manage.py createsuperuser"))

        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING("Next"))
        self.stdout.write("  python manage.py runserver   →   http://localhost:8000")
