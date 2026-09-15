#!/usr/bin/env python
"""Campy AI — Django management entrypoint."""
import os
import sys

#: Substrings every supported backend uses to say "that table isn't there".
MISSING_TABLE = ("no such table", "does not exist", "doesn't exist", "undefined table")


def unmigrated_database_advice(exc) -> str | None:
    """Turn a missing-table database error into the instruction that fixes it.

    Django prints "You have N unapplied migration(s)" and then runs the command
    anyway, so a first-time user meets a forty-line traceback instead of the one
    line they need. Returns None for database errors that mean something else.
    """
    if not any(hint in str(exc).lower() for hint in MISSING_TABLE):
        return None
    return (
        "\nThe database has not been set up yet, so this command has nothing "
        "to read.\n\n"
        "  python manage.py setup\n\n"
        "creates the tables and the demo workspace in one step. To do only the "
        "tables:\n\n"
        "  python manage.py migrate\n\n"
        f"(the database reported: {exc})\n"
    )


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "campyai.settings.dev")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and available "
            "on your PYTHONPATH environment variable? Did you forget to "
            "activate a virtual environment?"
        ) from exc

    from django.db.utils import OperationalError, ProgrammingError

    try:
        execute_from_command_line(sys.argv)
    except (OperationalError, ProgrammingError) as exc:
        advice = unmigrated_database_advice(exc)
        if advice is None:
            raise
        sys.stderr.write(advice)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
