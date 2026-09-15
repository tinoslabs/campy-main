"""Give Role a public UUID.

Adding a unique column to a populated table needs three steps: create it
nullable, backfill a distinct value per row, then apply the constraint.
Doing it in one step would fail on any deployment that already has roles.
"""
import uuid

from django.db import migrations, models


def populate_uids(apps, schema_editor):
    Role = apps.get_model("accounts", "Role")
    for role in Role.objects.filter(uid__isnull=True).iterator(chunk_size=200):
        role.uid = uuid.uuid4()
        role.save(update_fields=["uid"])


def noop(apps, schema_editor):
    """Reversing drops the column outright, so there is nothing to undo."""


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0002_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="role",
            name="uid",
            field=models.UUIDField(null=True, editable=False, db_index=True),
        ),
        migrations.RunPython(populate_uids, noop),
        migrations.AlterField(
            model_name="role",
            name="uid",
            field=models.UUIDField(
                default=uuid.uuid4, editable=False, unique=True, db_index=True
            ),
        ),
    ]
