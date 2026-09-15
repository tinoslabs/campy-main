"""Reusable abstract models shared by every Campy AI app."""
from __future__ import annotations

import uuid

from django.conf import settings
from django.db import models
from django.utils import timezone
from django.utils.text import slugify


class TimeStampedModel(models.Model):
    """Adds ``created_at`` / ``updated_at`` bookkeeping."""

    created_at = models.DateTimeField(default=timezone.now, editable=False, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class UUIDModel(models.Model):
    """Public-facing opaque identifier — safe to expose in URLs and the API."""

    uid = models.UUIDField(default=uuid.uuid4, unique=True, editable=False, db_index=True)

    class Meta:
        abstract = True


class SoftDeleteQuerySet(models.QuerySet):
    def alive(self):
        return self.filter(deleted_at__isnull=True)

    def dead(self):
        return self.filter(deleted_at__isnull=False)

    def delete(self):
        return self.update(deleted_at=timezone.now())

    def hard_delete(self):
        return super().delete()


class SoftDeleteManager(models.Manager.from_queryset(SoftDeleteQuerySet)):
    def get_queryset(self):
        return super().get_queryset().filter(deleted_at__isnull=True)


class SoftDeleteModel(models.Model):
    """Records are archived rather than destroyed — required for audit trails."""

    deleted_at = models.DateTimeField(null=True, blank=True, editable=False, db_index=True)

    objects = SoftDeleteManager()
    all_objects = SoftDeleteQuerySet.as_manager()

    class Meta:
        abstract = True

    def delete(self, using=None, keep_parents=False):  # noqa: D102
        self.deleted_at = timezone.now()
        self.save(update_fields=["deleted_at", "updated_at"] if hasattr(self, "updated_at") else ["deleted_at"])

    def restore(self):
        self.deleted_at = None
        self.save(update_fields=["deleted_at"])

    def hard_delete(self, using=None, keep_parents=False):
        return super().delete(using=using, keep_parents=keep_parents)


class SluggedModel(models.Model):
    """Auto-populates a unique slug from ``slug_source_field``."""

    slug = models.SlugField(max_length=180, unique=True, blank=True)

    slug_source_field = "name"

    class Meta:
        abstract = True

    def build_slug(self) -> str:
        base = slugify(getattr(self, self.slug_source_field, "") or "item")[:150] or "item"
        candidate = base
        model = self.__class__
        counter = 2
        while model._default_manager.filter(slug=candidate).exclude(pk=self.pk).exists():
            candidate = f"{base}-{counter}"
            counter += 1
        return candidate

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = self.build_slug()
        super().save(*args, **kwargs)


class OrganizationOwnedQuerySet(models.QuerySet):
    def for_organization(self, organization):
        if organization is None:
            return self.none()
        return self.filter(organization=organization)

    def for_request(self, request):
        """Superadmins see everything; everyone else is scoped to their org."""
        user = getattr(request, "user", None)
        if user is not None and getattr(user, "is_superadmin", False):
            return self
        return self.for_organization(getattr(request, "organization", None))


class OrganizationOwnedModel(models.Model):
    """Every tenant-scoped record hangs off an :class:`~apps.accounts.models.Organization`."""

    organization = models.ForeignKey(
        "accounts.Organization",
        on_delete=models.CASCADE,
        related_name="%(class)s_set",
        db_index=True,
    )

    objects = OrganizationOwnedQuerySet.as_manager()

    class Meta:
        abstract = True


class ActorStampedModel(models.Model):
    """Tracks which user created / last touched a record."""

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="%(app_label)s_%(class)s_created",
    )
    updated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="%(app_label)s_%(class)s_updated",
    )

    class Meta:
        abstract = True
