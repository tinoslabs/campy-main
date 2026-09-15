"""Tenancy, identity and role-based access control for Campy AI."""
from __future__ import annotations

import secrets
from datetime import timedelta

from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.core.validators import RegexValidator
from django.db import models, transaction
from django.utils import timezone
from django.utils.functional import cached_property

from apps.core.models import (
    SluggedModel,
    SoftDeleteModel,
    TimeStampedModel,
    UUIDModel,
)
from apps.core.utils import hash_secret, random_token

from . import permissions as perms


# ---------------------------------------------------------------------------
# Organisation (tenant)
# ---------------------------------------------------------------------------
class OrganizationQuerySet(models.QuerySet):
    def active(self):
        return self.filter(status=Organization.Status.ACTIVE)

    def for_user(self, user):
        if user is None or not user.is_authenticated:
            return self.none()
        if user.is_superadmin:
            return self
        return self.filter(memberships__user=user, memberships__status=Membership.Status.ACTIVE)


class Organization(UUIDModel, SluggedModel, TimeStampedModel, SoftDeleteModel):
    """A customer workspace. Every operational record is scoped to one."""

    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        TRIAL = "trial", "Trial"
        SUSPENDED = "suspended", "Suspended"
        CLOSED = "closed", "Closed"

    class Industry(models.TextChoices):
        OFFICE = "office", "Corporate office"
        RETAIL = "retail", "Retail / showroom"
        BANK = "bank", "Bank / finance"
        HOSPITAL = "hospital", "Hospital / clinic"
        EDUCATION = "education", "Educational institution"
        WAREHOUSE = "warehouse", "Warehouse / storage"
        MANUFACTURING = "manufacturing", "Manufacturing / plant"
        HOSPITALITY = "hospitality", "Hotel / hospitality"
        SERVICE_CENTER = "service_center", "Service centre"
        OTHER = "other", "Other"

    name = models.CharField(max_length=180)
    legal_name = models.CharField(max_length=220, blank=True)
    industry = models.CharField(max_length=32, choices=Industry.choices, default=Industry.OFFICE)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.TRIAL, db_index=True)

    logo = models.ImageField(upload_to="organizations/logos/", blank=True, null=True)
    website = models.URLField(blank=True)
    contact_email = models.EmailField(blank=True)
    contact_phone = models.CharField(max_length=32, blank=True)

    address_line1 = models.CharField(max_length=200, blank=True)
    address_line2 = models.CharField(max_length=200, blank=True)
    city = models.CharField(max_length=100, blank=True)
    state = models.CharField(max_length=100, blank=True)
    postal_code = models.CharField(max_length=20, blank=True)
    country = models.CharField(max_length=2, default="IN", help_text="ISO 3166-1 alpha-2")

    tax_id = models.CharField("GSTIN / VAT / Tax ID", max_length=64, blank=True)
    timezone = models.CharField(max_length=64, default="Asia/Kolkata")
    locale = models.CharField(max_length=16, default="en-us")

    # Operational preferences (retention, privacy, working hours, ...).
    settings = models.JSONField(default=dict, blank=True)

    onboarded_at = models.DateTimeField(null=True, blank=True)
    trial_ends_at = models.DateTimeField(null=True, blank=True)

    objects = OrganizationQuerySet.as_manager()

    class Meta:
        ordering = ["name"]
        indexes = [models.Index(fields=["status", "created_at"])]

    def __str__(self) -> str:
        return self.name

    def get_absolute_url(self) -> str:
        from django.urls import reverse

        return reverse("dashboard:organization_detail", args=[self.slug])

    @property
    def is_operational(self) -> bool:
        return self.status in {self.Status.ACTIVE, self.Status.TRIAL} and self.deleted_at is None

    @property
    def address(self) -> str:
        parts = [self.address_line1, self.address_line2, self.city, self.state, self.postal_code]
        return ", ".join(part for part in parts if part)

    def setting(self, key: str, default=None):
        return (self.settings or {}).get(key, default)

    def set_setting(self, key: str, value) -> None:
        data = dict(self.settings or {})
        data[key] = value
        self.settings = data

    @cached_property
    def active_subscription(self):
        return (
            self.subscriptions.filter(status__in=["active", "trialing", "past_due"])
            .select_related("package")
            .order_by("-created_at")
            .first()
        )

    def refresh_subscription(self):
        """Drop the cached subscription so the next read hits the database.

        Must be called by anything that starts, changes or cancels a plan:
        otherwise the in-memory instance keeps whatever it saw first — which,
        for a workspace being activated, is ``None``.
        """
        self.__dict__.pop("active_subscription", None)
        return self.active_subscription

    @property
    def package(self):
        subscription = self.active_subscription
        return subscription.package if subscription else None

    def has_feature(self, feature: str) -> bool:
        """Feature gate driven by the subscription package."""
        package = self.package
        if package is None:
            return False
        return package.has_feature(feature)

    def quota(self, key: str, default: int = 0) -> int:
        package = self.package
        if package is None:
            return default
        return package.quota(key, default)

    def bootstrap_default_roles(self) -> None:
        """Create the built-in role set for a brand-new workspace."""
        for template in perms.ROLE_TEMPLATES:
            Role.objects.get_or_create(
                organization=self,
                code=template.code,
                defaults={
                    "name": template.name,
                    "description": template.description,
                    "permissions": list(template.permissions),
                    "rank": template.rank,
                    "is_system": True,
                },
            )


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------
class UserManager(BaseUserManager):
    use_in_migrations = True

    def _create_user(self, email: str, password: str | None, **extra):
        if not email:
            raise ValueError("Users must have an email address")
        email = self.normalize_email(email).lower()
        # Deliberately do NOT set a username here: two different addresses can
        # share a local-part (rival@example.com and rival@other.com), and a
        # forced username would collide. `User.save` derives a unique one.
        user = self.model(email=email, **extra)
        if password:
            user.set_password(password)
        else:
            user.set_unusable_password()
        user.save(using=self._db)
        return user

    def create_user(self, email: str, password: str | None = None, **extra):
        extra.setdefault("platform_role", User.PlatformRole.CUSTOMER)
        extra.setdefault("is_staff", False)
        extra.setdefault("is_superuser", False)
        return self._create_user(email, password, **extra)

    def create_superuser(self, email: str, password: str | None = None, **extra):
        extra.setdefault("platform_role", User.PlatformRole.SUPERADMIN)
        extra.setdefault("is_staff", True)
        extra.setdefault("is_superuser", True)
        extra.setdefault("is_active", True)
        extra.setdefault("email_verified", True)
        if extra["is_staff"] is not True:
            raise ValueError("Superuser must have is_staff=True.")
        if extra["is_superuser"] is not True:
            raise ValueError("Superuser must have is_superuser=True.")
        return self._create_user(email, password, **extra)


phone_validator = RegexValidator(
    r"^[0-9+()\-\s]{6,32}$", "Enter a valid phone number."
)


class User(AbstractBaseUser, PermissionsMixin, UUIDModel, TimeStampedModel):
    """Campy AI account. Email is the login handle; ``username`` stays for display."""

    class PlatformRole(models.TextChoices):
        SUPERADMIN = "superadmin", "Super Administrator"
        PLATFORM_STAFF = "platform_staff", "Platform Staff"
        CUSTOMER = "customer", "Customer User"

    email = models.EmailField(unique=True, db_index=True)
    username = models.CharField(max_length=150, unique=True)
    full_name = models.CharField(max_length=180, blank=True)
    phone = models.CharField(max_length=32, blank=True, validators=[phone_validator])
    job_title = models.CharField(max_length=120, blank=True)
    avatar = models.ImageField(upload_to="users/avatars/", blank=True, null=True)

    platform_role = models.CharField(
        max_length=20, choices=PlatformRole.choices, default=PlatformRole.CUSTOMER, db_index=True
    )

    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False, help_text="Access to the Django admin site.")
    email_verified = models.BooleanField(default=False)
    phone_verified = models.BooleanField(default=False)

    mfa_enabled = models.BooleanField(default=False)
    mfa_secret = models.CharField(max_length=64, blank=True)

    timezone = models.CharField(max_length=64, default="Asia/Kolkata")
    locale = models.CharField(max_length=16, default="en-us")
    theme = models.CharField(
        max_length=10,
        choices=[("system", "System"), ("light", "Light"), ("dark", "Dark")],
        default="system",
    )
    notification_preferences = models.JSONField(default=dict, blank=True)

    last_seen_at = models.DateTimeField(null=True, blank=True)
    last_login_ip = models.GenericIPAddressField(null=True, blank=True)
    failed_login_attempts = models.PositiveIntegerField(default=0)
    locked_until = models.DateTimeField(null=True, blank=True)

    # Remembers which workspace the user was last looking at.
    active_organization = models.ForeignKey(
        Organization, null=True, blank=True, on_delete=models.SET_NULL, related_name="+"
    )

    objects = UserManager()

    USERNAME_FIELD = "email"
    #: Nothing beyond the email and a password. The manager deliberately leaves
    #: `username` unset so `save` can derive a unique one, so demanding it here
    #: only made `createsuperuser` ask for a value the model discards.
    REQUIRED_FIELDS: list[str] = []

    class Meta:
        ordering = ["full_name", "email"]
        indexes = [models.Index(fields=["platform_role", "is_active"])]

    def __str__(self) -> str:
        return self.full_name or self.email

    def save(self, *args, **kwargs):
        self.email = (self.email or "").lower().strip()
        self.username = self._unique_username(self.username)
        super().save(*args, **kwargs)

    def _unique_username(self, preferred: str = "") -> str:
        """Return a free username, keeping *preferred* when it is available.

        Called on every save so a supplied username is validated too, not just
        a derived one — otherwise two accounts whose emails share a local-part
        collide on the unique constraint.
        """
        base = (preferred or "").strip() or (self.email.split("@")[0][:140] or "user")
        if not User.objects.filter(username=base).exclude(pk=self.pk).exists():
            return base
        counter = 2
        while User.objects.filter(username=f"{base}{counter}").exclude(pk=self.pk).exists():
            counter += 1
        return f"{base}{counter}"

    # -- identity ---------------------------------------------------------
    def get_full_name(self) -> str:
        return self.full_name or self.username

    def get_short_name(self) -> str:
        return (self.full_name or self.username).split(" ")[0]

    @property
    def initials(self) -> str:
        source = (self.full_name or self.username or self.email).strip()
        chunks = [part for part in source.replace(".", " ").split(" ") if part]
        if len(chunks) >= 2:
            return (chunks[0][0] + chunks[1][0]).upper()
        return source[:2].upper()

    @property
    def is_superadmin(self) -> bool:
        return self.platform_role == self.PlatformRole.SUPERADMIN or self.is_superuser

    @property
    def is_platform_team(self) -> bool:
        return self.is_superadmin or self.platform_role == self.PlatformRole.PLATFORM_STAFF

    @property
    def is_locked(self) -> bool:
        return bool(self.locked_until and self.locked_until > timezone.now())

    def register_failed_login(self, threshold: int = 8, lock_minutes: int = 15) -> None:
        self.failed_login_attempts += 1
        if self.failed_login_attempts >= threshold:
            self.locked_until = timezone.now() + timedelta(minutes=lock_minutes)
            self.failed_login_attempts = 0
        self.save(update_fields=["failed_login_attempts", "locked_until"])

    def register_successful_login(self, ip: str | None = None) -> None:
        self.failed_login_attempts = 0
        self.locked_until = None
        self.last_seen_at = timezone.now()
        if ip:
            self.last_login_ip = ip
        self.save(update_fields=["failed_login_attempts", "locked_until", "last_seen_at", "last_login_ip"])

    # -- tenancy ----------------------------------------------------------
    @property
    def organizations(self):
        return Organization.objects.for_user(self).distinct()

    def membership_for(self, organization) -> "Membership | None":
        if organization is None:
            return None
        return (
            Membership.objects.filter(
                user=self, organization=organization, status=Membership.Status.ACTIVE
            )
            .select_related("role", "organization")
            .first()
        )

    def resolve_organization(self, requested=None) -> Organization | None:
        """Pick the workspace this request should operate in."""
        if requested is not None:
            if self.is_superadmin or self.membership_for(requested):
                return requested
            return None
        if self.active_organization_id:
            candidate = self.active_organization
            if candidate and (self.is_superadmin or self.membership_for(candidate)):
                return candidate
        return self.organizations.first()

    # -- permissions ------------------------------------------------------
    def permission_codes(self, organization) -> set[str]:
        """Effective permission codes for this user inside *organization*."""
        if self.is_superadmin:
            return set(perms.ALL_CODES)
        codes: set[str] = set()
        if self.platform_role == self.PlatformRole.PLATFORM_STAFF:
            codes.update(
                {
                    "platform.organizations", "platform.packages", "platform.users",
                    "platform.cms", "platform.models",
                }
            )
        membership = self.membership_for(organization)
        if membership:
            codes.update(membership.permission_codes())
        return codes

    def has_campy_perm(self, code: str, organization=None, *, check_feature: bool = True) -> bool:
        """Does this user hold *code* in *organization*?

        Platform permissions (``platform.*``) are workspace-independent.  Every
        other permission is checked against the membership **and** — unless
        ``check_feature=False`` — against the workspace's subscription package.
        """
        if self.is_superadmin:
            return True
        if not self.is_active:
            return False
        if code.startswith("platform."):
            return code in self.permission_codes(None)
        if organization is None:
            return False
        if code not in self.permission_codes(organization):
            return False
        if check_feature:
            required = perms.PERMISSION_MAP.get(code)
            if required and required.requires_feature:
                return organization.has_feature(required.requires_feature)
        return True

    def has_any_campy_perm(self, codes, organization=None) -> bool:
        return any(self.has_campy_perm(code, organization) for code in codes)


# ---------------------------------------------------------------------------
# Roles & memberships
# ---------------------------------------------------------------------------
class Role(UUIDModel, TimeStampedModel):
    """A named bundle of permission codes, scoped to one organisation.

    ``organization=None`` marks a *platform template* that new workspaces copy.
    """

    organization = models.ForeignKey(
        Organization, null=True, blank=True, on_delete=models.CASCADE, related_name="roles"
    )
    code = models.SlugField(max_length=60)
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    permissions = models.JSONField(default=list, blank=True)
    rank = models.PositiveIntegerField(
        default=100, help_text="Lower rank = more senior. Prevents privilege escalation."
    )
    is_system = models.BooleanField(default=False, help_text="Built-in roles cannot be deleted.")
    color = models.CharField(max_length=7, default="#4f46e5")

    class Meta:
        ordering = ["rank", "name"]
        constraints = [
            models.UniqueConstraint(fields=["organization", "code"], name="uniq_role_per_org")
        ]

    def __str__(self) -> str:
        return f"{self.name} ({self.organization or 'template'})"

    def save(self, *args, **kwargs):
        self.permissions = perms.validate(self.permissions)
        super().save(*args, **kwargs)

    @cached_property
    def expanded_permissions(self) -> set[str]:
        return perms.expand(self.permissions)

    @property
    def permission_count(self) -> int:
        return len(self.expanded_permissions)

    @property
    def is_owner_role(self) -> bool:
        return self.code == perms.DEFAULT_OWNER_ROLE

    def grants(self, code: str) -> bool:
        return code in self.expanded_permissions


class MembershipQuerySet(models.QuerySet):
    def active(self):
        return self.filter(status=Membership.Status.ACTIVE)


class Membership(UUIDModel, TimeStampedModel):
    """Join table binding a :class:`User` to an :class:`Organization` with a :class:`Role`."""

    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        SUSPENDED = "suspended", "Suspended"
        REMOVED = "removed", "Removed"

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="memberships")
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name="memberships")
    role = models.ForeignKey(Role, on_delete=models.PROTECT, related_name="memberships")
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.ACTIVE, db_index=True)

    # Optional per-member overrides on top of the role.
    extra_permissions = models.JSONField(default=list, blank=True)
    denied_permissions = models.JSONField(default=list, blank=True)

    # Restrict a member to specific sites (empty = all sites in the workspace).
    site_scope = models.ManyToManyField("cameras.Site", blank=True, related_name="scoped_memberships")

    title = models.CharField(max_length=120, blank=True)
    invited_by = models.ForeignKey(
        User, null=True, blank=True, on_delete=models.SET_NULL, related_name="invited_memberships"
    )
    joined_at = models.DateTimeField(default=timezone.now)
    last_active_at = models.DateTimeField(null=True, blank=True)

    objects = MembershipQuerySet.as_manager()

    class Meta:
        ordering = ["role__rank", "user__full_name"]
        constraints = [
            models.UniqueConstraint(fields=["user", "organization"], name="uniq_membership_per_org")
        ]
        indexes = [models.Index(fields=["organization", "status"])]

    def __str__(self) -> str:
        return f"{self.user} @ {self.organization} — {self.role.name}"

    def save(self, *args, **kwargs):
        self.extra_permissions = perms.validate(self.extra_permissions)
        self.denied_permissions = perms.validate(self.denied_permissions)
        super().save(*args, **kwargs)

    def permission_codes(self) -> set[str]:
        if self.status != self.Status.ACTIVE:
            return set()
        codes = set(self.role.expanded_permissions)
        codes |= perms.expand(self.extra_permissions)
        codes -= perms.expand(self.denied_permissions)
        return codes

    @property
    def is_owner(self) -> bool:
        return self.role.is_owner_role

    @property
    def scoped_site_ids(self) -> list[int] | None:
        """``None`` means every site; otherwise the allow-list of site ids."""
        ids = list(self.site_scope.values_list("id", flat=True))
        return ids or None


class Invitation(UUIDModel, TimeStampedModel):
    """Email invitation to join a workspace with a pre-assigned role."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        ACCEPTED = "accepted", "Accepted"
        REVOKED = "revoked", "Revoked"
        EXPIRED = "expired", "Expired"

    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name="invitations")
    email = models.EmailField()
    role = models.ForeignKey(Role, on_delete=models.CASCADE, related_name="invitations")
    token = models.CharField(max_length=64, unique=True, default=random_token)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING, db_index=True)
    message = models.TextField(blank=True)
    invited_by = models.ForeignKey(User, null=True, on_delete=models.SET_NULL, related_name="sent_invitations")
    expires_at = models.DateTimeField()
    accepted_at = models.DateTimeField(null=True, blank=True)
    accepted_by = models.ForeignKey(
        User, null=True, blank=True, on_delete=models.SET_NULL, related_name="accepted_invitations"
    )

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["organization", "status"])]

    def __str__(self) -> str:
        return f"Invite {self.email} → {self.organization}"

    def save(self, *args, **kwargs):
        self.email = (self.email or "").lower().strip()
        if not self.expires_at:
            self.expires_at = timezone.now() + timedelta(days=14)
        super().save(*args, **kwargs)

    @property
    def is_expired(self) -> bool:
        return timezone.now() >= self.expires_at

    @property
    def is_actionable(self) -> bool:
        return self.status == self.Status.PENDING and not self.is_expired

    def get_accept_url(self) -> str:
        from django.urls import reverse

        return reverse("accounts:accept_invitation", args=[self.token])

    @transaction.atomic
    def accept(self, user: User) -> Membership:
        if not self.is_actionable:
            raise ValueError("This invitation is no longer valid.")
        membership, _ = Membership.objects.update_or_create(
            user=user,
            organization=self.organization,
            defaults={
                "role": self.role,
                "status": Membership.Status.ACTIVE,
                "invited_by": self.invited_by,
                "joined_at": timezone.now(),
            },
        )
        self.status = self.Status.ACCEPTED
        self.accepted_at = timezone.now()
        self.accepted_by = user
        self.save(update_fields=["status", "accepted_at", "accepted_by", "updated_at"])
        if user.active_organization_id is None:
            user.active_organization = self.organization
            user.save(update_fields=["active_organization"])
        return membership


# ---------------------------------------------------------------------------
# Machine credentials
# ---------------------------------------------------------------------------
class APIKey(UUIDModel, TimeStampedModel):
    """Server-to-server credential for edge devices and integrations.

    The plaintext key is shown exactly once; only its SHA-256 digest is stored.
    """

    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name="api_keys")
    name = models.CharField(max_length=120)
    prefix = models.CharField(max_length=12, unique=True, db_index=True)
    key_hash = models.CharField(max_length=64, db_index=True)
    scopes = models.JSONField(default=list, blank=True, help_text="Permission codes granted to this key.")
    created_by = models.ForeignKey(User, null=True, on_delete=models.SET_NULL, related_name="+")
    last_used_at = models.DateTimeField(null=True, blank=True)
    last_used_ip = models.GenericIPAddressField(null=True, blank=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    revoked_at = models.DateTimeField(null=True, blank=True)
    request_count = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "API key"

    def __str__(self) -> str:
        return f"{self.name} ({self.prefix}…)"

    @property
    def is_active(self) -> bool:
        if self.revoked_at:
            return False
        if self.expires_at and self.expires_at <= timezone.now():
            return False
        return True

    def revoke(self) -> None:
        self.revoked_at = timezone.now()
        self.save(update_fields=["revoked_at", "updated_at"])

    def grants(self, code: str) -> bool:
        return code in perms.expand(self.scopes)

    @classmethod
    def issue(cls, organization: Organization, name: str, scopes: list[str], created_by=None, expires_at=None):
        """Create a key and return ``(instance, plaintext)``."""
        prefix = f"cai_{secrets.token_hex(4)}"
        secret = random_token(48)
        plaintext = f"{prefix}.{secret}"
        instance = cls.objects.create(
            organization=organization,
            name=name,
            prefix=prefix,
            key_hash=hash_secret(plaintext),
            scopes=perms.validate(scopes),
            created_by=created_by,
            expires_at=expires_at,
        )
        return instance, plaintext


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------
class AuditLog(TimeStampedModel):
    """Immutable record of who did what, where and when."""

    class Action(models.TextChoices):
        CREATE = "create", "Created"
        UPDATE = "update", "Updated"
        DELETE = "delete", "Deleted"
        LOGIN = "login", "Signed in"
        LOGIN_FAILED = "login_failed", "Sign-in failed"
        LOGOUT = "logout", "Signed out"
        VIEW = "view", "Viewed"
        EXPORT = "export", "Exported"
        DEPLOY = "deploy", "Deployed"
        PAYMENT = "payment", "Payment"
        PERMISSION = "permission", "Permission change"
        SECURITY = "security", "Security event"

    organization = models.ForeignKey(
        Organization, null=True, blank=True, on_delete=models.SET_NULL, related_name="audit_logs"
    )
    actor = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL, related_name="audit_logs")
    actor_label = models.CharField(max_length=180, blank=True, help_text="Denormalised — survives user deletion.")
    action = models.CharField(max_length=20, choices=Action.choices, db_index=True)
    target_type = models.CharField(max_length=80, blank=True, db_index=True)
    target_id = models.CharField(max_length=64, blank=True)
    target_label = models.CharField(max_length=220, blank=True)
    summary = models.CharField(max_length=400, blank=True)
    changes = models.JSONField(default=dict, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.CharField(max_length=300, blank=True)
    is_sensitive = models.BooleanField(default=False)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["organization", "-created_at"]),
            models.Index(fields=["actor", "-created_at"]),
            models.Index(fields=["target_type", "target_id"]),
        ]

    def __str__(self) -> str:
        return f"{self.actor_label or 'system'} {self.action} {self.target_label}"

    @classmethod
    def record(
        cls,
        *,
        action: str,
        actor=None,
        organization=None,
        target=None,
        summary: str = "",
        changes: dict | None = None,
        request=None,
        sensitive: bool = False,
    ) -> "AuditLog":
        from apps.core.utils import client_ip

        target_type = target_id = target_label = ""
        if target is not None:
            target_type = target.__class__.__name__
            target_id = str(getattr(target, "pk", "") or "")
            target_label = str(target)[:220]
        return cls.objects.create(
            organization=organization,
            actor=actor if getattr(actor, "pk", None) else None,
            actor_label=str(actor)[:180] if actor else "system",
            action=action,
            target_type=target_type,
            target_id=target_id,
            target_label=target_label,
            summary=summary[:400],
            changes=changes or {},
            ip_address=client_ip(request) if request else None,
            user_agent=(request.META.get("HTTP_USER_AGENT", "")[:300] if request else ""),
            is_sensitive=sensitive,
        )


class EmailToken(TimeStampedModel):
    """Single-use token for email verification and password reset."""

    class Purpose(models.TextChoices):
        VERIFY_EMAIL = "verify_email", "Verify email"
        RESET_PASSWORD = "reset_password", "Reset password"

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="email_tokens")
    purpose = models.CharField(max_length=20, choices=Purpose.choices)
    token = models.CharField(max_length=64, unique=True, default=random_token, db_index=True)
    expires_at = models.DateTimeField()
    used_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def save(self, *args, **kwargs):
        if not self.expires_at:
            hours = 48 if self.purpose == self.Purpose.VERIFY_EMAIL else 2
            self.expires_at = timezone.now() + timedelta(hours=hours)
        super().save(*args, **kwargs)

    @property
    def is_valid(self) -> bool:
        return self.used_at is None and timezone.now() < self.expires_at

    def consume(self) -> None:
        self.used_at = timezone.now()
        self.save(update_fields=["used_at", "updated_at"])
