"""Campy AI permission registry and role templates.

Django's built-in ``auth.Permission`` is table-driven and per-model; Campy AI
needs *tenant-scoped*, *feature-scoped* permissions that can also be gated by
the organisation's subscription package.  This module is the single source of
truth for every capability in the product.

A permission code looks like ``"cameras.manage"``: ``<module>.<action>``.
Roles hold a flat list of codes; ``"*"`` means "everything in this workspace".
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Permission:
    code: str
    label: str
    module: str
    description: str = ""
    # Feature flag from the billing package required for this permission to
    # be usable. ``None`` means it is available on every plan.
    requires_feature: str | None = None


@dataclass(frozen=True)
class RoleTemplate:
    code: str
    name: str
    description: str
    permissions: list[str] = field(default_factory=list)
    # Lower rank == more powerful. Used to stop privilege escalation.
    rank: int = 100
    is_billing_role: bool = False


# ---------------------------------------------------------------------------
# Permission catalogue
# ---------------------------------------------------------------------------
MODULES = {
    "org": "Organisation",
    "people": "People & Roles",
    "cameras": "Cameras & Sites",
    "zones": "Zones & Geofences",
    "live": "Live Monitoring",
    "events": "Events & Incidents",
    "alerts": "Alert Rules",
    "employees": "Employee Directory",
    "analytics": "Analytics & Reports",
    "training": "AI Training",
    "models": "Model Registry",
    "billing": "Billing & Packages",
    "cms": "Content Management",
    "api": "API & Integrations",
    "audit": "Audit & Compliance",
    "platform": "Platform Administration",
}

PERMISSIONS: tuple[Permission, ...] = (
    # -- organisation ------------------------------------------------------
    Permission("org.view", "View organisation profile", "org"),
    Permission("org.manage", "Edit organisation settings", "org"),
    Permission("org.delete", "Delete / close the organisation", "org"),
    # -- people ------------------------------------------------------------
    Permission("people.view", "View members", "people"),
    Permission("people.invite", "Invite new members", "people"),
    Permission("people.manage", "Edit members and assign roles", "people"),
    Permission("people.remove", "Remove members", "people"),
    Permission("people.roles", "Create and edit custom roles", "people"),
    # -- cameras -----------------------------------------------------------
    Permission("cameras.view", "View cameras and sites", "cameras"),
    Permission("cameras.manage", "Add, edit and remove cameras and sites", "cameras"),
    Permission("cameras.control", "Start / stop analytics on a camera", "cameras"),
    Permission("cameras.credentials", "View or edit stream credentials", "cameras"),
    # -- zones -------------------------------------------------------------
    Permission("zones.view", "View zones and geofences", "zones"),
    Permission(
        "zones.manage",
        "Draw and edit geofences",
        "zones",
        requires_feature="geofencing",
    ),
    # -- live --------------------------------------------------------------
    Permission("live.view", "Open the live monitoring wall", "live"),
    Permission("live.snapshot", "Capture and download snapshots", "live"),
    # -- events ------------------------------------------------------------
    Permission("events.view", "View detections and incidents", "events"),
    Permission("events.acknowledge", "Acknowledge incidents", "events"),
    Permission("events.resolve", "Resolve and close incidents", "events"),
    Permission("events.export", "Export event data and evidence", "events"),
    Permission("events.delete", "Delete events and evidence", "events"),
    # -- alerts ------------------------------------------------------------
    Permission("alerts.view", "View alert rules", "alerts"),
    Permission("alerts.manage", "Create and edit alert rules and escalation", "alerts"),
    Permission("alerts.channels", "Configure notification channels", "alerts"),
    # -- employees ---------------------------------------------------------
    Permission("employees.view", "View the employee directory", "employees"),
    Permission("employees.manage", "Add and edit employees", "employees"),
    Permission(
        "employees.enroll",
        "Enrol employee faces for recognition",
        "employees",
        requires_feature="face_recognition",
    ),
    # -- analytics ---------------------------------------------------------
    Permission("analytics.view", "View dashboards and analytics", "analytics"),
    Permission("analytics.export", "Export reports", "analytics", requires_feature="reports"),
    Permission("analytics.schedule", "Schedule recurring reports", "analytics", requires_feature="reports"),
    # -- training ----------------------------------------------------------
    Permission("training.view", "View datasets and training jobs", "training", requires_feature="custom_training"),
    Permission("training.datasets", "Create and edit datasets", "training", requires_feature="custom_training"),
    Permission("training.annotate", "Label and annotate samples", "training", requires_feature="custom_training"),
    Permission("training.run", "Launch training jobs", "training", requires_feature="custom_training"),
    # -- models ------------------------------------------------------------
    Permission("models.view", "View the model registry", "models"),
    Permission("models.deploy", "Promote a model version to production", "models", requires_feature="custom_training"),
    Permission("models.rollback", "Roll back a deployed model", "models", requires_feature="custom_training"),
    # -- billing -----------------------------------------------------------
    Permission("billing.view", "View plan, invoices and usage", "billing"),
    Permission("billing.manage", "Change plan and payment methods", "billing"),
    Permission("billing.invoices", "Download invoices and receipts", "billing"),
    # -- cms ---------------------------------------------------------------
    Permission("cms.view", "View website content", "cms"),
    Permission("cms.manage", "Create and edit pages, posts and media", "cms"),
    Permission("cms.publish", "Publish and unpublish content", "cms"),
    # -- api ---------------------------------------------------------------
    Permission("api.view", "View API keys and webhooks", "api"),
    Permission("api.manage", "Create and revoke API keys and webhooks", "api", requires_feature="api_access"),
    # -- audit -------------------------------------------------------------
    Permission("audit.view", "View the audit log", "audit"),
    Permission("audit.export", "Export audit records", "audit"),
    # -- platform (Campy AI staff only) ------------------------------------
    Permission("platform.organizations", "Manage every customer organisation", "platform"),
    Permission("platform.packages", "Create and edit subscription packages", "platform"),
    Permission("platform.users", "Manage every user on the platform", "platform"),
    Permission("platform.models", "Manage global base models", "platform"),
    Permission("platform.cms", "Manage the marketing website", "platform"),
    Permission("platform.settings", "Edit platform-wide settings", "platform"),
    Permission("platform.impersonate", "Sign in as a customer for support", "platform"),
)

PERMISSION_MAP: dict[str, Permission] = {perm.code: perm for perm in PERMISSIONS}
ALL_CODES: tuple[str, ...] = tuple(PERMISSION_MAP)
WILDCARD = "*"


def permissions_by_module() -> dict[str, list[Permission]]:
    grouped: dict[str, list[Permission]] = {key: [] for key in MODULES}
    for perm in PERMISSIONS:
        grouped.setdefault(perm.module, []).append(perm)
    return grouped


def expand(codes: list[str] | tuple[str, ...]) -> set[str]:
    """Resolve wildcards and ``module.*`` patterns into concrete codes."""
    resolved: set[str] = set()
    for code in codes or ():
        if code == WILDCARD:
            return set(ALL_CODES)
        if code.endswith(".*"):
            prefix = code[:-1]
            resolved.update(c for c in ALL_CODES if c.startswith(prefix))
        elif code in PERMISSION_MAP:
            resolved.add(code)
    return resolved


def validate(codes: list[str]) -> list[str]:
    """Drop unknown codes so a stale role can never grant a phantom permission."""
    return [code for code in codes or [] if code == WILDCARD or code.endswith(".*") or code in PERMISSION_MAP]


# ---------------------------------------------------------------------------
# Built-in role templates
# ---------------------------------------------------------------------------
VIEWER_CODES = [
    "org.view", "people.view", "cameras.view", "zones.view", "live.view",
    "events.view", "alerts.view", "employees.view", "analytics.view", "models.view",
]

ROLE_TEMPLATES: tuple[RoleTemplate, ...] = (
    RoleTemplate(
        code="owner",
        name="Owner",
        description="Full control of the workspace, including billing and closure.",
        permissions=[WILDCARD],
        rank=0,
        is_billing_role=True,
    ),
    RoleTemplate(
        code="admin",
        name="Administrator",
        description="Runs the workspace day to day. Everything except closing the account.",
        permissions=[
            "org.view", "org.manage",
            "people.*", "cameras.*", "zones.*", "live.*", "events.*", "alerts.*",
            "employees.*", "analytics.*", "training.*", "models.*",
            "billing.view", "billing.invoices",
            "api.*", "audit.view", "audit.export",
        ],
        rank=10,
    ),
    RoleTemplate(
        code="security_manager",
        name="Security Manager",
        description="Owns cameras, geofences, incidents and escalation policy.",
        permissions=[
            "org.view", "people.view",
            "cameras.view", "cameras.manage", "cameras.control",
            "zones.view", "zones.manage",
            "live.view", "live.snapshot",
            "events.view", "events.acknowledge", "events.resolve", "events.export",
            "alerts.view", "alerts.manage", "alerts.channels",
            "employees.view", "analytics.view", "analytics.export",
            "models.view", "audit.view",
        ],
        rank=20,
    ),
    RoleTemplate(
        code="hr_manager",
        name="HR Manager",
        description="Employee directory, attendance and behaviour insight — no camera configuration.",
        permissions=[
            "org.view", "people.view",
            "cameras.view", "live.view",
            "events.view", "events.acknowledge",
            "employees.view", "employees.manage", "employees.enroll",
            "analytics.view", "analytics.export", "analytics.schedule",
        ],
        rank=30,
    ),
    RoleTemplate(
        code="operator",
        name="Control-Room Operator",
        description="Watches the live wall and triages incoming alerts.",
        permissions=[
            "cameras.view", "zones.view",
            "live.view", "live.snapshot",
            "events.view", "events.acknowledge",
            "alerts.view", "employees.view", "analytics.view",
        ],
        rank=40,
    ),
    RoleTemplate(
        code="ai_engineer",
        name="AI Engineer",
        description="Builds datasets, trains Campy models and promotes new versions.",
        permissions=[
            "org.view", "cameras.view", "zones.view", "live.view", "live.snapshot",
            "events.view", "events.export",
            "training.*", "models.*",
            "analytics.view", "api.view", "api.manage",
        ],
        rank=40,
    ),
    RoleTemplate(
        code="auditor",
        name="Auditor",
        description="Read-only access plus the full compliance audit trail.",
        permissions=VIEWER_CODES + ["audit.view", "audit.export", "events.export", "billing.view"],
        rank=60,
    ),
    RoleTemplate(
        code="viewer",
        name="Viewer",
        description="Read-only access to dashboards, cameras and incidents.",
        permissions=VIEWER_CODES,
        rank=80,
    ),
)

ROLE_TEMPLATE_MAP: dict[str, RoleTemplate] = {tpl.code: tpl for tpl in ROLE_TEMPLATES}
DEFAULT_MEMBER_ROLE = "viewer"
DEFAULT_OWNER_ROLE = "owner"
