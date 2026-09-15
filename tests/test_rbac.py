"""Role-based access control, tenancy isolation and feature gating."""
from __future__ import annotations

from django.test import TestCase
from django.urls import reverse

from apps.accounts.models import Membership, Organization, User
from apps.accounts.permissions import ALL_CODES, ROLE_TEMPLATE_MAP, expand
from apps.billing.models import Package
from apps.billing.services import activate_subscription
from apps.cameras.models import Camera, Site


def make_package(name: str, features: dict, quotas: dict | None = None) -> Package:
    return Package.objects.create(
        name=name, features=features,
        quotas=quotas or {"cameras": 10, "sites": 2, "users": 5, "employees": 50},
    )


def make_workspace(name: str, package: Package | None = None) -> Organization:
    organization = Organization.objects.create(name=name, country="IN", status="active")
    if package is not None:
        activate_subscription(organization, package, currency="INR")
    return organization


def add_member(organization: Organization, email: str, role_code: str) -> User:
    user = User.objects.create_user(email=email, password="TestPass!2024", full_name=email.split("@")[0])
    Membership.objects.create(
        user=user, organization=organization, role=organization.roles.get(code=role_code)
    )
    user.active_organization = organization
    user.save(update_fields=["active_organization"])
    return user


class PermissionRegistryTests(TestCase):
    def test_role_templates_only_reference_real_permissions(self):
        """A typo in a role template would silently grant nothing — catch it here."""
        for code, template in ROLE_TEMPLATE_MAP.items():
            resolved = expand(template.permissions)
            self.assertTrue(resolved, f"role '{code}' resolves to no permissions")
            self.assertTrue(
                resolved.issubset(set(ALL_CODES)),
                f"role '{code}' references unknown codes: {resolved - set(ALL_CODES)}",
            )

    def test_owner_has_every_permission(self):
        self.assertEqual(expand(ROLE_TEMPLATE_MAP["owner"].permissions), set(ALL_CODES))

    def test_viewer_has_no_write_permissions(self):
        viewer = expand(ROLE_TEMPLATE_MAP["viewer"].permissions)
        forbidden = {"cameras.manage", "events.resolve", "billing.manage", "people.manage", "org.delete"}
        self.assertEqual(viewer & forbidden, set())

    def test_new_workspace_gets_every_built_in_role(self):
        organization = make_workspace("Bootstrap Co")
        self.assertEqual(
            set(organization.roles.values_list("code", flat=True)), set(ROLE_TEMPLATE_MAP)
        )


class RoleEnforcementTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.full_package = make_package(
            "Full", {key: True for key in [
                "gesture_tracking", "face_recognition", "geofencing", "crowd_management",
                "theft_detection", "object_detection", "fire_detection", "custom_training",
                "reports", "api_access", "audit_log",
            ]}
        )
        cls.org = make_workspace("Enforcement Co", cls.full_package)
        cls.owner = add_member(cls.org, "owner@enforce.test", "owner")
        cls.viewer = add_member(cls.org, "viewer@enforce.test", "viewer")
        cls.operator = add_member(cls.org, "operator@enforce.test", "operator")
        cls.engineer = add_member(cls.org, "engineer@enforce.test", "ai_engineer")
        cls.hr = add_member(cls.org, "hr@enforce.test", "hr_manager")
        # A camera needs somewhere to live; without a site the form correctly
        # redirects to create one first.
        cls.site = Site.objects.create(organization=cls.org, name="Enforcement HQ")

    def test_each_role_gets_exactly_what_it_should(self):
        cases = [
            (self.owner, "billing.manage", True),
            (self.owner, "org.delete", True),
            (self.viewer, "events.view", True),
            (self.viewer, "events.resolve", False),
            (self.viewer, "cameras.manage", False),
            (self.viewer, "billing.manage", False),
            (self.operator, "events.acknowledge", True),
            (self.operator, "events.resolve", False),
            (self.operator, "cameras.manage", False),
            (self.engineer, "training.run", True),
            (self.engineer, "models.deploy", True),
            (self.engineer, "people.manage", False),
            (self.hr, "employees.enroll", True),
            (self.hr, "cameras.manage", False),
        ]
        for user, code, expected in cases:
            with self.subTest(user=user.email, code=code):
                self.assertEqual(user.has_campy_perm(code, self.org), expected)

    def test_viewer_is_refused_by_the_view_layer(self):
        self.client.force_login(self.viewer)
        response = self.client.get(reverse("dashboard:camera_create"))
        self.assertEqual(response.status_code, 403)

    def test_viewer_can_still_read(self):
        self.client.force_login(self.viewer)
        self.assertEqual(self.client.get(reverse("dashboard:events")).status_code, 200)

    def test_owner_may_reach_the_camera_form(self):
        self.client.force_login(self.owner)
        self.assertEqual(self.client.get(reverse("dashboard:camera_create")).status_code, 200)


class FeatureGatingTests(TestCase):
    def test_permission_is_refused_when_the_plan_lacks_its_feature(self):
        """A permission is not enough — the package must include the feature."""
        basic = make_package("Basic", {"geofencing": True, "face_recognition": False})
        organization = make_workspace("Gated Co", basic)
        owner = add_member(organization, "owner@gated.test", "owner")

        self.assertTrue(owner.has_campy_perm("zones.manage", organization))
        self.assertFalse(owner.has_campy_perm("employees.enroll", organization))
        # The permission itself is held — only the feature gate blocks it.
        self.assertIn("employees.enroll", owner.permission_codes(organization))
        self.assertTrue(
            owner.has_campy_perm("employees.enroll", organization, check_feature=False)
        )

    def test_camera_hides_analytics_the_plan_does_not_include(self):
        basic = make_package("Basic2", {"geofencing": True, "fire_detection": False})
        organization = make_workspace("Analytics Co", basic)
        site = Site.objects.create(organization=organization, name="HQ")
        camera = Camera.objects.create(
            organization=organization, site=site, name="Cam 1",
            protocol=Camera.Protocol.DEMO,
            enabled_analytics=["geofence", "fire", "theft"],
        )
        self.assertIn("geofence", camera.active_analytics())
        self.assertNotIn("fire", camera.active_analytics())
        self.assertIn("fire", camera.blocked_analytics)


class TenantIsolationTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        package = make_package("Iso", {"geofencing": True})
        cls.alpha = make_workspace("Alpha Ltd", package)
        cls.beta = make_workspace("Beta Ltd", package)
        cls.alpha_owner = add_member(cls.alpha, "owner@alpha.test", "owner")
        cls.beta_owner = add_member(cls.beta, "owner@beta.test", "owner")

        cls.alpha_site = Site.objects.create(organization=cls.alpha, name="Alpha HQ")
        cls.alpha_camera = Camera.objects.create(
            organization=cls.alpha, site=cls.alpha_site, name="Alpha Lobby",
            protocol=Camera.Protocol.DEMO,
        )

    def test_one_workspace_cannot_open_another_workspaces_camera(self):
        self.client.force_login(self.beta_owner)
        response = self.client.get(
            reverse("dashboard:camera_detail", args=[self.alpha_camera.uid])
        )
        self.assertEqual(response.status_code, 404)

    def test_camera_list_shows_only_your_own(self):
        self.client.force_login(self.beta_owner)
        response = self.client.get(reverse("dashboard:cameras"))
        self.assertNotContains(response, "Alpha Lobby")

    def test_owner_of_the_other_workspace_sees_it_fine(self):
        self.client.force_login(self.alpha_owner)
        response = self.client.get(reverse("dashboard:cameras"))
        self.assertContains(response, "Alpha Lobby")

    def test_permissions_do_not_carry_across_workspaces(self):
        """Being an owner of Alpha grants nothing at all in Beta."""
        self.assertTrue(self.alpha_owner.has_campy_perm("cameras.manage", self.alpha))
        self.assertFalse(self.alpha_owner.has_campy_perm("cameras.manage", self.beta))
        self.assertEqual(self.alpha_owner.permission_codes(self.beta), set())


class PrivilegeEscalationTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        package = make_package("Esc", {"geofencing": True})
        cls.org = make_workspace("Escalation Co", package)
        cls.owner = add_member(cls.org, "owner@esc.test", "owner")
        cls.operator = add_member(cls.org, "operator@esc.test", "operator")

    def test_a_member_cannot_grant_a_role_more_senior_than_their_own(self):
        from apps.accounts.forms import InvitationForm

        form = InvitationForm(self.org, self.operator)
        offered = set(form.fields["role"].queryset.values_list("code", flat=True))
        self.assertNotIn("owner", offered)
        self.assertNotIn("admin", offered)

    def test_an_owner_may_grant_any_role(self):
        from apps.accounts.forms import InvitationForm

        form = InvitationForm(self.org, self.owner)
        offered = set(form.fields["role"].queryset.values_list("code", flat=True))
        self.assertIn("owner", offered)

    def test_the_last_owner_cannot_be_removed(self):
        self.client.force_login(self.owner)
        membership = Membership.objects.get(user=self.owner, organization=self.org)
        # Removing yourself is refused outright.
        response = self.client.post(
            reverse("accounts:remove_member", args=[membership.uid]), follow=True
        )
        membership.refresh_from_db()
        self.assertEqual(membership.status, Membership.Status.ACTIVE)
        self.assertEqual(response.status_code, 200)

    def test_membership_denied_permissions_win_over_the_role(self):
        membership = Membership.objects.get(user=self.owner, organization=self.org)
        membership.denied_permissions = ["billing.manage"]
        membership.save()
        self.owner.refresh_from_db()
        self.assertNotIn("billing.manage", self.owner.permission_codes(self.org))

    def test_membership_extra_permissions_add_to_the_role(self):
        membership = Membership.objects.get(user=self.operator, organization=self.org)
        self.assertNotIn("events.resolve", self.operator.permission_codes(self.org))
        membership.extra_permissions = ["events.resolve"]
        membership.save()
        self.assertIn("events.resolve", self.operator.permission_codes(self.org))


class UsernameCollisionTests(TestCase):
    def test_addresses_sharing_a_local_part_get_distinct_usernames(self):
        """Regression: forcing username from the email local-part collided."""
        first = User.objects.create_user(email="chris@one.example", password="TestPass!2024")
        second = User.objects.create_user(email="chris@two.example", password="TestPass!2024")
        third = User.objects.create_user(email="chris@three.example", password="TestPass!2024")
        usernames = {first.username, second.username, third.username}
        self.assertEqual(len(usernames), 3, usernames)
