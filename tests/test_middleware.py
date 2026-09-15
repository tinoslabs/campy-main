"""The request-scoped tenancy middleware, exercised past its throttle.

`_touch` only writes once every five minutes, and signing in stamps
`last_seen_at`. Every other test in this suite logs in and immediately makes
requests, so it sits inside that window and the write never runs — which is how
a crash on that line survived 151 passing tests and a sweep of every route.
These tests push `last_seen_at` back into the past so the branch actually runs.
"""
from __future__ import annotations

from datetime import timedelta

from django.test import TestCase
from django.utils import timezone
from django.utils.functional import SimpleLazyObject

from apps.accounts.models import Membership, Organization, User
from tests.test_platform import full_workspace


class LastSeenTests(TestCase):
    def setUp(self):
        self.organization, self.owner = full_workspace("Touch Co")

    def age_out(self, user, minutes=10):
        """Make the throttle window expired, as it is for any real session."""
        stale = timezone.now() - timedelta(minutes=minutes)
        User.objects.filter(pk=user.pk).update(last_seen_at=stale)
        return stale

    def test_request_user_really_is_lazy(self):
        """Guards the premise: if it stopped being lazy these tests prove nothing."""
        captured = {}

        def probe(request):
            captured["type"] = type(request.user)
            captured["class"] = request.user.__class__
            return None

        from apps.accounts import middleware

        original = middleware.CurrentOrganizationMiddleware.__call__

        def wrapped(self, request):
            probe(request)
            return original(self, request)

        middleware.CurrentOrganizationMiddleware.__call__ = wrapped
        try:
            self.client.force_login(self.owner)
            self.client.get("/app/")
        finally:
            middleware.CurrentOrganizationMiddleware.__call__ = original

        self.assertIs(captured["type"], SimpleLazyObject)
        self.assertIs(captured["class"], User)

    def test_a_stale_session_updates_last_seen_instead_of_crashing(self):
        self.client.force_login(self.owner)
        stale = self.age_out(self.owner)

        response = self.client.get("/app/")

        self.assertEqual(response.status_code, 200)
        self.owner.refresh_from_db()
        self.assertGreater(self.owner.last_seen_at, stale)

    def test_a_user_who_has_never_been_seen_is_stamped(self):
        self.client.force_login(self.owner)
        User.objects.filter(pk=self.owner.pk).update(last_seen_at=None)

        self.assertEqual(self.client.get("/app/").status_code, 200)

        self.owner.refresh_from_db()
        self.assertIsNotNone(self.owner.last_seen_at)

    def test_the_membership_is_touched_alongside_the_user(self):
        membership = Membership.objects.get(user=self.owner, organization=self.organization)
        Membership.objects.filter(pk=membership.pk).update(last_active_at=None)
        self.client.force_login(self.owner)
        self.age_out(self.owner)

        self.assertEqual(self.client.get("/app/").status_code, 200)

        membership.refresh_from_db()
        self.assertIsNotNone(membership.last_active_at)

    def test_a_fresh_session_does_not_write_on_every_request(self):
        """The throttle is the point: one write per five minutes, not per page."""
        self.client.force_login(self.owner)
        self.age_out(self.owner)
        self.client.get("/app/")
        self.owner.refresh_from_db()
        first = self.owner.last_seen_at

        for _ in range(3):
            self.client.get("/app/")

        self.owner.refresh_from_db()
        self.assertEqual(self.owner.last_seen_at, first)

    def test_the_public_site_works_for_a_stale_signed_in_user(self):
        """The reported crash was on `/`, which authenticated users also visit."""
        self.client.force_login(self.owner)
        self.age_out(self.owner)

        for path in ("/", "/features/", "/pricing/", "/blog/"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

    def test_a_stale_user_with_no_workspace_still_gets_through(self):
        """No membership means `_touch` is called with membership=None."""
        loner = User.objects.create_user(email="loner@touch.test", password="TestPass!2024",
                                         full_name="Loner")
        self.client.force_login(loner)
        User.objects.filter(pk=loner.pk).update(last_seen_at=timezone.now() - timedelta(hours=2))

        self.assertEqual(self.client.get("/").status_code, 200)

        loner.refresh_from_db()
        self.assertGreater(loner.last_seen_at, timezone.now() - timedelta(minutes=1))

    def test_an_anonymous_visitor_is_untouched(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertFalse(User.objects.filter(last_seen_at__isnull=False)
                         .exclude(pk=self.owner.pk).exists())


class WorkspaceResolutionTests(TestCase):
    """The rest of the middleware's job, also only reachable through a request."""

    def setUp(self):
        self.organization, self.owner = full_workspace("Resolve Co")

    def test_the_org_query_parameter_switches_and_remembers_the_workspace(self):
        second = Organization.objects.create(name="Second Site", country="IN", status="active")
        Membership.objects.create(user=self.owner, organization=second,
                                  role=second.roles.get(code="owner"))
        self.client.force_login(self.owner)

        response = self.client.get(f"/app/?org={second.slug}")

        # The switch happens; the subscription guard then sends an unsubscribed
        # workspace to billing, which is the behaviour we want to hold.
        self.assertRedirects(response, "/billing/packages/", fetch_redirect_response=False)
        self.owner.refresh_from_db()
        self.assertEqual(self.owner.active_organization_id, second.pk)

    def test_an_unknown_workspace_slug_falls_back_rather_than_erroring(self):
        self.client.force_login(self.owner)
        response = self.client.get("/app/?org=does-not-exist")
        self.assertEqual(response.status_code, 200)

    def test_another_tenants_slug_does_not_grant_access_to_it(self):
        stranger = Organization.objects.create(name="Not Yours", country="IN", status="active")
        self.client.force_login(self.owner)

        response = self.client.get(f"/app/?org={stranger.slug}")

        # Guessing a slug resolves to no workspace at all, rather than to
        # somebody else's — the user is sent to create one of their own.
        self.assertEqual(response.status_code, 302)
        self.assertNotIn(stranger.slug, response["Location"])
        self.owner.refresh_from_db()
        self.assertNotEqual(self.owner.active_organization_id, stranger.pk)
