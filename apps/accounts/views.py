"""Authentication, workspace and team-management views."""
from __future__ import annotations

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import get_user_model, login, logout
from django.contrib.auth.decorators import login_required
from django.core.mail import send_mail
from django.db import transaction
from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST


from .access import require_organization, require_perm
from .forms import (
    AcceptInvitationForm,
    InvitationForm,
    LoginForm,
    MembershipForm,
    OrganizationCreateForm,
    OrganizationForm,
    PasswordChangeForm,
    PasswordResetForm,
    PasswordResetRequestForm,
    ProfileForm,
    RoleForm,
    SignupForm,
)
from .models import AuditLog, EmailToken, Invitation, Membership, Organization, Role
from .permissions import permissions_by_module, validate

User = get_user_model()


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
def login_view(request):
    if request.user.is_authenticated:
        return redirect("dashboard:home")

    form = LoginForm(request, data=request.POST or None)
    if request.method == "POST" and form.is_valid():
        login(request, form.user, backend="apps.accounts.backends.EmailOrUsernameBackend")
        if not form.cleaned_data.get("remember_me"):
            request.session.set_expiry(0)          # expires when the browser closes
        destination = request.GET.get("next") or request.POST.get("next")
        if destination and destination.startswith("/"):
            return HttpResponseRedirect(destination)
        return redirect("dashboard:home")

    return render(request, "accounts/login.html", {"form": form, "next": request.GET.get("next", "")})


def signup_view(request):
    if request.user.is_authenticated:
        return redirect("dashboard:home")

    site_settings = getattr(request, "site_settings", None)
    if site_settings is not None and not site_settings.signups_enabled:
        messages.info(request, "Self-service sign-up is currently closed. Please request a demo.")
        return redirect("cms:contact")

    form = SignupForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        with transaction.atomic():
            user, organization = form.save()
            _start_default_trial(organization)

        login(request, user, backend="apps.accounts.backends.EmailOrUsernameBackend")
        _send_verification_email(request, user)
        AuditLog.record(
            action=AuditLog.Action.CREATE, actor=user, organization=organization,
            target=organization, summary=f"Created workspace {organization.name}", request=request,
        )
        messages.success(
            request,
            f"Welcome to {settings.CAMPY['BRAND_NAME']}. Your workspace '{organization.name}' is ready.",
        )
        return redirect("dashboard:onboarding")

    return render(request, "accounts/signup.html", {"form": form})


def _start_default_trial(organization) -> None:
    """Put a brand-new workspace on the cheapest public package's trial."""
    from apps.billing.models import Package
    from apps.billing.services import start_trial

    package = (
        Package.objects.filter(is_active=True, audience=Package.Audience.PUBLIC)
        .order_by("sort_order", "price_inr")
        .first()
    )
    if package is not None:
        start_trial(organization, package)


@require_POST
def logout_view(request):
    logout(request)
    messages.info(request, "You have been signed out.")
    return redirect("cms:home")


def password_reset_request(request):
    form = PasswordResetRequestForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = User.objects.filter(email__iexact=form.cleaned_data["email"]).first()
        if user is not None:
            token = EmailToken.objects.create(user=user, purpose=EmailToken.Purpose.RESET_PASSWORD)
            link = request.build_absolute_uri(reverse("accounts:password_reset_confirm", args=[token.token]))
            send_mail(
                subject=f"Reset your {settings.CAMPY['BRAND_NAME']} password",
                message=(
                    f"Hello {user.get_short_name()},\n\n"
                    f"Use this link to choose a new password (valid for 2 hours):\n{link}\n\n"
                    "If you did not request this, you can safely ignore this email."
                ),
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[user.email],
                fail_silently=True,
            )
        # Always report success — revealing which emails exist is an enumeration hole.
        messages.success(
            request, "If an account exists for that address, a reset link is on its way."
        )
        return redirect("accounts:login")
    return render(request, "accounts/password_reset_request.html", {"form": form})


def password_reset_confirm(request, token: str):
    email_token = EmailToken.objects.filter(
        token=token, purpose=EmailToken.Purpose.RESET_PASSWORD
    ).select_related("user").first()

    if email_token is None or not email_token.is_valid:
        messages.error(request, "That password reset link is invalid or has expired.")
        return redirect("accounts:password_reset")

    form = PasswordResetForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = email_token.user
        user.set_password(form.cleaned_data["new_password2"])
        user.failed_login_attempts = 0
        user.locked_until = None
        user.save(update_fields=["password", "failed_login_attempts", "locked_until"])
        email_token.consume()
        AuditLog.record(
            action=AuditLog.Action.SECURITY, actor=user,
            summary="Password reset via email link", request=request, sensitive=True,
        )
        messages.success(request, "Your password has been changed. Please sign in.")
        return redirect("accounts:login")

    return render(request, "accounts/password_reset_confirm.html", {"form": form, "token": token})


def _send_verification_email(request, user) -> None:
    token = EmailToken.objects.create(user=user, purpose=EmailToken.Purpose.VERIFY_EMAIL)
    link = request.build_absolute_uri(reverse("accounts:verify_email", args=[token.token]))
    send_mail(
        subject=f"Confirm your {settings.CAMPY['BRAND_NAME']} email address",
        message=(
            f"Hello {user.get_short_name()},\n\n"
            f"Please confirm your email address:\n{link}\n\n"
            f"— The {settings.CAMPY['BRAND_NAME']} team"
        ),
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=[user.email],
        fail_silently=True,
    )


def verify_email(request, token: str):
    email_token = EmailToken.objects.filter(
        token=token, purpose=EmailToken.Purpose.VERIFY_EMAIL
    ).select_related("user").first()
    if email_token is None or not email_token.is_valid:
        messages.error(request, "That verification link is invalid or has expired.")
        return redirect("dashboard:home" if request.user.is_authenticated else "accounts:login")

    user = email_token.user
    user.email_verified = True
    user.save(update_fields=["email_verified"])
    email_token.consume()
    messages.success(request, "Your email address is confirmed.")
    return redirect("dashboard:home" if request.user.is_authenticated else "accounts:login")


# ---------------------------------------------------------------------------
# Invitations
# ---------------------------------------------------------------------------
def accept_invitation(request, token: str):
    invitation = get_object_or_404(
        Invitation.objects.select_related("organization", "role"), token=token
    )
    if not invitation.is_actionable:
        messages.error(request, "This invitation is no longer valid. Ask for a new one.")
        return redirect("accounts:login")

    # Already signed in with the invited address → accept straight away.
    if request.user.is_authenticated:
        if request.user.email.lower() != invitation.email.lower():
            messages.warning(
                request,
                f"This invitation is for {invitation.email}. Sign out and sign in with that address.",
            )
            return redirect("dashboard:home")
        membership = invitation.accept(request.user)
        AuditLog.record(
            action=AuditLog.Action.PERMISSION, actor=request.user,
            organization=invitation.organization, target=membership,
            summary=f"Joined {invitation.organization.name} as {invitation.role.name}", request=request,
        )
        messages.success(request, f"You have joined {invitation.organization.name}.")
        return redirect("dashboard:home")

    existing = User.objects.filter(email__iexact=invitation.email).first()
    if existing is not None:
        messages.info(request, "Sign in to accept this invitation.")
        return redirect(f"{reverse('accounts:login')}?next={invitation.get_accept_url()}")

    form = AcceptInvitationForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        with transaction.atomic():
            user = User.objects.create_user(
                email=invitation.email,
                password=form.cleaned_data["password2"],
                full_name=form.cleaned_data["full_name"],
                email_verified=True,
            )
            invitation.accept(user)
        login(request, user, backend="apps.accounts.backends.EmailOrUsernameBackend")
        messages.success(request, f"Welcome to {invitation.organization.name}.")
        return redirect("dashboard:home")

    return render(request, "accounts/accept_invitation.html", {"form": form, "invitation": invitation})


@login_required
@require_perm("people.invite")
def invite_member(request):
    organization = request.organization
    form = InvitationForm(organization, request.user, data=request.POST or None)

    if request.method == "POST" and form.is_valid():
        from apps.billing.middleware import can_add

        allowed, reason = can_add(organization, "users")
        if not allowed:
            messages.error(request, reason)
            return redirect("dashboard:team")

        invitation = form.save()
        link = request.build_absolute_uri(invitation.get_accept_url())
        send_mail(
            subject=f"{request.user.get_full_name()} invited you to {organization.name} on {settings.CAMPY['BRAND_NAME']}",
            message=(
                f"{request.user.get_full_name()} has invited you to join the "
                f"'{organization.name}' workspace as {invitation.role.name}.\n\n"
                f"{invitation.message}\n\n"
                f"Accept the invitation:\n{link}\n\n"
                f"This link expires on {invitation.expires_at:%d %b %Y}."
            ),
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[invitation.email],
            fail_silently=True,
        )
        AuditLog.record(
            action=AuditLog.Action.PERMISSION, actor=request.user, organization=organization,
            target=invitation, summary=f"Invited {invitation.email} as {invitation.role.name}",
            request=request,
        )
        messages.success(request, f"Invitation sent to {invitation.email}.")
        return redirect("dashboard:team")

    return render(request, "accounts/invite_member.html", {"form": form})


@login_required
@require_perm("people.invite")
@require_POST
def revoke_invitation(request, uid):
    invitation = get_object_or_404(Invitation, uid=uid, organization=request.organization)
    invitation.status = Invitation.Status.REVOKED
    invitation.save(update_fields=["status", "updated_at"])
    messages.success(request, f"Invitation to {invitation.email} revoked.")
    return redirect("dashboard:team")


# ---------------------------------------------------------------------------
# Members & roles
# ---------------------------------------------------------------------------
@login_required
@require_perm("people.manage")
def edit_membership(request, uid):
    organization = request.organization
    membership = get_object_or_404(
        Membership.objects.select_related("user", "role"), uid=uid, organization=organization
    )

    # Guard: nobody may edit someone more senior than themselves.
    editor_membership = request.user.membership_for(organization)
    if (
        not request.user.is_superadmin
        and editor_membership is not None
        and membership.role.rank < editor_membership.role.rank
    ):
        messages.error(request, "You cannot change the role of someone more senior than you.")
        return redirect("dashboard:team")

    form = MembershipForm(
        organization, request.user, data=request.POST or None, instance=membership
    )
    if request.method == "POST" and form.is_valid():
        if membership.is_owner and form.cleaned_data["role"].code != "owner":
            owners = Membership.objects.filter(
                organization=organization, role__code="owner", status=Membership.Status.ACTIVE
            ).exclude(pk=membership.pk)
            if not owners.exists():
                messages.error(request, "A workspace must always keep at least one owner.")
                return redirect("dashboard:team")

        previous_role = membership.role.name
        form.save()
        AuditLog.record(
            action=AuditLog.Action.PERMISSION, actor=request.user, organization=organization,
            target=membership,
            summary=f"Changed {membership.user.email} from {previous_role} to {membership.role.name}",
            changes={"from": previous_role, "to": membership.role.name},
            request=request, sensitive=True,
        )
        messages.success(request, f"Updated {membership.user.get_full_name()}.")
        return redirect("dashboard:team")

    return render(request, "accounts/edit_membership.html", {"form": form, "membership": membership})


@login_required
@require_perm("people.remove")
@require_POST
def remove_member(request, uid):
    organization = request.organization
    membership = get_object_or_404(Membership, uid=uid, organization=organization)

    if membership.user_id == request.user.pk:
        messages.error(request, "You cannot remove yourself. Ask another owner to do it.")
        return redirect("dashboard:team")
    if membership.is_owner:
        owners = Membership.objects.filter(
            organization=organization, role__code="owner", status=Membership.Status.ACTIVE
        ).exclude(pk=membership.pk)
        if not owners.exists():
            messages.error(request, "A workspace must always keep at least one owner.")
            return redirect("dashboard:team")

    email = membership.user.email
    membership.status = Membership.Status.REMOVED
    membership.save(update_fields=["status", "updated_at"])
    AuditLog.record(
        action=AuditLog.Action.PERMISSION, actor=request.user, organization=organization,
        summary=f"Removed {email} from the workspace", request=request, sensitive=True,
    )
    messages.success(request, f"{email} has been removed from this workspace.")
    return redirect("dashboard:team")


@login_required
@require_perm("people.roles")
def role_list(request):
    from django.db.models import Count, Q

    roles = (
        Role.objects.filter(organization=request.organization)
        .annotate(
            member_count=Count(
                "memberships", filter=Q(memberships__status=Membership.Status.ACTIVE)
            )
        )
        .order_by("rank", "name")
    )
    return render(
        request,
        "accounts/roles.html",
        {"roles": roles, "active": "team", "permission_groups": permissions_by_module()},
    )


@login_required
@require_perm("people.roles")
def role_edit(request, uid=None):
    organization = request.organization
    role = get_object_or_404(Role, uid=uid, organization=organization) if uid else None

    if role is not None and role.is_system and role.code == "owner":
        messages.error(request, "The Owner role always has full access and cannot be edited.")
        return redirect("accounts:roles")

    form = RoleForm(organization, data=request.POST or None, instance=role)
    if request.method == "POST" and form.is_valid():
        selected = validate(request.POST.getlist("permissions"))
        if not selected:
            messages.error(request, "Select at least one permission for this role.")
        else:
            saved = form.save(permissions=selected)
            AuditLog.record(
                action=AuditLog.Action.PERMISSION, actor=request.user, organization=organization,
                target=saved, summary=f"{'Updated' if role else 'Created'} role {saved.name}",
                changes={"permissions": selected}, request=request, sensitive=True,
            )
            messages.success(request, f"Role '{saved.name}' saved.")
            return redirect("accounts:roles")

    current = set(role.expanded_permissions) if role else set()
    return render(
        request,
        "accounts/role_form.html",
        {
            "form": form,
            "role": role,
            "permission_groups": permissions_by_module(),
            "current_permissions": current,
            "active": "team",
        },
    )


@login_required
@require_perm("people.roles")
@require_POST
def role_delete(request, uid):
    role = get_object_or_404(Role, uid=uid, organization=request.organization)
    if role.is_system:
        messages.error(request, "Built-in roles cannot be deleted.")
    elif role.memberships.exists():
        messages.error(
            request, f"'{role.name}' is still assigned to {role.memberships.count()} member(s)."
        )
    else:
        name = role.name
        role.delete()
        messages.success(request, f"Role '{name}' deleted.")
    return redirect("accounts:roles")


# ---------------------------------------------------------------------------
# Profile & security
# ---------------------------------------------------------------------------
@login_required
def profile(request):
    form = ProfileForm(request.POST or None, request.FILES or None, instance=request.user)
    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(request, "Your profile has been updated.")
        return redirect("accounts:profile")
    return render(request, "accounts/profile.html", {"form": form, "active": "profile"})


@login_required
def security(request):
    form = PasswordChangeForm(request.user, data=request.POST or None)
    if request.method == "POST" and form.is_valid():
        form.save()
        login(request, request.user, backend="apps.accounts.backends.EmailOrUsernameBackend")
        AuditLog.record(
            action=AuditLog.Action.SECURITY, actor=request.user,
            summary="Changed account password", request=request, sensitive=True,
        )
        messages.success(request, "Your password has been changed.")
        return redirect("accounts:security")

    sessions = AuditLog.objects.filter(
        actor=request.user, action__in=[AuditLog.Action.LOGIN, AuditLog.Action.LOGIN_FAILED]
    )[:15]
    return render(
        request, "accounts/security.html", {"form": form, "recent_activity": sessions, "active": "profile"}
    )


# ---------------------------------------------------------------------------
# Organisations
# ---------------------------------------------------------------------------
@login_required
def organization_create(request):
    form = OrganizationCreateForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        with transaction.atomic():
            organization = form.save(commit=False)
            organization.contact_email = request.user.email
            organization.status = Organization.Status.TRIAL
            organization.save()
            owner_role = organization.roles.get(code="owner")
            Membership.objects.create(
                user=request.user, organization=organization, role=owner_role, title="Owner"
            )
            request.user.active_organization = organization
            request.user.save(update_fields=["active_organization"])
            _start_default_trial(organization)

        messages.success(request, f"Workspace '{organization.name}' created.")
        return redirect("dashboard:onboarding")

    return render(request, "accounts/organization_create.html", {"form": form})


@login_required
@require_organization
@require_perm("org.manage")
def organization_edit(request):
    organization = request.organization
    form = OrganizationForm(request.POST or None, request.FILES or None, instance=organization)
    if request.method == "POST" and form.is_valid():
        form.save()
        AuditLog.record(
            action=AuditLog.Action.UPDATE, actor=request.user, organization=organization,
            target=organization, summary="Updated organisation profile", request=request,
        )
        messages.success(request, "Organisation details saved.")
        return redirect("dashboard:settings")
    return render(request, "accounts/organization_edit.html", {"form": form, "active": "settings"})


@login_required
@require_POST
def switch_organization(request, slug):
    organization = get_object_or_404(Organization, slug=slug)
    if not (request.user.is_superadmin or request.user.membership_for(organization)):
        messages.error(request, "You do not have access to that workspace.")
        return redirect("dashboard:home")
    request.user.active_organization = organization
    request.user.save(update_fields=["active_organization"])
    return redirect("dashboard:home")
