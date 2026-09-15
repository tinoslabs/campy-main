"""Authentication, workspace and team-management forms."""
from __future__ import annotations

from django import forms
from django.contrib.auth import authenticate, get_user_model, password_validation
from django.utils.text import slugify

from .models import Invitation, Membership, Organization, Role
from .permissions import ROLE_TEMPLATES, permissions_by_module

User = get_user_model()


def _widget(placeholder: str = "", **attrs):
    base = {"class": "input"}
    if placeholder:
        base["placeholder"] = placeholder
    base.update(attrs)
    return base


class LoginForm(forms.Form):
    username = forms.CharField(
        label="Email or username",
        widget=forms.TextInput(attrs=_widget("you@company.com", autofocus=True, autocomplete="username")),
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs=_widget("••••••••", autocomplete="current-password"))
    )
    remember_me = forms.BooleanField(required=False, initial=True, label="Keep me signed in")

    def __init__(self, request=None, *args, **kwargs):
        self.request = request
        self.user = None
        super().__init__(*args, **kwargs)

    def clean(self):
        cleaned = super().clean()
        identifier = (cleaned.get("username") or "").strip()
        password = cleaned.get("password")
        if identifier and password:
            candidate = User.objects.filter(email__iexact=identifier).first() or \
                User.objects.filter(username__iexact=identifier).first()
            if candidate and candidate.is_locked:
                raise forms.ValidationError(
                    "This account is temporarily locked after repeated failed sign-in attempts. "
                    "Try again shortly or reset your password."
                )
            self.user = authenticate(self.request, username=identifier, password=password)
            if self.user is None:
                raise forms.ValidationError("That email/username and password do not match an account.")
            if not self.user.is_active:
                raise forms.ValidationError("This account has been deactivated. Contact your administrator.")
        return cleaned


class SignupForm(forms.Form):
    """Creates a user *and* their first workspace in one step."""

    full_name = forms.CharField(
        max_length=180, widget=forms.TextInput(attrs=_widget("Asha Menon", autofocus=True, autocomplete="name"))
    )
    email = forms.EmailField(
        widget=forms.EmailInput(attrs=_widget("you@company.com", autocomplete="email"))
    )
    phone = forms.CharField(
        required=False, max_length=32, widget=forms.TextInput(attrs=_widget("+91 98765 43210"))
    )
    organization_name = forms.CharField(
        label="Company / organisation",
        max_length=180,
        widget=forms.TextInput(attrs=_widget("Acme Manufacturing Pvt Ltd")),
    )
    industry = forms.ChoiceField(
        choices=Organization.Industry.choices, initial=Organization.Industry.OFFICE,
        widget=forms.Select(attrs={"class": "input"}),
    )
    country = forms.CharField(
        max_length=2, initial="IN", widget=forms.TextInput(attrs=_widget("IN", maxlength="2"))
    )
    password1 = forms.CharField(
        label="Password",
        widget=forms.PasswordInput(attrs=_widget("At least 10 characters", autocomplete="new-password")),
    )
    password2 = forms.CharField(
        label="Confirm password",
        widget=forms.PasswordInput(attrs=_widget("Repeat your password", autocomplete="new-password")),
    )
    accept_terms = forms.BooleanField(
        label="I agree to the Terms of Service and Privacy Policy",
        error_messages={"required": "You must accept the terms to create an account."},
    )

    def clean_email(self):
        email = self.cleaned_data["email"].lower().strip()
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError("An account already exists with this email address.")
        return email

    def clean_country(self):
        return (self.cleaned_data.get("country") or "IN").upper()[:2]

    def clean_password2(self):
        password1 = self.cleaned_data.get("password1")
        password2 = self.cleaned_data.get("password2")
        if password1 and password2 and password1 != password2:
            raise forms.ValidationError("The two passwords do not match.")
        password_validation.validate_password(password2)
        return password2

    def save(self) -> tuple[User, Organization]:
        data = self.cleaned_data
        user = User.objects.create_user(
            email=data["email"],
            password=data["password1"],
            full_name=data["full_name"],
            phone=data.get("phone", ""),
        )
        organization = Organization.objects.create(
            name=data["organization_name"],
            industry=data["industry"],
            country=data["country"],
            contact_email=data["email"],
            contact_phone=data.get("phone", ""),
            status=Organization.Status.TRIAL,
        )
        # bootstrap_default_roles runs on post_save; the owner role exists here.
        owner_role = organization.roles.get(code="owner")
        Membership.objects.create(
            user=user, organization=organization, role=owner_role, title="Owner"
        )
        user.active_organization = organization
        user.save(update_fields=["active_organization"])
        return user, organization


class OrganizationForm(forms.ModelForm):
    class Meta:
        model = Organization
        fields = [
            "name", "legal_name", "industry", "website", "contact_email", "contact_phone",
            "address_line1", "address_line2", "city", "state", "postal_code", "country",
            "tax_id", "timezone", "logo",
        ]
        widgets = {
            "name": forms.TextInput(attrs=_widget()),
            "legal_name": forms.TextInput(attrs=_widget("Registered legal entity name")),
            "industry": forms.Select(attrs={"class": "input"}),
            "website": forms.URLInput(attrs=_widget("https://example.com")),
            "contact_email": forms.EmailInput(attrs=_widget()),
            "contact_phone": forms.TextInput(attrs=_widget()),
            "address_line1": forms.TextInput(attrs=_widget()),
            "address_line2": forms.TextInput(attrs=_widget()),
            "city": forms.TextInput(attrs=_widget()),
            "state": forms.TextInput(attrs=_widget()),
            "postal_code": forms.TextInput(attrs=_widget()),
            "country": forms.TextInput(attrs=_widget("IN", maxlength="2")),
            "tax_id": forms.TextInput(attrs=_widget("29ABCDE1234F1Z5")),
            "timezone": forms.TextInput(attrs=_widget("Asia/Kolkata")),
        }

    def clean_country(self):
        return (self.cleaned_data.get("country") or "IN").upper()[:2]


class OrganizationCreateForm(forms.ModelForm):
    class Meta:
        model = Organization
        fields = ["name", "industry", "country", "timezone"]
        widgets = {
            "name": forms.TextInput(attrs=_widget("Acme Manufacturing", autofocus=True)),
            "industry": forms.Select(attrs={"class": "input"}),
            "country": forms.TextInput(attrs=_widget("IN", maxlength="2")),
            "timezone": forms.TextInput(attrs=_widget("Asia/Kolkata")),
        }

    def clean_name(self):
        name = self.cleaned_data["name"].strip()
        if Organization.objects.filter(slug=slugify(name)).exists():
            raise forms.ValidationError("A workspace with a very similar name already exists.")
        return name


class ProfileForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ["full_name", "phone", "job_title", "avatar", "timezone", "locale", "theme"]
        widgets = {
            "full_name": forms.TextInput(attrs=_widget()),
            "phone": forms.TextInput(attrs=_widget()),
            "job_title": forms.TextInput(attrs=_widget("Security Manager")),
            "timezone": forms.TextInput(attrs=_widget()),
            "locale": forms.TextInput(attrs=_widget()),
            "theme": forms.Select(attrs={"class": "input"}),
        }


class PasswordChangeForm(forms.Form):
    current_password = forms.CharField(widget=forms.PasswordInput(attrs=_widget(autocomplete="current-password")))
    new_password1 = forms.CharField(
        label="New password", widget=forms.PasswordInput(attrs=_widget(autocomplete="new-password"))
    )
    new_password2 = forms.CharField(
        label="Confirm new password", widget=forms.PasswordInput(attrs=_widget(autocomplete="new-password"))
    )

    def __init__(self, user, *args, **kwargs):
        self.user = user
        super().__init__(*args, **kwargs)

    def clean_current_password(self):
        password = self.cleaned_data["current_password"]
        if not self.user.check_password(password):
            raise forms.ValidationError("Your current password is not correct.")
        return password

    def clean_new_password2(self):
        first = self.cleaned_data.get("new_password1")
        second = self.cleaned_data.get("new_password2")
        if first and second and first != second:
            raise forms.ValidationError("The two passwords do not match.")
        password_validation.validate_password(second, self.user)
        return second

    def save(self) -> User:
        self.user.set_password(self.cleaned_data["new_password2"])
        self.user.save(update_fields=["password"])
        return self.user


class PasswordResetRequestForm(forms.Form):
    email = forms.EmailField(widget=forms.EmailInput(attrs=_widget("you@company.com", autofocus=True)))


class PasswordResetForm(forms.Form):
    new_password1 = forms.CharField(
        label="New password", widget=forms.PasswordInput(attrs=_widget(autofocus=True, autocomplete="new-password"))
    )
    new_password2 = forms.CharField(
        label="Confirm new password", widget=forms.PasswordInput(attrs=_widget(autocomplete="new-password"))
    )

    def clean_new_password2(self):
        first = self.cleaned_data.get("new_password1")
        second = self.cleaned_data.get("new_password2")
        if first and second and first != second:
            raise forms.ValidationError("The two passwords do not match.")
        password_validation.validate_password(second)
        return second


class InvitationForm(forms.ModelForm):
    class Meta:
        model = Invitation
        fields = ["email", "role", "message"]
        widgets = {
            "email": forms.EmailInput(attrs=_widget("colleague@company.com", autofocus=True)),
            "role": forms.Select(attrs={"class": "input"}),
            "message": forms.Textarea(attrs=_widget("Optional note for the invitation email", rows=3)),
        }

    def __init__(self, organization, inviter=None, *args, **kwargs):
        self.organization = organization
        self.inviter = inviter
        super().__init__(*args, **kwargs)
        roles = Role.objects.filter(organization=organization)
        # Nobody may hand out a role more senior than their own.
        if inviter is not None and not inviter.is_superadmin:
            membership = inviter.membership_for(organization)
            if membership is not None:
                roles = roles.filter(rank__gte=membership.role.rank)
        self.fields["role"].queryset = roles.order_by("rank")

    def clean_email(self):
        email = self.cleaned_data["email"].lower().strip()
        if Membership.objects.filter(
            organization=self.organization, user__email__iexact=email, status=Membership.Status.ACTIVE
        ).exists():
            raise forms.ValidationError("That person is already a member of this workspace.")
        if Invitation.objects.filter(
            organization=self.organization, email__iexact=email, status=Invitation.Status.PENDING
        ).exists():
            raise forms.ValidationError("An invitation is already pending for that address.")
        return email

    def save(self, commit=True):
        invitation = super().save(commit=False)
        invitation.organization = self.organization
        invitation.invited_by = self.inviter
        if commit:
            invitation.save()
        return invitation


class MembershipForm(forms.ModelForm):
    class Meta:
        model = Membership
        fields = ["role", "title", "status", "site_scope"]
        widgets = {
            "role": forms.Select(attrs={"class": "input"}),
            "title": forms.TextInput(attrs=_widget("Shift Supervisor")),
            "status": forms.Select(attrs={"class": "input"}),
            "site_scope": forms.SelectMultiple(attrs={"class": "input", "size": 6}),
        }

    def __init__(self, organization, editor=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["role"].queryset = Role.objects.filter(organization=organization).order_by("rank")
        self.fields["site_scope"].queryset = organization.sites.all()
        self.fields["site_scope"].help_text = "Leave empty to grant access to every site."
        self.fields["site_scope"].required = False
        if editor is not None and not editor.is_superadmin:
            membership = editor.membership_for(organization)
            if membership is not None:
                self.fields["role"].queryset = self.fields["role"].queryset.filter(
                    rank__gte=membership.role.rank
                )


class RoleForm(forms.ModelForm):
    """Custom role builder — permissions come from checkboxes, not a text field."""

    class Meta:
        model = Role
        fields = ["name", "description", "rank", "color"]
        widgets = {
            "name": forms.TextInput(attrs=_widget("Night Shift Supervisor", autofocus=True)),
            "description": forms.Textarea(attrs=_widget("What this role is for", rows=2)),
            "rank": forms.NumberInput(attrs=_widget(min="0", max="999")),
            "color": forms.TextInput(attrs={"class": "input", "type": "color"}),
        }

    def __init__(self, organization, *args, **kwargs):
        self.organization = organization
        super().__init__(*args, **kwargs)
        self.fields["rank"].help_text = "Lower numbers are more senior. Owners are 0."
        self.permission_groups = permissions_by_module()

    def clean_name(self):
        name = self.cleaned_data["name"].strip()
        code = slugify(name)[:60]
        clash = Role.objects.filter(organization=self.organization, code=code)
        if self.instance.pk:
            clash = clash.exclude(pk=self.instance.pk)
        if clash.exists():
            raise forms.ValidationError("A role with this name already exists in the workspace.")
        self._code = code
        return name

    def save(self, commit=True, permissions=None):
        role = super().save(commit=False)
        role.organization = self.organization
        if not role.code:
            role.code = getattr(self, "_code", slugify(role.name)[:60])
        if permissions is not None:
            role.permissions = list(permissions)
        if commit:
            role.save()
        return role


class AcceptInvitationForm(forms.Form):
    """Sign-up form shown to an invited person who has no account yet."""

    full_name = forms.CharField(max_length=180, widget=forms.TextInput(attrs=_widget(autofocus=True)))
    password1 = forms.CharField(label="Password", widget=forms.PasswordInput(attrs=_widget()))
    password2 = forms.CharField(label="Confirm password", widget=forms.PasswordInput(attrs=_widget()))

    def clean_password2(self):
        first = self.cleaned_data.get("password1")
        second = self.cleaned_data.get("password2")
        if first and second and first != second:
            raise forms.ValidationError("The two passwords do not match.")
        password_validation.validate_password(second)
        return second


ROLE_TEMPLATE_CHOICES = [(t.code, f"{t.name} — {t.description}") for t in ROLE_TEMPLATES]
