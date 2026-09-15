from django.urls import path

from . import views

app_name = "accounts"

urlpatterns = [
    # Authentication
    path("login/", views.login_view, name="login"),
    path("signup/", views.signup_view, name="signup"),
    path("logout/", views.logout_view, name="logout"),
    path("password/reset/", views.password_reset_request, name="password_reset"),
    path("password/reset/<str:token>/", views.password_reset_confirm, name="password_reset_confirm"),
    path("verify/<str:token>/", views.verify_email, name="verify_email"),

    # Account
    path("profile/", views.profile, name="profile"),
    path("security/", views.security, name="security"),

    # Workspaces
    path("workspace/new/", views.organization_create, name="organization_create"),
    path("workspace/edit/", views.organization_edit, name="organization_edit"),
    path("workspace/switch/<slug:slug>/", views.switch_organization, name="switch_organization"),

    # Team
    path("invite/", views.invite_member, name="invite_member"),
    path("invite/<uuid:uid>/revoke/", views.revoke_invitation, name="revoke_invitation"),
    path("join/<str:token>/", views.accept_invitation, name="accept_invitation"),
    path("members/<uuid:uid>/edit/", views.edit_membership, name="edit_membership"),
    path("members/<uuid:uid>/remove/", views.remove_member, name="remove_member"),

    # Roles
    path("roles/", views.role_list, name="roles"),
    path("roles/new/", views.role_edit, name="role_create"),
    path("roles/<uuid:uid>/edit/", views.role_edit, name="role_edit"),
    path("roles/<uuid:uid>/delete/", views.role_delete, name="role_delete"),
]
