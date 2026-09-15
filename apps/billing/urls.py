from django.urls import path

from . import views

app_name = "billing"

urlpatterns = [
    path("", views.overview, name="overview"),
    path("packages/", views.packages, name="packages"),
    path("packages/<slug:slug>/", views.package_detail, name="package_detail"),
    path("packages/<slug:slug>/checkout/", views.checkout, name="checkout"),
    path("packages/<slug:slug>/trial/", views.start_trial_view, name="start_trial"),
    path("payments/<uuid:uid>/razorpay/confirm/", views.razorpay_confirm, name="razorpay_confirm"),
    path("invoices/", views.invoices, name="invoices"),
    path("invoices/<uuid:uid>/", views.invoice_detail, name="invoice_detail"),
    path("invoices/<uuid:uid>/success/", views.checkout_success, name="checkout_success"),
    path("invoices/<uuid:uid>/cancelled/", views.checkout_cancelled, name="checkout_cancelled"),
    path("cancel/", views.cancel, name="cancel"),
]
