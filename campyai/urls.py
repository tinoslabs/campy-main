"""Root URL configuration for Campy AI."""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

from apps.billing import views as billing_views
from apps.core import views as core_views

handler400 = "apps.core.views.bad_request"
handler403 = "apps.core.views.permission_denied"
handler404 = "apps.core.views.page_not_found"
handler500 = "apps.core.views.server_error"

urlpatterns = [
    # Operational
    path("healthz", core_views.healthz, name="healthz"),
    path("robots.txt", core_views.robots_txt, name="robots"),
    path("sitemap.xml", core_views.sitemap_xml, name="sitemap"),

    # Platform
    path("django-admin/", admin.site.urls),
    path("accounts/", include("apps.accounts.urls", namespace="accounts")),
    path("app/", include("apps.dashboard.urls", namespace="dashboard")),
    path("billing/", include("apps.billing.urls", namespace="billing")),
    path("api/v1/", include("apps.api.urls", namespace="api")),

    # Gateway callbacks — kept outside the app namespaces so the URLs stay
    # stable even if the billing app is remounted.
    path("webhooks/razorpay/", billing_views.razorpay_webhook, name="razorpay_webhook"),
    path("webhooks/stripe/", billing_views.stripe_webhook, name="stripe_webhook"),

    # Public website (CMS) — must stay last so its catch-all slug does not
    # shadow the application routes above.
    path("", include("apps.cms.urls", namespace="cms")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.BASE_DIR / "static")

admin.site.site_header = "Campy AI Platform Administration"
admin.site.site_title = "Campy AI"
admin.site.index_title = "Platform control"
