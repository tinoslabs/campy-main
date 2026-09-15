"""Navigation and site-wide CMS content."""
from __future__ import annotations


def cms_navigation(request):
    settings_obj = getattr(request, "site_settings", None)
    menus: dict[str, list] = {}
    try:
        from .models import Menu

        for menu in Menu.objects.prefetch_related("items"):
            menus[menu.location] = list(menu.root_items)
    except Exception:  # noqa: BLE001 - before first migrate
        menus = {}

    return {
        "site_settings": settings_obj,
        "nav_header": menus.get("header", []),
        "nav_footer_product": menus.get("footer_product", []),
        "nav_footer_company": menus.get("footer_company", []),
        "nav_footer_legal": menus.get("footer_legal", []),
        "site_announcement": getattr(settings_obj, "announcement", "") if settings_obj else "",
        "site_announcement_url": getattr(settings_obj, "announcement_url", "") if settings_obj else "",
    }
