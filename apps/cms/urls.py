from django.urls import path

from . import views

app_name = "cms"

urlpatterns = [
    # CMS editor (must come before the catch-all page route)
    path("manage/pages/", views.cms_pages, name="cms_pages"),
    path("manage/pages/new/", views.cms_page_form, name="cms_page_create"),
    path("manage/pages/<int:pk>/", views.cms_page_form, name="cms_page_edit"),
    path("manage/pages/<int:pk>/publish/", views.cms_page_publish, name="cms_page_publish"),
    path("manage/pages/<int:page_pk>/blocks/new/", views.cms_block_form, name="cms_block_create"),
    path("manage/pages/<int:page_pk>/blocks/<int:pk>/", views.cms_block_form, name="cms_block_edit"),
    path("manage/pages/<int:page_pk>/blocks/<int:pk>/delete/", views.cms_block_delete, name="cms_block_delete"),
    path("manage/posts/", views.cms_posts, name="cms_posts"),
    path("manage/posts/new/", views.cms_post_form, name="cms_post_create"),
    path("manage/posts/<int:pk>/", views.cms_post_form, name="cms_post_edit"),
    path("manage/leads/", views.cms_leads, name="cms_leads"),
    path("manage/leads/<int:pk>/status/", views.cms_lead_status, name="cms_lead_status"),
    path("manage/settings/", views.cms_settings, name="cms_settings"),
    path("manage/media/", views.cms_media, name="cms_media"),
    path("manage/faqs/", views.cms_faqs, name="cms_faqs"),
    path("manage/testimonials/", views.cms_testimonials, name="cms_testimonials"),
    path("manage/menus/", views.cms_menus, name="cms_menus"),
    path("manage/menus/<int:pk>/delete/", views.cms_menu_item_delete, name="cms_menu_item_delete"),

    # Public site
    path("", views.home, name="home"),
    path("features/", views.features, name="features"),
    path("pricing/", views.pricing, name="pricing"),
    path("contact/", views.contact, name="contact"),
    path("contact/thanks/", views.contact_thanks, name="contact_thanks"),
    path("demo/", views.demo, name="demo"),
    path("faq/", views.faq_list, name="faq"),
    path("blog/", views.blog, name="blog"),
    path("blog/category/<slug:slug>/", views.blog_category, name="blog_category"),
    path("blog/tag/<slug:slug>/", views.blog_tag, name="blog_tag"),
    path("blog/<slug:slug>/", views.blog_post, name="blog_post"),

    # Catch-all CMS page — keep last.
    path("<slug:slug>/", views.page_detail, name="page"),
]
