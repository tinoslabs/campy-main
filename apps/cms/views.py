"""Public marketing site and the CMS editor."""
from __future__ import annotations

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.mail import send_mail
from django.core.paginator import Paginator
from django.db.models import Count, Q
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from apps.accounts.access import platform_team_required
from apps.core.utils import client_ip

from .forms import (
    ContactForm,
    ContentBlockForm,
    FAQForm,
    MediaAssetForm,
    MenuItemForm,
    PageForm,
    PostForm,
    SiteSettingsForm,
    TestimonialForm,
)
from .models import FAQ, Category, ContentBlock, Lead, MediaAsset, Menu, MenuItem, Page, Post, SiteSettings, Tag, Testimonial


# ---------------------------------------------------------------------------
# Public site
# ---------------------------------------------------------------------------
def home(request):
    """The marketing home page.

    A CMS page flagged ``is_homepage`` takes over if one exists; otherwise the
    built-in landing page is rendered, so the site is never blank on a fresh
    install.
    """
    page = Page.objects.published().filter(is_homepage=True).prefetch_related("blocks").first()
    if page is not None:
        return render(request, "cms/page.html", _page_context(request, page))

    from apps.billing.models import FEATURES, Package

    return render(
        request,
        "public/home.html",
        {
            "packages": Package.objects.filter(is_active=True, audience=Package.Audience.PUBLIC)
            .order_by("sort_order", "price_inr")[:4],
            "testimonials": Testimonial.objects.filter(is_published=True)[:3],
            "faqs": FAQ.objects.filter(is_published=True)[:6],
            "posts": Post.objects.published()[:3],
            "feature_labels": dict(FEATURES),
        },
    )


def _page_context(request, page) -> dict:
    Page.objects.filter(pk=page.pk).update(view_count=page.view_count + 1)
    return {"page": page, "blocks": page.visible_blocks}


def page_detail(request, slug):
    page = get_object_or_404(
        Page.objects.published().prefetch_related("blocks"), slug=slug
    )
    return render(request, "cms/page.html", _page_context(request, page))


def features(request):
    from apps.aiengine.detectors import analytic_metadata

    return render(
        request,
        "public/features.html",
        {
            "analytics": analytic_metadata(),
            "testimonials": Testimonial.objects.filter(is_published=True)[:2],
        },
    )


def pricing(request):
    from apps.billing.models import FEATURES, QUOTAS, Package

    packages = Package.objects.filter(
        is_active=True, audience=Package.Audience.PUBLIC
    ).order_by("sort_order", "price_inr")
    currency = request.GET.get("currency", "INR").upper()
    if currency not in {"INR", "USD"}:
        currency = "INR"

    return render(
        request,
        "public/pricing.html",
        {
            "packages": packages,
            "currency": currency,
            "symbol": "₹" if currency == "INR" else "$",
            "features": FEATURES,
            "quotas": QUOTAS,
            "faqs": FAQ.objects.filter(is_published=True, category__iexact="pricing")[:8],
        },
    )


def contact(request):
    form = ContactForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        lead = form.save(commit=False)
        lead.source = request.POST.get("source", Lead.Source.CONTACT)
        lead.ip_address = client_ip(request) or None
        lead.referrer = request.META.get("HTTP_REFERER", "")[:300]
        lead.utm = {
            key: request.GET.get(key, "")
            for key in ("utm_source", "utm_medium", "utm_campaign")
            if request.GET.get(key)
        }
        lead.save()

        site_settings = getattr(request, "site_settings", None)
        recipient = getattr(site_settings, "contact_email", None) or settings.CAMPY["SUPPORT_EMAIL"]
        send_mail(
            subject=f"New {lead.get_source_display()} from {lead.name} ({lead.company or 'no company'})",
            message=(
                f"Name: {lead.name}\nEmail: {lead.email}\nPhone: {lead.phone}\n"
                f"Company: {lead.company}\nRole: {lead.job_title}\n"
                f"Country: {lead.country}\nIndustry: {lead.industry}\n"
                f"Cameras: {lead.camera_count}\n\n{lead.message}"
            ),
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[recipient],
            fail_silently=True,
        )
        messages.success(
            request, "Thank you — we have your enquiry and will be in touch within one working day."
        )
        return redirect("cms:contact_thanks")

    return render(request, "public/contact.html", {"form": form})


def contact_thanks(request):
    return render(request, "public/contact_thanks.html", {})


def demo(request):
    form = ContactForm(request.POST or None, initial={"message": "I would like a live demonstration."})
    if request.method == "POST" and form.is_valid():
        lead = form.save(commit=False)
        lead.source = Lead.Source.DEMO
        lead.ip_address = client_ip(request) or None
        lead.save()
        messages.success(request, "Demo requested. We will confirm a time by email shortly.")
        return redirect("cms:contact_thanks")
    return render(request, "public/demo.html", {"form": form})


# ---------------------------------------------------------------------------
# Blog
# ---------------------------------------------------------------------------
def blog(request):
    posts = Post.objects.published().select_related("category", "author").prefetch_related("tags")
    if request.GET.get("q"):
        term = request.GET["q"]
        posts = posts.filter(
            Q(title__icontains=term) | Q(excerpt__icontains=term) | Q(body__icontains=term)
        )

    paginator = Paginator(posts, 9)
    page = paginator.get_page(request.GET.get("page"))
    return render(
        request,
        "public/blog.html",
        {
            "page_obj": page, "posts": page.object_list,
            "featured": Post.objects.published().filter(is_featured=True).first(),
            "categories": Category.objects.annotate(
                post_count=Count("posts", filter=Q(posts__status="published"))
            ).filter(post_count__gt=0),
        },
    )


def blog_post(request, slug):
    post = get_object_or_404(
        Post.objects.published().select_related("category", "author").prefetch_related("tags"), slug=slug
    )
    Post.objects.filter(pk=post.pk).update(view_count=post.view_count + 1)
    related = (
        Post.objects.published()
        .filter(category=post.category).exclude(pk=post.pk)[:3]
        if post.category_id else Post.objects.published().exclude(pk=post.pk)[:3]
    )
    return render(request, "public/blog_post.html", {"post": post, "related": related})


def blog_category(request, slug):
    category = get_object_or_404(Category, slug=slug)
    posts = Post.objects.published().filter(category=category)
    paginator = Paginator(posts, 9)
    page = paginator.get_page(request.GET.get("page"))
    return render(
        request, "public/blog.html",
        {"page_obj": page, "posts": page.object_list, "category": category,
         "categories": Category.objects.all()},
    )


def blog_tag(request, slug):
    tag = get_object_or_404(Tag, slug=slug)
    posts = Post.objects.published().filter(tags=tag)
    paginator = Paginator(posts, 9)
    page = paginator.get_page(request.GET.get("page"))
    return render(
        request, "public/blog.html",
        {"page_obj": page, "posts": page.object_list, "tag": tag,
         "categories": Category.objects.all()},
    )


def faq_list(request):
    faqs = FAQ.objects.filter(is_published=True)
    grouped: dict[str, list] = {}
    for item in faqs:
        grouped.setdefault(item.category or "General", []).append(item)
    return render(request, "public/faq.html", {"grouped": grouped})


# ---------------------------------------------------------------------------
# CMS editor (platform team)
# ---------------------------------------------------------------------------
@login_required
@platform_team_required
def cms_pages(request):
    return render(
        request,
        "cms/admin/pages.html",
        {
            "active": "admin_cms",
            "pages": Page.objects.annotate(block_count=Count("blocks")).order_by("sort_order", "title"),
        },
    )


@login_required
@platform_team_required
def cms_page_form(request, pk=None):
    page = get_object_or_404(Page, pk=pk) if pk else None
    form = PageForm(request.POST or None, request.FILES or None, instance=page)

    if request.method == "POST" and form.is_valid():
        saved = form.save(commit=False)
        if page is None:
            saved.created_by = request.user
        saved.updated_by = request.user
        if saved.status == Page.Status.PUBLISHED and not saved.published_at:
            saved.published_at = timezone.now()
            saved.published_by = request.user
        saved.save()
        messages.success(request, f"Page '{saved.title}' saved.")
        return redirect("cms:cms_page_edit", pk=saved.pk)

    return render(
        request,
        "cms/admin/page_form.html",
        {
            "active": "admin_cms", "form": form, "page_obj": page,
            "blocks": page.blocks.order_by("sort_order") if page else [],
        },
    )


@login_required
@platform_team_required
def cms_block_form(request, page_pk, pk=None):
    page = get_object_or_404(Page, pk=page_pk)
    block = get_object_or_404(ContentBlock, pk=pk, page=page) if pk else None
    form = ContentBlockForm(request.POST or None, request.FILES or None, instance=block)

    if request.method == "POST" and form.is_valid():
        saved = form.save(commit=False)
        saved.page = page
        saved.save()
        messages.success(request, "Content block saved.")
        return redirect("cms:cms_page_edit", pk=page.pk)

    return render(
        request, "cms/admin/block_form.html",
        {"active": "admin_cms", "form": form, "page_obj": page, "block": block},
    )


@login_required
@platform_team_required
@require_POST
def cms_block_delete(request, page_pk, pk):
    block = get_object_or_404(ContentBlock, pk=pk, page__pk=page_pk)
    block.delete()
    messages.success(request, "Block removed.")
    return redirect("cms:cms_page_edit", pk=page_pk)


@login_required
@platform_team_required
@require_POST
def cms_page_publish(request, pk):
    page = get_object_or_404(Page, pk=pk)
    if page.is_published:
        page.unpublish()
        messages.success(request, f"'{page.title}' is now a draft.")
    else:
        page.publish(request.user)
        messages.success(request, f"'{page.title}' is live.")
    return redirect("cms:cms_pages")


@login_required
@platform_team_required
def cms_posts(request):
    return render(
        request, "cms/admin/posts.html",
        {"active": "admin_cms", "posts": Post.objects.select_related("category").order_by("-created_at")},
    )


@login_required
@platform_team_required
def cms_post_form(request, pk=None):
    post = get_object_or_404(Post, pk=pk) if pk else None
    form = PostForm(request.POST or None, request.FILES or None, instance=post)

    if request.method == "POST" and form.is_valid():
        saved = form.save(commit=False)
        if post is None:
            saved.author = request.user
            saved.created_by = request.user
        saved.updated_by = request.user
        if saved.status == Post.Status.PUBLISHED and not saved.published_at:
            saved.published_at = timezone.now()
        saved.save()
        form.save_m2m()
        messages.success(request, f"Post '{saved.title}' saved.")
        return redirect("cms:cms_posts")

    return render(request, "cms/admin/post_form.html", {"active": "admin_cms", "form": form, "post": post})


@login_required
@platform_team_required
def cms_leads(request):
    leads = Lead.objects.select_related("interested_package", "assigned_to")
    if request.GET.get("status"):
        leads = leads.filter(status=request.GET["status"])

    paginator = Paginator(leads, 40)
    page = paginator.get_page(request.GET.get("page"))
    return render(
        request,
        "cms/admin/leads.html",
        {
            "active": "admin_cms", "page_obj": page, "leads": page.object_list,
            "status_choices": Lead.Status.choices, "total": paginator.count,
        },
    )


@login_required
@platform_team_required
@require_POST
def cms_lead_status(request, pk):
    lead = get_object_or_404(Lead, pk=pk)
    status = request.POST.get("status")
    if status in dict(Lead.Status.choices):
        lead.status = status
        if status == Lead.Status.CONTACTED and not lead.contacted_at:
            lead.contacted_at = timezone.now()
        lead.internal_notes = request.POST.get("notes", lead.internal_notes)
        lead.save(update_fields=["status", "contacted_at", "internal_notes", "updated_at"])
        messages.success(request, f"Lead marked {status}.")
    return redirect("cms:cms_leads")


@login_required
@platform_team_required
def cms_settings(request):
    instance = SiteSettings.load()
    form = SiteSettingsForm(request.POST or None, request.FILES or None, instance=instance)
    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(request, "Site settings saved.")
        return redirect("cms:cms_settings")
    return render(request, "cms/admin/settings.html", {"active": "admin_cms", "form": form})


@login_required
@platform_team_required
def cms_media(request):
    if request.method == "POST":
        form = MediaAssetForm(request.POST, request.FILES)
        if form.is_valid():
            asset = form.save(commit=False)
            asset.created_by = request.user
            if asset.file:
                asset.file_size = asset.file.size
            asset.save()
            messages.success(request, f"'{asset.title}' uploaded.")
            return redirect("cms:cms_media")
    else:
        form = MediaAssetForm()

    return render(
        request, "cms/admin/media.html",
        {"active": "admin_cms", "form": form, "assets": MediaAsset.objects.all()[:120]},
    )


@login_required
@platform_team_required
def cms_faqs(request):
    if request.method == "POST":
        form = FAQForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "FAQ added.")
            return redirect("cms:cms_faqs")
    else:
        form = FAQForm()
    return render(
        request, "cms/admin/faqs.html",
        {"active": "admin_cms", "form": form, "faqs": FAQ.objects.all()},
    )


@login_required
@platform_team_required
def cms_testimonials(request):
    if request.method == "POST":
        form = TestimonialForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            messages.success(request, "Testimonial added.")
            return redirect("cms:cms_testimonials")
    else:
        form = TestimonialForm()
    return render(
        request, "cms/admin/testimonials.html",
        {"active": "admin_cms", "form": form, "testimonials": Testimonial.objects.all()},
    )


@login_required
@platform_team_required
def cms_menus(request):
    if request.method == "POST":
        form = MenuItemForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Menu item saved.")
            return redirect("cms:cms_menus")
    else:
        form = MenuItemForm()
    return render(
        request,
        "cms/admin/menus.html",
        {
            "active": "admin_cms", "form": form,
            "menus": Menu.objects.prefetch_related("items"),
        },
    )


@login_required
@platform_team_required
@require_POST
def cms_menu_item_delete(request, pk):
    item = get_object_or_404(MenuItem, pk=pk)
    item.delete()
    messages.success(request, "Menu item removed.")
    return redirect("cms:cms_menus")
