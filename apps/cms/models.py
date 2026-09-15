"""Full content-management system for the Campy AI marketing site."""
from __future__ import annotations

from django.conf import settings
from django.db import models
from django.urls import reverse
from django.utils import timezone
from django.utils.text import slugify

from apps.core.models import (
    ActorStampedModel,
    SluggedModel,
    SoftDeleteModel,
    TimeStampedModel,
    UUIDModel,
)


class PublishableQuerySet(models.QuerySet):
    def published(self):
        now = timezone.now()
        return self.filter(status="published").filter(
            models.Q(published_at__isnull=True) | models.Q(published_at__lte=now)
        )


class PublishableModel(models.Model):
    """Shared publishing workflow: draft → review → published."""

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        REVIEW = "review", "In review"
        PUBLISHED = "published", "Published"
        ARCHIVED = "archived", "Archived"

    status = models.CharField(max_length=10, choices=Status.choices, default=Status.DRAFT, db_index=True)
    published_at = models.DateTimeField(null=True, blank=True)
    published_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True, blank=True, on_delete=models.SET_NULL,
        related_name="%(app_label)s_%(class)s_published",
    )

    objects = PublishableQuerySet.as_manager()

    class Meta:
        abstract = True

    @property
    def is_published(self) -> bool:
        if self.status != self.Status.PUBLISHED:
            return False
        return self.published_at is None or self.published_at <= timezone.now()

    def publish(self, user=None) -> None:
        self.status = self.Status.PUBLISHED
        self.published_at = self.published_at or timezone.now()
        self.published_by = user
        self.save(update_fields=["status", "published_at", "published_by", "updated_at"])

    def unpublish(self) -> None:
        self.status = self.Status.DRAFT
        self.save(update_fields=["status", "updated_at"])


class SEOModel(models.Model):
    """Per-page search and social metadata."""

    seo_title = models.CharField(max_length=70, blank=True, help_text="Best under 60 characters")
    seo_description = models.CharField(max_length=180, blank=True, help_text="Best under 160 characters")
    seo_keywords = models.CharField(max_length=250, blank=True)
    og_image = models.ImageField(upload_to="cms/og/", blank=True, null=True)
    canonical_url = models.URLField(blank=True)
    no_index = models.BooleanField(default=False)

    class Meta:
        abstract = True

    @property
    def meta_title(self) -> str:
        return self.seo_title or getattr(self, "title", "") or getattr(self, "name", "")

    @property
    def meta_description(self) -> str:
        return self.seo_description or (getattr(self, "excerpt", "") or "")[:180]


# ---------------------------------------------------------------------------
# Pages & blocks
# ---------------------------------------------------------------------------
class Page(UUIDModel, TimeStampedModel, PublishableModel, SEOModel, SoftDeleteModel, ActorStampedModel):
    """A marketing page assembled from ordered content blocks."""

    class Template(models.TextChoices):
        DEFAULT = "default", "Standard page"
        LANDING = "landing", "Landing page"
        FULL_WIDTH = "full_width", "Full width"
        LEGAL = "legal", "Legal / policy"
        CONTACT = "contact", "Contact"

    title = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True, blank=True)
    parent = models.ForeignKey("self", null=True, blank=True, on_delete=models.SET_NULL, related_name="children")
    template = models.CharField(max_length=16, choices=Template.choices, default=Template.DEFAULT)

    excerpt = models.TextField(blank=True)
    body = models.TextField(blank=True, help_text="Markdown-ish body used when no blocks are defined")
    hero_image = models.ImageField(upload_to="cms/pages/", blank=True, null=True)

    show_in_navigation = models.BooleanField(default=False)
    sort_order = models.PositiveIntegerField(default=100)
    is_homepage = models.BooleanField(default=False)
    view_count = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["sort_order", "title"]
        indexes = [models.Index(fields=["status", "slug"])]

    def __str__(self) -> str:
        return self.title

    def save(self, *args, **kwargs):
        if not self.slug:
            base = slugify(self.title)[:190] or "page"
            candidate, counter = base, 2
            while Page.objects.filter(slug=candidate).exclude(pk=self.pk).exists():
                candidate = f"{base}-{counter}"
                counter += 1
            self.slug = candidate
        if self.is_homepage:
            Page.objects.filter(is_homepage=True).exclude(pk=self.pk).update(is_homepage=False)
        super().save(*args, **kwargs)

    def get_absolute_url(self) -> str:
        if self.is_homepage:
            return reverse("cms:home")
        return reverse("cms:page", args=[self.slug])

    @property
    def breadcrumbs(self) -> list:
        chain, node, guard = [], self, 0
        while node is not None and guard < 8:
            chain.insert(0, node)
            node = node.parent
            guard += 1
        return chain

    @property
    def visible_blocks(self):
        return self.blocks.filter(is_visible=True).order_by("sort_order")


class ContentBlock(UUIDModel, TimeStampedModel):
    """One section of a page. The block *kind* selects the template partial."""

    class Kind(models.TextChoices):
        HERO = "hero", "Hero banner"
        RICH_TEXT = "rich_text", "Rich text"
        FEATURES = "features", "Feature grid"
        STATS = "stats", "Statistics band"
        CTA = "cta", "Call to action"
        TESTIMONIALS = "testimonials", "Testimonials"
        PRICING = "pricing", "Pricing table"
        FAQ = "faq", "FAQ accordion"
        LOGOS = "logos", "Logo strip"
        GALLERY = "gallery", "Image gallery"
        STEPS = "steps", "Process steps"
        VIDEO = "video", "Video embed"
        CONTACT_FORM = "contact_form", "Contact form"
        HTML = "html", "Raw HTML"

    page = models.ForeignKey(Page, on_delete=models.CASCADE, related_name="blocks")
    kind = models.CharField(max_length=16, choices=Kind.choices, default=Kind.RICH_TEXT)
    sort_order = models.PositiveIntegerField(default=100)
    is_visible = models.BooleanField(default=True)

    heading = models.CharField(max_length=220, blank=True)
    subheading = models.CharField(max_length=320, blank=True)
    body = models.TextField(blank=True)
    image = models.ImageField(upload_to="cms/blocks/", blank=True, null=True)

    primary_label = models.CharField(max_length=60, blank=True)
    primary_url = models.CharField(max_length=300, blank=True)
    secondary_label = models.CharField(max_length=60, blank=True)
    secondary_url = models.CharField(max_length=300, blank=True)

    #: Kind-specific structured payload, e.g. the list of feature cards.
    items = models.JSONField(default=list, blank=True)
    options = models.JSONField(default=dict, blank=True)
    background = models.CharField(
        max_length=12,
        choices=[("light", "Light"), ("dark", "Dark"), ("accent", "Accent"), ("muted", "Muted")],
        default="light",
    )

    class Meta:
        ordering = ["sort_order"]

    def __str__(self) -> str:
        return f"{self.get_kind_display()} — {self.heading or self.page.title}"

    @property
    def template_name(self) -> str:
        return f"cms/blocks/{self.kind}.html"


# ---------------------------------------------------------------------------
# Blog
# ---------------------------------------------------------------------------
class Category(UUIDModel, SluggedModel, TimeStampedModel):
    name = models.CharField(max_length=120)
    description = models.TextField(blank=True)
    colour = models.CharField(max_length=7, default="#4f46e5")
    sort_order = models.PositiveIntegerField(default=100)

    class Meta:
        ordering = ["sort_order", "name"]
        verbose_name_plural = "categories"

    def __str__(self) -> str:
        return self.name

    def get_absolute_url(self) -> str:
        return reverse("cms:blog_category", args=[self.slug])


class Tag(UUIDModel, SluggedModel, TimeStampedModel):
    name = models.CharField(max_length=80)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name

    def get_absolute_url(self) -> str:
        return reverse("cms:blog_tag", args=[self.slug])


class Post(UUIDModel, TimeStampedModel, PublishableModel, SEOModel, SoftDeleteModel, ActorStampedModel):
    title = models.CharField(max_length=240)
    slug = models.SlugField(max_length=240, unique=True, blank=True)
    excerpt = models.TextField(blank=True)
    body = models.TextField(blank=True)
    cover_image = models.ImageField(upload_to="cms/posts/", blank=True, null=True)

    category = models.ForeignKey(Category, null=True, blank=True, on_delete=models.SET_NULL, related_name="posts")
    tags = models.ManyToManyField(Tag, blank=True, related_name="posts")
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="posts"
    )
    author_name = models.CharField(max_length=140, blank=True, help_text="Overrides the linked author")

    reading_minutes = models.PositiveIntegerField(default=0)
    is_featured = models.BooleanField(default=False)
    view_count = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["-published_at", "-created_at"]
        indexes = [models.Index(fields=["status", "-published_at"])]

    def __str__(self) -> str:
        return self.title

    def save(self, *args, **kwargs):
        if not self.slug:
            base = slugify(self.title)[:230] or "post"
            candidate, counter = base, 2
            while Post.objects.filter(slug=candidate).exclude(pk=self.pk).exists():
                candidate = f"{base}-{counter}"
                counter += 1
            self.slug = candidate
        words = len((self.body or "").split())
        self.reading_minutes = max(1, round(words / 200)) if words else 1
        super().save(*args, **kwargs)

    def get_absolute_url(self) -> str:
        return reverse("cms:blog_post", args=[self.slug])

    @property
    def display_author(self) -> str:
        return self.author_name or (str(self.author) if self.author else "Campy AI")


# ---------------------------------------------------------------------------
# Navigation, media, site settings
# ---------------------------------------------------------------------------
class Menu(UUIDModel, TimeStampedModel):
    class Location(models.TextChoices):
        HEADER = "header", "Header"
        FOOTER_PRODUCT = "footer_product", "Footer — Product"
        FOOTER_COMPANY = "footer_company", "Footer — Company"
        FOOTER_LEGAL = "footer_legal", "Footer — Legal"
        DASHBOARD = "dashboard", "Dashboard"

    name = models.CharField(max_length=120)
    location = models.CharField(max_length=20, choices=Location.choices, unique=True)

    class Meta:
        ordering = ["location"]

    def __str__(self) -> str:
        return self.name

    @property
    def root_items(self):
        return self.items.filter(parent__isnull=True, is_visible=True).order_by("sort_order")


class MenuItem(UUIDModel, TimeStampedModel):
    menu = models.ForeignKey(Menu, on_delete=models.CASCADE, related_name="items")
    parent = models.ForeignKey("self", null=True, blank=True, on_delete=models.CASCADE, related_name="children")
    label = models.CharField(max_length=80)
    url = models.CharField(max_length=300, blank=True)
    page = models.ForeignKey(Page, null=True, blank=True, on_delete=models.SET_NULL, related_name="menu_items")
    named_url = models.CharField(max_length=120, blank=True, help_text="Django URL name, e.g. cms:pricing")
    icon = models.CharField(max_length=40, blank=True)
    sort_order = models.PositiveIntegerField(default=100)
    is_visible = models.BooleanField(default=True)
    open_in_new_tab = models.BooleanField(default=False)
    highlight = models.BooleanField(default=False, help_text="Render as a button")

    class Meta:
        ordering = ["sort_order", "label"]

    def __str__(self) -> str:
        return self.label

    @property
    def resolved_url(self) -> str:
        if self.page_id:
            return self.page.get_absolute_url()
        if self.named_url:
            try:
                return reverse(self.named_url)
            except Exception:  # noqa: BLE001 - a bad name must not break the nav
                return "#"
        return self.url or "#"

    @property
    def visible_children(self):
        return self.children.filter(is_visible=True).order_by("sort_order")


class MediaAsset(UUIDModel, TimeStampedModel, ActorStampedModel):
    """Central media library shared by pages, posts and blocks."""

    class Kind(models.TextChoices):
        IMAGE = "image", "Image"
        DOCUMENT = "document", "Document"
        VIDEO = "video", "Video"

    title = models.CharField(max_length=200)
    kind = models.CharField(max_length=10, choices=Kind.choices, default=Kind.IMAGE)
    file = models.FileField(upload_to="cms/library/%Y/%m/")
    alt_text = models.CharField(max_length=250, blank=True, help_text="Required for accessibility")
    caption = models.CharField(max_length=300, blank=True)
    tags = models.JSONField(default=list, blank=True)
    width = models.PositiveIntegerField(default=0)
    height = models.PositiveIntegerField(default=0)
    file_size = models.BigIntegerField(default=0)
    usage_count = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.title

    @property
    def size_label(self) -> str:
        size = float(self.file_size)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.0f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"


class FAQ(UUIDModel, TimeStampedModel):
    question = models.CharField(max_length=300)
    answer = models.TextField()
    category = models.CharField(max_length=80, blank=True)
    sort_order = models.PositiveIntegerField(default=100)
    is_published = models.BooleanField(default=True)
    helpful_count = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["sort_order", "question"]
        verbose_name = "FAQ"
        verbose_name_plural = "FAQs"

    def __str__(self) -> str:
        return self.question


class Testimonial(UUIDModel, TimeStampedModel):
    quote = models.TextField()
    author_name = models.CharField(max_length=140)
    author_title = models.CharField(max_length=160, blank=True)
    company = models.CharField(max_length=160, blank=True)
    avatar = models.ImageField(upload_to="cms/testimonials/", blank=True, null=True)
    logo = models.ImageField(upload_to="cms/logos/", blank=True, null=True)
    rating = models.PositiveSmallIntegerField(default=5)
    industry = models.CharField(max_length=80, blank=True)
    is_featured = models.BooleanField(default=False)
    is_published = models.BooleanField(default=True)
    sort_order = models.PositiveIntegerField(default=100)

    class Meta:
        ordering = ["sort_order", "-created_at"]

    def __str__(self) -> str:
        return f"{self.author_name} — {self.company}"

    @property
    def stars(self) -> range:
        return range(min(max(self.rating, 0), 5))


class Lead(UUIDModel, TimeStampedModel):
    """A contact / demo request from the marketing site."""

    class Status(models.TextChoices):
        NEW = "new", "New"
        CONTACTED = "contacted", "Contacted"
        QUALIFIED = "qualified", "Qualified"
        CONVERTED = "converted", "Converted"
        CLOSED = "closed", "Closed"

    class Source(models.TextChoices):
        CONTACT = "contact", "Contact form"
        DEMO = "demo", "Demo request"
        PRICING = "pricing", "Pricing enquiry"
        NEWSLETTER = "newsletter", "Newsletter"

    name = models.CharField(max_length=160)
    email = models.EmailField()
    phone = models.CharField(max_length=32, blank=True)
    company = models.CharField(max_length=180, blank=True)
    job_title = models.CharField(max_length=120, blank=True)
    country = models.CharField(max_length=80, blank=True)
    industry = models.CharField(max_length=80, blank=True)
    camera_count = models.CharField(max_length=40, blank=True)
    message = models.TextField(blank=True)

    source = models.CharField(max_length=12, choices=Source.choices, default=Source.CONTACT)
    interested_package = models.ForeignKey(
        "billing.Package", null=True, blank=True, on_delete=models.SET_NULL, related_name="leads"
    )
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.NEW, db_index=True)

    assigned_to = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="leads"
    )
    internal_notes = models.TextField(blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    referrer = models.CharField(max_length=300, blank=True)
    utm = models.JSONField(default=dict, blank=True)
    contacted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["status", "-created_at"])]

    def __str__(self) -> str:
        return f"{self.name} <{self.email}>"


class SiteSettings(TimeStampedModel):
    """Singleton holding global site configuration editable from the CMS."""

    site_name = models.CharField(max_length=120, default="Campy AI")
    tagline = models.CharField(max_length=220, default="Turn ordinary CCTV into workplace intelligence")
    logo = models.ImageField(upload_to="cms/brand/", blank=True, null=True)
    logo_dark = models.ImageField(upload_to="cms/brand/", blank=True, null=True)
    favicon = models.ImageField(upload_to="cms/brand/", blank=True, null=True)

    contact_email = models.EmailField(default="hello@campy.ai")
    support_email = models.EmailField(default="support@campy.ai")
    sales_phone = models.CharField(max_length=32, blank=True)
    address = models.TextField(blank=True)

    twitter_url = models.URLField(blank=True)
    linkedin_url = models.URLField(blank=True)
    youtube_url = models.URLField(blank=True)
    github_url = models.URLField(blank=True)

    default_seo_title = models.CharField(max_length=70, blank=True)
    default_seo_description = models.CharField(max_length=180, blank=True)
    google_analytics_id = models.CharField(max_length=40, blank=True)
    custom_head_html = models.TextField(blank=True)
    custom_footer_html = models.TextField(blank=True)

    primary_colour = models.CharField(max_length=7, default="#4f46e5")
    accent_colour = models.CharField(max_length=7, default="#06b6d4")

    maintenance_mode = models.BooleanField(default=False)
    maintenance_message = models.TextField(blank=True)
    signups_enabled = models.BooleanField(default=True)
    announcement = models.CharField(max_length=300, blank=True)
    announcement_url = models.CharField(max_length=300, blank=True)

    class Meta:
        verbose_name = "Site settings"
        verbose_name_plural = "Site settings"

    def __str__(self) -> str:
        return self.site_name

    def save(self, *args, **kwargs):
        self.pk = 1          # enforce the singleton
        super().save(*args, **kwargs)

    @classmethod
    def load(cls) -> "SiteSettings":
        instance, _ = cls.objects.get_or_create(pk=1)
        return instance
