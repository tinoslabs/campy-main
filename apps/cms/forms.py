"""CMS editing forms and the public lead-capture forms."""
from __future__ import annotations

from django import forms

from .models import FAQ, ContentBlock, Lead, MediaAsset, MenuItem, Page, Post, SiteSettings, Testimonial


def _input(placeholder: str = "", **attrs):
    base = {"class": "input"}
    if placeholder:
        base["placeholder"] = placeholder
    base.update(attrs)
    return base


class ContactForm(forms.ModelForm):
    """Public contact / demo request. Includes a honeypot."""

    website = forms.CharField(
        required=False, widget=forms.TextInput(attrs={"style": "display:none", "tabindex": "-1", "autocomplete": "off"}),
        label="Leave this blank",
    )

    class Meta:
        model = Lead
        fields = [
            "name", "email", "phone", "company", "job_title",
            "country", "industry", "camera_count", "message",
        ]
        widgets = {
            "name": forms.TextInput(attrs=_input("Asha Menon", autofocus=True)),
            "email": forms.EmailInput(attrs=_input("you@company.com")),
            "phone": forms.TextInput(attrs=_input("+91 98765 43210")),
            "company": forms.TextInput(attrs=_input("Acme Manufacturing")),
            "job_title": forms.TextInput(attrs=_input("Head of Security")),
            "country": forms.TextInput(attrs=_input("India")),
            "industry": forms.TextInput(attrs=_input("Warehousing")),
            "camera_count": forms.Select(
                attrs={"class": "input"},
                choices=[
                    ("", "How many cameras?"), ("1-10", "1 – 10"), ("11-50", "11 – 50"),
                    ("51-200", "51 – 200"), ("200+", "More than 200"),
                ],
            ),
            "message": forms.Textarea(attrs=_input("Tell us what you would like to monitor", rows=4)),
        }

    def clean_website(self):
        """A filled honeypot means a bot; reject without telling it why."""
        if self.cleaned_data.get("website"):
            raise forms.ValidationError("Submission rejected.")
        return ""


class PageForm(forms.ModelForm):
    class Meta:
        model = Page
        fields = [
            "title", "slug", "parent", "template", "excerpt", "body", "hero_image",
            "status", "published_at", "show_in_navigation", "sort_order", "is_homepage",
            "seo_title", "seo_description", "seo_keywords", "og_image", "canonical_url", "no_index",
        ]
        widgets = {
            "title": forms.TextInput(attrs=_input("How it works", autofocus=True)),
            "slug": forms.TextInput(attrs=_input("Leave blank to generate from the title")),
            "parent": forms.Select(attrs={"class": "input"}),
            "template": forms.Select(attrs={"class": "input"}),
            "excerpt": forms.Textarea(attrs=_input(rows=2)),
            "body": forms.Textarea(attrs=_input(rows=14)),
            "status": forms.Select(attrs={"class": "input"}),
            "published_at": forms.DateTimeInput(attrs=_input(type="datetime-local"), format="%Y-%m-%dT%H:%M"),
            "sort_order": forms.NumberInput(attrs=_input(min="0")),
            "seo_title": forms.TextInput(attrs=_input("Under 60 characters")),
            "seo_description": forms.Textarea(attrs=_input("Under 160 characters", rows=2)),
            "seo_keywords": forms.TextInput(attrs=_input("cctv analytics, workplace safety")),
            "canonical_url": forms.URLInput(attrs=_input()),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["slug"].required = False
        self.fields["parent"].required = False
        self.fields["parent"].queryset = Page.objects.exclude(pk=self.instance.pk or 0)


class ContentBlockForm(forms.ModelForm):
    items_json = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs=_input('[{"title": "Fire detection", "body": "…"}]', rows=8)),
        label="Block items (JSON)",
        help_text="Feature cards, stats, steps and FAQ entries live here.",
    )

    class Meta:
        model = ContentBlock
        fields = [
            "kind", "sort_order", "is_visible", "heading", "subheading", "body", "image",
            "primary_label", "primary_url", "secondary_label", "secondary_url", "background",
        ]
        widgets = {
            "kind": forms.Select(attrs={"class": "input"}),
            "sort_order": forms.NumberInput(attrs=_input(min="0")),
            "heading": forms.TextInput(attrs=_input()),
            "subheading": forms.TextInput(attrs=_input()),
            "body": forms.Textarea(attrs=_input(rows=6)),
            "primary_label": forms.TextInput(attrs=_input("Start free trial")),
            "primary_url": forms.TextInput(attrs=_input("/accounts/signup/")),
            "secondary_label": forms.TextInput(attrs=_input("Book a demo")),
            "secondary_url": forms.TextInput(attrs=_input("/contact/")),
            "background": forms.Select(attrs={"class": "input"}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.pk:
            import json

            self.fields["items_json"].initial = json.dumps(self.instance.items or [], indent=2)

    def clean_items_json(self):
        import json

        raw = (self.cleaned_data.get("items_json") or "").strip()
        if not raw:
            return []
        try:
            value = json.loads(raw)
        except ValueError as exc:
            raise forms.ValidationError(f"That is not valid JSON: {exc}")
        if not isinstance(value, list):
            raise forms.ValidationError("Block items must be a JSON list.")
        return value

    def save(self, commit=True):
        block = super().save(commit=False)
        block.items = self.cleaned_data.get("items_json") or []
        if commit:
            block.save()
        return block


class PostForm(forms.ModelForm):
    class Meta:
        model = Post
        fields = [
            "title", "slug", "excerpt", "body", "cover_image", "category", "tags",
            "author_name", "status", "published_at", "is_featured",
            "seo_title", "seo_description", "og_image", "no_index",
        ]
        widgets = {
            "title": forms.TextInput(attrs=_input(autofocus=True)),
            "slug": forms.TextInput(attrs=_input("Generated from the title if blank")),
            "excerpt": forms.Textarea(attrs=_input(rows=2)),
            "body": forms.Textarea(attrs=_input(rows=18)),
            "category": forms.Select(attrs={"class": "input"}),
            "tags": forms.SelectMultiple(attrs={"class": "input", "size": 6}),
            "author_name": forms.TextInput(attrs=_input()),
            "status": forms.Select(attrs={"class": "input"}),
            "published_at": forms.DateTimeInput(attrs=_input(type="datetime-local"), format="%Y-%m-%dT%H:%M"),
            "seo_title": forms.TextInput(attrs=_input()),
            "seo_description": forms.Textarea(attrs=_input(rows=2)),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name in ("slug", "category", "tags"):
            self.fields[name].required = False


class SiteSettingsForm(forms.ModelForm):
    class Meta:
        model = SiteSettings
        exclude = ["created_at", "updated_at"]
        widgets = {
            "site_name": forms.TextInput(attrs=_input()),
            "tagline": forms.TextInput(attrs=_input()),
            "contact_email": forms.EmailInput(attrs=_input()),
            "support_email": forms.EmailInput(attrs=_input()),
            "sales_phone": forms.TextInput(attrs=_input()),
            "address": forms.Textarea(attrs=_input(rows=3)),
            "twitter_url": forms.URLInput(attrs=_input()),
            "linkedin_url": forms.URLInput(attrs=_input()),
            "youtube_url": forms.URLInput(attrs=_input()),
            "github_url": forms.URLInput(attrs=_input()),
            "default_seo_title": forms.TextInput(attrs=_input()),
            "default_seo_description": forms.Textarea(attrs=_input(rows=2)),
            "google_analytics_id": forms.TextInput(attrs=_input("G-XXXXXXXXXX")),
            "custom_head_html": forms.Textarea(attrs=_input(rows=4)),
            "custom_footer_html": forms.Textarea(attrs=_input(rows=4)),
            "primary_colour": forms.TextInput(attrs={"class": "input", "type": "color"}),
            "accent_colour": forms.TextInput(attrs={"class": "input", "type": "color"}),
            "maintenance_message": forms.Textarea(attrs=_input(rows=3)),
            "announcement": forms.TextInput(attrs=_input("New: fire detection is now available")),
            "announcement_url": forms.TextInput(attrs=_input("/blog/fire-detection/")),
        }


class FAQForm(forms.ModelForm):
    class Meta:
        model = FAQ
        fields = ["question", "answer", "category", "sort_order", "is_published"]
        widgets = {
            "question": forms.TextInput(attrs=_input(autofocus=True)),
            "answer": forms.Textarea(attrs=_input(rows=4)),
            "category": forms.TextInput(attrs=_input("Pricing")),
            "sort_order": forms.NumberInput(attrs=_input(min="0")),
        }


class TestimonialForm(forms.ModelForm):
    class Meta:
        model = Testimonial
        fields = [
            "quote", "author_name", "author_title", "company", "avatar", "logo",
            "rating", "industry", "is_featured", "is_published", "sort_order",
        ]
        widgets = {
            "quote": forms.Textarea(attrs=_input(rows=4, autofocus=True)),
            "author_name": forms.TextInput(attrs=_input()),
            "author_title": forms.TextInput(attrs=_input("Head of Operations")),
            "company": forms.TextInput(attrs=_input()),
            "rating": forms.NumberInput(attrs=_input(min="1", max="5")),
            "industry": forms.TextInput(attrs=_input()),
            "sort_order": forms.NumberInput(attrs=_input(min="0")),
        }


class MediaAssetForm(forms.ModelForm):
    class Meta:
        model = MediaAsset
        fields = ["title", "kind", "file", "alt_text", "caption"]
        widgets = {
            "title": forms.TextInput(attrs=_input(autofocus=True)),
            "kind": forms.Select(attrs={"class": "input"}),
            "alt_text": forms.TextInput(attrs=_input("Describe the image for screen readers")),
            "caption": forms.TextInput(attrs=_input()),
        }


class MenuItemForm(forms.ModelForm):
    class Meta:
        model = MenuItem
        fields = ["menu", "parent", "label", "url", "page", "named_url", "sort_order", "is_visible", "highlight"]
        widgets = {
            "menu": forms.Select(attrs={"class": "input"}),
            "parent": forms.Select(attrs={"class": "input"}),
            "label": forms.TextInput(attrs=_input(autofocus=True)),
            "url": forms.TextInput(attrs=_input("/pricing/ or https://…")),
            "page": forms.Select(attrs={"class": "input"}),
            "named_url": forms.TextInput(attrs=_input("cms:pricing")),
            "sort_order": forms.NumberInput(attrs=_input(min="0")),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name in ("parent", "url", "page", "named_url"):
            self.fields[name].required = False
