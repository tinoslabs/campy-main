"""Populate a working Campy AI deployment: packages, a demo workspace, CMS content.

Run once after ``migrate``::

    python manage.py seed_demo

It is idempotent — running it again updates rather than duplicates.
"""
from __future__ import annotations

import random
from datetime import timedelta
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

User = get_user_model()


class Command(BaseCommand):
    help = "Seed packages, CMS content and a fully populated demo workspace."

    def add_arguments(self, parser):
        parser.add_argument("--password", default="CampyDemo!2024", help="Password for the seeded accounts")
        parser.add_argument("--events", type=int, default=140, help="How many demo events to generate")
        parser.add_argument("--skip-demo-org", action="store_true", help="Only seed packages and CMS")

    @transaction.atomic
    def handle(self, *args, **options):
        password = options["password"]
        self.stdout.write(self.style.MIGRATE_HEADING("Seeding Campy AI"))

        packages = self.seed_packages()
        self.seed_cms()
        superadmin = self.seed_superadmin(password)

        if not options["skip_demo_org"]:
            self.seed_demo_workspace(packages, password, options["events"])

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("Seed complete."))
        self.stdout.write("")
        self.stdout.write(self.style.HTTP_INFO("  Sign in at /accounts/login/"))
        self.stdout.write(f"  Super admin      admin@campy.ai            {password}")
        if not options["skip_demo_org"]:
            self.stdout.write(f"  Workspace owner  owner@acme-demo.com       {password}")
            self.stdout.write(f"  Security manager security@acme-demo.com    {password}")
            self.stdout.write(f"  Operator         operator@acme-demo.com    {password}")
            self.stdout.write(f"  AI engineer      ai@acme-demo.com          {password}")
            self.stdout.write(f"  Viewer           viewer@acme-demo.com      {password}")
        _ = superadmin

    # ------------------------------------------------------------------
    # Packages
    # ------------------------------------------------------------------
    def seed_packages(self):
        from apps.billing.models import Package

        definitions = [
            {
                "name": "Starter", "tagline": "One location, the essentials switched on.",
                "price_inr": Decimal("4999"), "price_usd": Decimal("69"),
                "trial_days": 14, "sort_order": 10, "colour": "#0891b2",
                "description": "For a single office, store or clinic getting started with camera analytics.",
                "features": {
                    "gesture_tracking": True, "geofencing": True, "crowd_management": True,
                    "fire_detection": True, "evidence_clips": True,
                },
                "quotas": {
                    "cameras": 5, "sites": 1, "users": 3, "employees": 25,
                    "storage_gb": 20, "retention_days": 30,
                    "training_jobs_month": 0, "api_calls_day": 0,
                },
            },
            {
                "name": "Professional", "tagline": "Multi-site operations with recognition and theft cover.",
                "price_inr": Decimal("14999"), "price_usd": Decimal("199"),
                "trial_days": 14, "sort_order": 20, "is_featured": True,
                "badge": "Most popular", "colour": "#4f46e5",
                "per_camera_inr": Decimal("499"), "per_camera_usd": Decimal("7"),
                "description": "Everything in Starter plus face recognition, theft detection, dangerous objects, reports and the API.",
                "features": {
                    "gesture_tracking": True, "face_recognition": True, "geofencing": True,
                    "crowd_management": True, "theft_detection": True, "object_detection": True,
                    "fire_detection": True, "custom_training": True, "reports": True,
                    "api_access": True, "multi_site": True, "evidence_clips": True,
                    "audit_log": True,
                },
                "quotas": {
                    "cameras": 25, "sites": 5, "users": 15, "employees": 250,
                    "storage_gb": 200, "retention_days": 90,
                    "training_jobs_month": 10, "api_calls_day": 50000,
                },
            },
            {
                "name": "Enterprise", "tagline": "Unlimited scale, your own trained models, priority support.",
                "price_inr": Decimal("49999"), "price_usd": Decimal("649"),
                "trial_days": 30, "sort_order": 30, "colour": "#7c3aed",
                "setup_fee_inr": Decimal("25000"), "setup_fee_usd": Decimal("299"),
                "per_camera_inr": Decimal("299"), "per_camera_usd": Decimal("4"),
                "description": "Every capability, unlimited cameras and sites, custom model training, SSO and white labelling.",
                "features": {key: True for key, _ in __import__(
                    "apps.billing.models", fromlist=["FEATURES"]).FEATURES},
                "quotas": {
                    "cameras": -1, "sites": -1, "users": -1, "employees": -1,
                    "storage_gb": 2000, "retention_days": 365,
                    "training_jobs_month": -1, "api_calls_day": -1,
                },
            },
        ]

        created = []
        for spec in definitions:
            package, was_created = Package.objects.update_or_create(
                name=spec["name"],
                defaults={
                    "tagline": spec["tagline"],
                    "description": spec.get("description", ""),
                    "price_inr": spec["price_inr"], "price_usd": spec["price_usd"],
                    "interval": Package.Interval.MONTHLY,
                    "trial_days": spec["trial_days"],
                    "setup_fee_inr": spec.get("setup_fee_inr", Decimal("0")),
                    "setup_fee_usd": spec.get("setup_fee_usd", Decimal("0")),
                    "per_camera_inr": spec.get("per_camera_inr", Decimal("0")),
                    "per_camera_usd": spec.get("per_camera_usd", Decimal("0")),
                    "features": spec["features"], "quotas": spec["quotas"],
                    "audience": Package.Audience.PUBLIC, "is_active": True,
                    "is_featured": spec.get("is_featured", False),
                    "sort_order": spec["sort_order"],
                    "badge": spec.get("badge", ""), "colour": spec["colour"],
                },
            )
            created.append(package)
            self.stdout.write(f"  {'created' if was_created else 'updated'} package: {package.name}")
        return {p.name: p for p in created}

    # ------------------------------------------------------------------
    # Platform accounts
    # ------------------------------------------------------------------
    def seed_superadmin(self, password: str):
        admin = User.objects.filter(email="admin@campy.ai").first()
        if admin is None:
            admin = User.objects.create_superuser(
                email="admin@campy.ai", password=password,
                full_name="Campy AI Administrator", username="campyadmin",
            )
            self.stdout.write("  created super admin: admin@campy.ai")
        else:
            admin.set_password(password)
            admin.platform_role = User.PlatformRole.SUPERADMIN
            admin.is_superuser = admin.is_staff = True
            admin.save()
            self.stdout.write("  updated super admin: admin@campy.ai")
        return admin

    # ------------------------------------------------------------------
    # CMS
    # ------------------------------------------------------------------
    def seed_cms(self):
        from apps.cms.models import FAQ, Category, Menu, MenuItem, Post, SiteSettings, Testimonial

        settings_obj = SiteSettings.load()
        settings_obj.site_name = "Campy AI"
        settings_obj.tagline = "Turn ordinary CCTV into workplace intelligence"
        settings_obj.contact_email = "hello@campy.ai"
        settings_obj.support_email = "support@campy.ai"
        settings_obj.default_seo_title = "Campy AI — AI analytics for the CCTV you already own"
        settings_obj.default_seo_description = (
            "Gesture tracking, face recognition, geofencing, crowd management, theft, "
            "dangerous object and fire detection on your existing cameras."
        )
        settings_obj.save()

        # -- navigation ------------------------------------------------
        menus = {
            Menu.Location.HEADER: [
                ("Features", "cms:features"), ("Pricing", "cms:pricing"),
                ("Blog", "cms:blog"), ("FAQ", "cms:faq"), ("Contact", "cms:contact"),
            ],
            Menu.Location.FOOTER_PRODUCT: [
                ("Features", "cms:features"), ("Pricing", "cms:pricing"), ("Book a demo", "cms:demo"),
            ],
            Menu.Location.FOOTER_COMPANY: [
                ("Blog", "cms:blog"), ("FAQ", "cms:faq"), ("Contact", "cms:contact"),
            ],
        }
        for location, items in menus.items():
            menu, _ = Menu.objects.get_or_create(
                location=location, defaults={"name": location.replace("_", " ").title()}
            )
            for index, (label, named_url) in enumerate(items):
                MenuItem.objects.update_or_create(
                    menu=menu, label=label,
                    defaults={"named_url": named_url, "sort_order": (index + 1) * 10},
                )

        # -- FAQs ------------------------------------------------------
        faqs = [
            ("Pricing", "Do I need to buy new cameras?",
             "No. Campy AI works with the CCTV you already own — anything that exposes RTSP, ONVIF, "
             "HTTP/MJPEG or HLS, which covers virtually every IP camera and NVR sold in the last decade."),
            ("Pricing", "What happens when my trial ends?",
             "Nothing is deleted. The workspace moves to a past-due state with a seven-day grace period, "
             "and monitoring keeps running while you decide. We do not switch off a security system the "
             "moment a card fails."),
            ("Pricing", "Can I change plans later?",
             "Yes, at any time. Upgrading applies immediately; downgrading applies from your next billing period "
             "so you never lose access you have already paid for."),
            ("Privacy", "Do you store our video?",
             "Only short evidence snapshots attached to events, and only if you leave that switched on. "
             "Retention is enforced automatically and you choose the window."),
            ("Privacy", "How does face recognition handle consent?",
             "Enrolment stays disabled for a person until you record their written consent in their profile. "
             "Enrolment then stores a mathematical signature, not a photograph, and a face cannot be "
             "reconstructed from it. Erasing it is one click."),
            ("Technical", "Where does the AI run?",
             "On your Campy AI deployment. The models are ours end to end — architecture, training loop and "
             "runtime — and no frame is ever sent to a third-party AI service."),
            ("Technical", "How many cameras can one server handle?",
             "With all seven analytics switched on, plan for two to three cameras per CPU core at a "
             "6 fps analysis rate. Cameras that need only one or two analytics, or that watch a quiet "
             "area, cost far less — and the benchmark command measures your own hardware exactly."),
            ("Technical", "Can I train it on my own footage?",
             "Yes — that is the point. Upload and label frames in the browser, press train, watch the "
             "accuracy climb, and deploy to every camera with one click."),
        ]
        for order, (category, question, answer) in enumerate(faqs):
            FAQ.objects.update_or_create(
                question=question,
                defaults={"answer": answer, "category": category, "sort_order": (order + 1) * 10},
            )

        # -- testimonials ----------------------------------------------
        testimonials = [
            ("We stopped finding out about incidents the next morning. The fire alert reached the shift "
             "supervisor's phone before the smoke reached the ceiling sensor.",
             "Ravi Subramanian", "Plant Safety Manager", "Northline Manufacturing", "manufacturing"),
            ("Our stock loss investigation used to mean four hours of scrubbing footage. Now we open the "
             "incident and the relevant ninety seconds are already grouped together.",
             "Anita Desai", "Head of Operations", "Meridian Warehousing", "warehouse"),
            ("The geofence around the server room paid for the whole system in the first month. It caught "
             "an access-badge sharing problem we had no idea existed.",
             "Thomas Okafor", "IT Security Lead", "Cardinal Financial", "bank"),
        ]
        for order, (quote, name, title, company, industry) in enumerate(testimonials):
            Testimonial.objects.update_or_create(
                author_name=name,
                defaults={
                    "quote": quote, "author_title": title, "company": company,
                    "industry": industry, "rating": 5, "is_published": True,
                    "is_featured": order == 0, "sort_order": (order + 1) * 10,
                },
            )

        # -- blog ------------------------------------------------------
        category, _ = Category.objects.get_or_create(
            name="Product", defaults={"description": "How Campy AI works and why."}
        )
        posts = [
            {
                "title": "Why your CCTV is a recording device, not a security system",
                "excerpt": "Cameras that only help after the fact are an insurance artefact. Here is what "
                           "changes when something actually watches them.",
                "body": (
                    "Most organisations own more camera coverage than they have ever meaningfully reviewed.\n\n"
                    "The footage exists. The problem is that watching it is a human task, and humans do not "
                    "scale to forty simultaneous feeds for eight hours at a time. Research on control-room "
                    "attention is consistent and unflattering: sustained visual monitoring degrades sharply "
                    "within twenty minutes.\n\n"
                    "So the camera becomes an evidence archive. Something goes wrong, somebody scrubs back, "
                    "and the organisation learns what happened after it has already cost them.\n\n"
                    "The shift is not better cameras. It is putting something on the feed that does not get "
                    "bored — that notices the restricted door opening at 21:14, the corridor filling past "
                    "safe occupancy, or the flicker pattern that is a fire rather than a reflection, and "
                    "says so in a sentence a human can act on immediately.\n\n"
                    "That is the whole design goal of Campy AI: not more data, fewer but better-timed "
                    "interruptions."
                ),
            },
            {
                "title": "Why we built our own models instead of calling an AI API",
                "excerpt": "Owning the architecture, the training loop and the runtime is not ideology. "
                           "It is what makes the product deployable at all.",
                "body": (
                    "There is an obvious shortcut when building camera analytics: send frames to somebody "
                    "else's vision API and render the results.\n\n"
                    "We did not take it, for three reasons.\n\n"
                    "**Footage cannot leave.** The customers who most need this — banks, hospitals, "
                    "manufacturing plants — cannot lawfully or contractually stream their camera feeds to a "
                    "third-party service. That rules the approach out before performance is even discussed.\n\n"
                    "**Per-frame pricing does not survive contact with video.** One camera at six frames per "
                    "second is over half a million frames a day. Multiply by forty cameras and the API bill "
                    "exceeds what any customer would pay for the product.\n\n"
                    "**Generic models do not know your site.** A model trained on internet photographs has "
                    "no concept of your uniform, your machinery, or what 'unsafe' means on your floor. Ours "
                    "are small enough to retrain on a few hundred of your own frames, in the browser, in a "
                    "minute.\n\n"
                    "The cost is that we had to build the whole stack — layers, gradients, optimisers, "
                    "training loop, weights format, inference runtime. The benefit is that seven analytics "
                    "run together on ordinary CPU cores — two to three cameras per core with everything "
                    "on — entirely inside your deployment, with nothing metered per frame."
                ),
            },
            {
                "title": "Alert fatigue is a design failure, not a user failure",
                "excerpt": "A detector that fires four hundred times a day gets muted. Here is how we "
                           "designed against that from the start.",
                "body": (
                    "Every monitoring system fails the same way. It works, it alerts, people act on the "
                    "alerts, and then — as the volume grows — people stop.\n\n"
                    "Muting is a rational response to a system that cries wolf. So the interesting design "
                    "question is not 'how sensitive can we make the detector', it is 'how few alerts can we "
                    "send while still catching what matters'.\n\n"
                    "Campy AI attacks this in four places.\n\n"
                    "**Persistence before firing.** A finding must hold for several consecutive frames. A "
                    "single-frame flicker is noise, not an event.\n\n"
                    "**Cooldowns.** Once a condition has been reported, it is not reported again until it "
                    "has genuinely changed.\n\n"
                    "**Incident grouping.** Ten geofence breaches by the same person over two minutes is one "
                    "incident with ten frames of evidence, not ten separate alerts.\n\n"
                    "**Feedback that actually feeds back.** When an operator marks an alert as a false alarm, "
                    "that frame becomes a hard negative in the next training run. The system's noise floor "
                    "drops because people used it, not despite it."
                ),
            },
        ]
        for post in posts:
            Post.objects.update_or_create(
                title=post["title"],
                defaults={
                    "excerpt": post["excerpt"], "body": post["body"], "category": category,
                    "author_name": "The Campy AI team", "status": Post.Status.PUBLISHED,
                    "published_at": timezone.now() - timedelta(days=random.randint(3, 40)),
                },
            )

        self.stdout.write("  seeded CMS: settings, menus, FAQs, testimonials, blog posts")

    # ------------------------------------------------------------------
    # Demo workspace
    # ------------------------------------------------------------------
    def seed_demo_workspace(self, packages, password: str, event_count: int):
        from apps.accounts.models import Membership, Organization
        from apps.billing.services import activate_subscription
        from apps.cameras.models import Camera, Employee, Site, Zone
        from apps.events.models import AlertRule, NotificationChannel

        organization, created = Organization.objects.update_or_create(
            slug="acme-demo",
            defaults={
                "name": "Acme Manufacturing (Demo)",
                "legal_name": "Acme Manufacturing Private Limited",
                "industry": Organization.Industry.MANUFACTURING,
                "status": Organization.Status.ACTIVE,
                "contact_email": "owner@acme-demo.com",
                "contact_phone": "+91 80 4567 8900",
                "address_line1": "Plot 14, Industrial Layout",
                "city": "Bengaluru", "state": "Karnataka",
                "postal_code": "560068", "country": "IN",
                "tax_id": "29ABCDE1234F1Z5", "timezone": "Asia/Kolkata",
                "onboarded_at": timezone.now(),
            },
        )
        organization.bootstrap_default_roles()
        self.stdout.write(f"  {'created' if created else 'updated'} workspace: {organization.name}")

        activate_subscription(organization, packages["Professional"], currency="INR")

        # -- people ----------------------------------------------------
        people = [
            ("owner@acme-demo.com", "Priya Raghavan", "owner", "Managing Director"),
            ("security@acme-demo.com", "Ravi Subramanian", "security_manager", "Head of Security"),
            ("operator@acme-demo.com", "Deepak Nair", "operator", "Control Room Operator"),
            ("ai@acme-demo.com", "Meera Krishnan", "ai_engineer", "AI Engineer"),
            ("viewer@acme-demo.com", "Sanjay Gupta", "viewer", "Operations Analyst"),
        ]
        for email, name, role_code, title in people:
            user, user_created = User.objects.get_or_create(
                email=email, defaults={"full_name": name, "email_verified": True}
            )
            user.set_password(password)
            user.full_name = name
            user.active_organization = organization
            user.save()
            role = organization.roles.get(code=role_code)
            Membership.objects.update_or_create(
                user=user, organization=organization,
                defaults={"role": role, "title": title, "status": Membership.Status.ACTIVE},
            )
            self.stdout.write(f"    {'+' if user_created else '·'} {name} ({role.name})")

        # -- sites & cameras -------------------------------------------
        site_specs = [
            ("Bengaluru Plant", "BLR-01", "Plot 14, Industrial Layout, Bengaluru"),
            ("Chennai Warehouse", "MAA-02", "Survey 88, Sriperumbudur, Chennai"),
        ]
        sites = []
        for name, code, address in site_specs:
            site, _ = Site.objects.update_or_create(
                organization=organization, name=name,
                defaults={
                    "code": code, "address": address, "city": name.split()[0],
                    "country": "IN", "timezone": "Asia/Kolkata", "is_active": True,
                    "working_hours": {day: ["08:00", "20:00"] for day in
                                      ["mon", "tue", "wed", "thu", "fri", "sat"]},
                },
            )
            sites.append(site)

        camera_specs = [
            (0, "Main Entrance", "Above reception, facing the doors", "walk",
             ["gesture", "face", "crowd", "geofence"]),
            (0, "Server Room Corridor", "Ceiling mount outside the server room", "restricted",
             ["gesture", "geofence", "face", "object"]),
            (0, "Assembly Line A", "Gantry above station 3", "running",
             ["gesture", "fire", "object", "geofence"]),
            (0, "Canteen", "Corner mount over the seating area", "crowd",
             ["crowd", "gesture"]),
            (1, "Loading Bay", "Pole mount facing the dock", "abandoned",
             ["object", "theft", "gesture", "geofence"]),
            (1, "Stock Aisle 3", "Aisle-end mount", "restricted",
             ["theft", "object", "gesture"]),
            (1, "Warehouse Floor", "High mount, west wall", "fire",
             ["fire", "crowd", "gesture", "object"]),
        ]
        cameras = []
        for site_index, name, note, scenario, analytics in camera_specs:
            camera, _ = Camera.objects.update_or_create(
                organization=organization, site=sites[site_index], name=name,
                defaults={
                    "location_note": note,
                    "protocol": Camera.Protocol.DEMO,
                    "stream_url": "",
                    "status": Camera.Status.ONLINE,
                    "status_detail": "Simulated stream — replace with your RTSP URL",
                    "last_seen_at": timezone.now(), "last_frame_at": timezone.now(),
                    "resolution_width": 1920, "resolution_height": 1080,
                    "target_fps": 6.0, "enabled_analytics": analytics,
                    "analytics_config": {"demo_scenario": scenario},
                    "record_evidence": True, "is_active": True,
                },
            )
            cameras.append(camera)
        self.stdout.write(f"    {len(sites)} sites, {len(cameras)} cameras")

        # -- zones -----------------------------------------------------
        zone_specs = [
            (1, "Server Room Doorway", Zone.Kind.RESTRICTED, Zone.Severity.CRITICAL,
             [[0.58, 0.30], [0.98, 0.30], [0.98, 0.95], [0.58, 0.95]], 0, 0),
            (0, "Entrance Lobby", Zone.Kind.COUNTING, Zone.Severity.LOW,
             [[0.08, 0.45], [0.92, 0.45], [0.92, 0.98], [0.08, 0.98]], 12, 0),
            (2, "Machine Guard Zone", Zone.Kind.HAZARD, Zone.Severity.CRITICAL,
             [[0.60, 0.35], [0.97, 0.35], [0.97, 0.90], [0.60, 0.90]], 0, 5),
            (3, "Canteen Seating", Zone.Kind.COUNTING, Zone.Severity.INFO,
             [[0.05, 0.40], [0.95, 0.40], [0.95, 0.98], [0.05, 0.98]], 30, 0),
            (4, "Dock Threshold", Zone.Kind.MONITORED, Zone.Severity.MEDIUM,
             [[0.10, 0.55], [0.90, 0.55], [0.90, 0.98], [0.10, 0.98]], 0, 0),
            (5, "High-Value Stock", Zone.Kind.SHELF, Zone.Severity.HIGH,
             [[0.66, 0.22], [0.99, 0.22], [0.99, 0.88], [0.66, 0.88]], 0, 10),
        ]
        for camera_index, name, kind, severity, polygon, occupancy, dwell in zone_specs:
            Zone.objects.update_or_create(
                camera=cameras[camera_index], name=name,
                defaults={
                    "kind": kind, "severity": severity, "polygon": polygon,
                    "max_occupancy": occupancy, "min_dwell_seconds": dwell,
                    "is_active": True,
                    "schedule": ({"days": ["sat", "sun"], "from": "20:00", "to": "07:00"}
                                 if kind == Zone.Kind.RESTRICTED else {}),
                },
            )
        self.stdout.write(f"    {len(zone_specs)} zones")

        # -- employees -------------------------------------------------
        employee_specs = [
            ("EMP-0101", "Arjun Pillai", "Line Supervisor", "Production"),
            ("EMP-0102", "Kavya Menon", "Quality Inspector", "Quality"),
            ("EMP-0103", "Rahul Verma", "Forklift Operator", "Logistics"),
            ("EMP-0104", "Sneha Iyer", "Safety Officer", "EHS"),
            ("EMP-0105", "Vikram Singh", "Security Guard", "Security"),
            ("EMP-0106", "Fatima Sheikh", "Stores Manager", "Logistics"),
            ("EMP-0107", "Joseph Mathew", "Maintenance Technician", "Engineering"),
            ("EMP-0108", "Divya Reddy", "Shift Coordinator", "Production"),
        ]
        employees = []
        for code, name, role, department in employee_specs:
            employee, _ = Employee.objects.update_or_create(
                organization=organization, employee_code=code,
                defaults={
                    "full_name": name, "role": role, "department": department,
                    "email": f"{name.split()[0].lower()}@acme-demo.com",
                    "is_active": True, "consent_given": True,
                    "consent_recorded_at": timezone.now() - timedelta(days=60),
                },
            )
            employee.sites.set(sites)
            employees.append(employee)
        self.stdout.write(f"    {len(employees)} employees")

        # -- alerting --------------------------------------------------
        email_channel, _ = NotificationChannel.objects.update_or_create(
            organization=organization, name="Security team email",
            defaults={
                "kind": NotificationChannel.Kind.EMAIL,
                "config": {"emails": ["security@acme-demo.com", "owner@acme-demo.com"]},
                "include_snapshot": True, "is_active": True,
            },
        )
        NotificationChannel.objects.update_or_create(
            organization=organization, name="Operations webhook",
            defaults={
                "kind": NotificationChannel.Kind.WEBHOOK,
                "config": {"url": "https://example.com/campy-hook", "secret": "demo-secret"},
                "is_active": True,
            },
        )

        critical_rule, _ = AlertRule.objects.update_or_create(
            organization=organization, name="Fire & critical safety → security team",
            defaults={
                "description": "Anything critical reaches a human within seconds, day or night.",
                "min_severity": "high", "min_confidence": 0.55,
                "analytics": ["fire", "object", "geofence"],
                "cooldown_seconds": 180, "escalate_after_seconds": 600,
                "create_incident": True, "is_active": True, "priority": 10,
            },
        )
        critical_rule.channels.set([email_channel])

        after_hours, _ = AlertRule.objects.update_or_create(
            organization=organization, name="After-hours movement",
            defaults={
                "description": "Any medium-or-worse finding outside working hours.",
                "min_severity": "medium", "min_confidence": 0.5,
                "active_schedule": {"days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
                                    "from": "20:00", "to": "07:00"},
                "cooldown_seconds": 300, "create_incident": True, "is_active": True, "priority": 20,
            },
        )
        after_hours.channels.set([email_channel])
        self.stdout.write("    2 alert rules, 2 notification channels")

        # -- events ----------------------------------------------------
        self.generate_events(organization, cameras, employees, event_count)

        # -- a dataset ready to train ----------------------------------
        self.seed_dataset(organization)

        # -- reports ---------------------------------------------------
        from apps.analytics.models import Report

        Report.objects.update_or_create(
            organization=organization, name="Weekly safety summary",
            defaults={
                "kind": Report.Kind.SAFETY, "frequency": Report.Frequency.WEEKLY,
                "date_range_days": 7, "recipients": ["owner@acme-demo.com"], "is_active": True,
            },
        )
        Report.objects.update_or_create(
            organization=organization, name="Monthly compliance export",
            defaults={
                "kind": Report.Kind.COMPLIANCE, "frequency": Report.Frequency.MONTHLY,
                "date_range_days": 30, "recipients": [], "is_active": True,
            },
        )

    def generate_events(self, organization, cameras, employees, count: int):
        """Create a realistic spread of historical events."""
        from apps.events.models import Event
        from apps.events.services import attach_to_incident

        if Event.objects.filter(organization=organization).count() >= count:
            self.stdout.write("    events already present — skipping")
            return

        rng = random.Random(42)
        catalogue = [
            ("geofence", "unauthorised_zone_entry", "high", "Unauthorised entry into {zone}",
             "A person entered the restricted area '{zone}'. No matching authorisation was found."),
            ("geofence", "zone_dwell", "medium", "Extended presence in {zone}",
             "A person has remained inside '{zone}' for longer than the configured limit."),
            ("fire", "smoke_detected", "high", "Smoke detected",
             "A spreading, low-texture grey region consistent with smoke was detected."),
            ("fire", "fire_detected", "critical", "Fire detected",
             "Flame-coloured pixels with fire-like flicker and a growing area were detected. "
             "Verify immediately and trigger your emergency procedure if confirmed."),
            ("crowd", "crowd_formation", "medium", "Crowd forming — {n} people",
             "{n} people have gathered in view. Sustained crowding raises safety risk."),
            ("crowd", "crowd_stagnation", "high", "Congestion — crowd is not moving",
             "A crowd has formed and movement has largely stopped."),
            ("gesture", "running_detected", "medium", "Running in the workplace",
             "Sustained running was observed. This can indicate an emergency, or an unsafe hurry."),
            ("gesture", "prolonged_inactivity", "low", "Prolonged inactivity",
             "A person has been stationary for an extended period. Worth a wellbeing check."),
            ("gesture", "possible_fall", "critical", "Possible fall detected",
             "A person's posture changed abruptly from upright to horizontal. Check on them."),
            ("gesture", "loitering", "medium", "Loitering detected",
             "A person has remained in roughly the same spot for an extended period."),
            ("theft", "suspicious_handling", "high", "Suspicious object handling — review recommended",
             "Behaviour consistent with concealed removal of an item was observed. This is a prompt "
             "to review the footage, not a conclusion about any individual."),
            ("theft", "stock_change", "medium", "Stock change in {zone}",
             "The contents of '{zone}' changed and stayed changed. Reconcile against expected movements."),
            ("object", "unattended_object", "medium", "Unattended object",
             "An object has been left unattended. Unattended items are both a trip hazard and a "
             "security concern."),
            ("object", "dangerous_object", "critical", "Sharp tool detected",
             "A sharp tool was detected outside its designated area."),
            ("face", "employee_present", "info", "{name} identified",
             "{name} was recognised on this camera."),
            ("face", "unknown_person", "medium", "Unrecognised person",
             "A person was observed whose face does not match any enrolled employee."),
        ]

        now = timezone.now()
        created = 0
        with_evidence = []
        # A few real simulator frames per camera, reused across its events, so
        # reports have genuine evidence without seeding a database of JPEGs.
        frame_pool = {camera.pk: self.frames_for(camera, rng) for camera in cameras}
        for _ in range(count):
            analytic, event_type, severity, title, description = rng.choice(catalogue)
            camera = rng.choice([c for c in cameras if analytic in c.enabled_analytics] or cameras)
            zone = camera.zones.order_by("?").first()
            employee = rng.choice(employees) if analytic == "face" and event_type == "employee_present" else (
                rng.choice(employees) if rng.random() < 0.35 else None
            )

            substitutions = {
                "zone": zone.name if zone else "the monitored area",
                "n": rng.randint(9, 22),
                "name": employee.full_name if employee else "An employee",
            }

            # Weight recent days more heavily — that is what real activity looks like.
            days_ago = min(int(abs(rng.gauss(0, 9))), 29)
            hour = rng.choices(
                range(24),
                weights=[1, 1, 1, 1, 2, 4, 7, 12, 16, 14, 12, 11, 14, 12, 11, 12, 14, 13, 9, 6, 4, 3, 2, 1],
            )[0]
            occurred = (now - timedelta(days=days_ago)).replace(
                hour=hour, minute=rng.randint(0, 59), second=rng.randint(0, 59)
            )
            if occurred > now:
                occurred = now - timedelta(minutes=rng.randint(1, 90))

            age_hours = (now - occurred).total_seconds() / 3600
            if age_hours < 3:
                status_value = Event.Status.OPEN
            elif rng.random() < 0.12:
                status_value = Event.Status.FALSE_POSITIVE
            elif rng.random() < 0.18:
                status_value = Event.Status.ACKNOWLEDGED
            else:
                status_value = Event.Status.RESOLVED

            event = Event.objects.create(
                organization=organization, camera=camera, zone=zone, employee=employee,
                analytic=analytic, event_type=event_type, severity=severity,
                status=status_value,
                title=title.format(**substitutions)[:220],
                description=description.format(**substitutions),
                confidence=round(rng.uniform(0.58, 0.97), 4),
                occurred_at=occurred, last_seen_at=occurred,
                occurrence_count=rng.choices([1, 1, 1, 2, 3], weights=[70, 12, 8, 6, 4])[0],
                bounding_box=self.demo_box(rng),
                track_id=rng.randint(1, 60),
                metadata={"seeded": True, "scenario": camera.analytics_config.get("demo_scenario")},
            )
            if status_value in {Event.Status.ACKNOWLEDGED, Event.Status.RESOLVED, Event.Status.FALSE_POSITIVE}:
                event.acknowledged_at = occurred + timedelta(minutes=rng.randint(1, 45))
                if status_value != Event.Status.ACKNOWLEDGED:
                    event.resolved_at = event.acknowledged_at + timedelta(minutes=rng.randint(2, 180))
                event.save(update_fields=["acknowledged_at", "resolved_at"])

            attach_to_incident(event)
            if len(with_evidence) < self.EVIDENCE_EVENTS and severity in {"high", "critical", "medium"}:
                with_evidence.append(event)
            created += 1

        attached = self.attach_evidence(with_evidence, frame_pool)
        self.stdout.write(f"    {created} events across 30 days, "
                          f"{attached} with evidence frames")

    #: Enough stored frames to show the evidence section of a report working,
    #: without seeding a demo database full of JPEGs.
    EVIDENCE_EVENTS = 24

    @staticmethod
    def demo_box(rng):
        """A plausible person-shaped box inside the analysed frame."""
        from django.conf import settings

        width = int(settings.CAMPY["WORKER_FRAME_WIDTH"])
        height = int(settings.CAMPY["WORKER_FRAME_HEIGHT"])
        box_h = rng.uniform(height * 0.35, height * 0.75)
        box_w = box_h * rng.uniform(0.3, 0.5)
        x1 = rng.uniform(0, max(width - box_w, 1))
        y1 = rng.uniform(0, max(height - box_h, 1))
        return [round(x1, 1), round(y1, 1), round(x1 + box_w, 1), round(y1 + box_h, 1)]

    def frames_for(self, camera, rng):
        """A few real simulator frames for this camera, reused across its events."""
        from django.conf import settings

        from apps.aiengine.simulation import SceneSimulator

        width = int(settings.CAMPY["WORKER_FRAME_WIDTH"])
        height = int(settings.CAMPY["WORKER_FRAME_HEIGHT"])
        scenario = (camera.analytics_config or {}).get("demo_scenario", "walk")
        simulator = SceneSimulator(width=width, height=height, seed=camera.pk or 7)
        producer = {
            "crowd": lambda: simulator.crowd(30, people=8),
            "fire": lambda: simulator.fire_outbreak(40, warmup=10),
            "abandoned": lambda: simulator.abandoned_object(40),
            "restricted": lambda: simulator.restricted_entry(30),
            "running": lambda: simulator.running_person(30),
            "idle": lambda: simulator.idle_person(30),
        }.get(scenario, lambda: simulator.walk_across(30))

        frames = []
        for index, frame in enumerate(producer()):
            if index % 8 == 0:
                frames.append(frame)
            if len(frames) >= 4:
                break
        return frames

    def attach_evidence(self, events, frame_pool):
        """Store a frame against each event so reports have something to show."""
        import io
        import random

        from django.core.files.base import ContentFile

        try:
            from PIL import Image
        except ImportError:
            return 0

        from apps.cameras.models import CameraSnapshot
        from apps.events.models import Event

        rng = random.Random(11)
        attached = 0
        for event in events:
            frames = frame_pool.get(event.camera_id)
            if not frames:
                continue
            frame = rng.choice(frames)
            buffer = io.BytesIO()
            Image.fromarray(frame.astype("uint8")).save(buffer, format="JPEG", quality=82)
            snapshot = CameraSnapshot(
                camera=event.camera, captured_at=event.occurred_at,
                width=frame.shape[1], height=frame.shape[0],
                reason=event.event_type,
                annotations={"events": [{"box": event.bounding_box}]},
            )
            snapshot.image.save(f"seed-{event.uid}.jpg", ContentFile(buffer.getvalue()), save=False)
            snapshot.save()
            Event.objects.filter(pk=event.pk).update(snapshot=snapshot)
            attached += 1
        return attached

    def seed_dataset(self, organization):
        """A dataset primed with generated samples so training works immediately."""
        import io

        import numpy as np
        from django.core.files.base import ContentFile

        from apps.training.models import Dataset, DatasetSample

        dataset, created = Dataset.objects.update_or_create(
            organization=organization, name="Fire & smoke detector v1",
            defaults={
                "description": "Bootstrap dataset for the fire/smoke classifier. Replace the "
                               "generated samples with frames from your own cameras.",
                "task": "classification",
                "classes": ["normal", "fire", "smoke"],
                "image_size": 64,
                "validation_split": 0.2,
                "augmentation": {"flip": True, "brightness": 0.2, "noise": 0.03, "shift": 0.08},
                "harvest_from_events": True,
                "harvest_analytics": ["fire"],
            },
        )
        if not created and dataset.sample_count:
            self.stdout.write("    dataset already populated — skipping samples")
            return

        try:
            from PIL import Image
        except ImportError:
            self.stdout.write(self.style.WARNING("    Pillow missing — skipping dataset samples"))
            return

        from apps.aiengine.simulation import synthetic_classification_dataset

        images, labels = synthetic_classification_dataset(
            dataset.classes, samples_per_class=45, size=dataset.image_size, seed=11
        )
        for index, (array, label_index) in enumerate(zip(images, labels)):
            buffer = io.BytesIO()
            Image.fromarray(np.asarray(array, dtype=np.uint8)).save(buffer, format="JPEG", quality=88)
            sample = DatasetSample(
                dataset=dataset,
                label=dataset.classes[int(label_index)],
                labelled_at=timezone.now(),
                notes="Generated bootstrap sample — replace with real footage before production use.",
            )
            sample.image.save(f"seed-{index}.jpg", ContentFile(buffer.getvalue()), save=False)
            sample.save()

        dataset.refresh_counts()
        self.stdout.write(f"    dataset '{dataset.name}' with {dataset.labelled_count} labelled samples")
