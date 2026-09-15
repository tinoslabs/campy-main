"""Analysis reports as PDF, with the frames that justify them.

Findings are only worth as much as the evidence behind them. A safety meeting,
an insurance claim or a disciplinary conversation all turn on being able to put
the moment itself on the table — so every report here carries the frame, with
the detection outlined and the zone it crossed drawn on top.

Built on ReportLab, which ships as a pure-Python wheel: a report must generate
on the customer's own server, including the Windows machines these often run
on, without asking anyone to install system libraries.
"""
from __future__ import annotations

import io
from datetime import timedelta

from django.utils import timezone
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from apps.aiengine.detectors import ANALYTIC_CHOICES
from apps.analytics.evidence import annotate_event
from apps.events.models import Event

ANALYTIC_LABELS = dict(ANALYTIC_CHOICES)

#: What each detector is actually looking for, in the language of the report's
#: reader rather than the engineer's. A finding nobody can interpret is noise.
ANALYTIC_NOTES = {
    "gesture": "Posture and activity — walking, standing, sitting, bending, running, "
               "prolonged inactivity and falls. Speeds are measured in body-heights "
               "per second, so a threshold holds at any camera distance.",
    "face": "Recognises employees who have been enrolled with recorded consent, so a "
            "finding can say who rather than somebody.",
    "geofence": "Entry into an area marked restricted, hazardous or staff-only. "
                "Containment is tested at a person's feet, not their centre.",
    "crowd": "How many people are in an area, and whether the flow through it is "
             "orderly or beginning to compress.",
    "theft": "Behaviour around stock — shelf dwell, concealment gestures, carry "
             "changes. This highlights moments worth reviewing; it does not "
             "conclude that a theft occurred.",
    "object": "Items that should not be where they are: a tool in a walkway, a bag "
              "left with nobody near it, an obstruction in an escape route.",
    "fire": "Flame colour, the way fire flickers, and growth over time, confirmed by "
            "a trained classifier. Early warning that complements a fire alarm; it "
            "is not a certified replacement for one.",
}

SEVERITY_COLOURS = {
    "critical": colors.HexColor("#a32418"),
    "high": colors.HexColor("#a32418"),
    "medium": colors.HexColor("#a8500a"),
    "low": colors.HexColor("#0a6d64"),
    "info": colors.HexColor("#4d625f"),
}
INK = colors.HexColor("#15201f")
MUTED = colors.HexColor("#4d625f")
RULE = colors.HexColor("#d2dedc")
PANEL = colors.HexColor("#f4f8f7")
ACCENT = colors.HexColor("#0a6d64")


def _styles():
    sheet = getSampleStyleSheet()

    def style(name, *, size, leading, bold=False, colour=INK, before=0, after=5):
        return ParagraphStyle(
            name, parent=sheet["BodyText"], alignment=TA_LEFT,
            fontName="Helvetica-Bold" if bold else "Helvetica",
            fontSize=size, leading=leading, textColor=colour,
            spaceBefore=before, spaceAfter=after,
        )

    return {
        "title": style("t", size=22, leading=26, bold=True, after=2),
        "subtitle": style("st", size=11, leading=15, colour=MUTED, after=10),
        "eyebrow": style("e", size=7.5, leading=10, colour=ACCENT, after=3),
        "h2": style("h2", size=13, leading=17, bold=True, before=12, after=5),
        "h3": style("h3", size=10.5, leading=14, bold=True, before=8, after=3),
        "body": style("b", size=9.5, leading=13.5),
        "note": style("n", size=8.5, leading=12, colour=MUTED, after=4),
        "caption": style("c", size=8, leading=11, colour=MUTED, before=3, after=10),
    }


WRAP_STYLE = ParagraphStyle("cell", fontName="Helvetica", fontSize=8.5, leading=11.5,
                            textColor=INK)
WRAP_HEADER = ParagraphStyle("cellhead", parent=WRAP_STYLE, fontName="Helvetica-Bold",
                             fontSize=7.5, leading=10, textColor=MUTED)


def _wrap(value, style):
    """Long text has to be a Paragraph; ReportLab will not wrap a bare string."""
    if isinstance(value, str) and len(value) > 34:
        return Paragraph(value, style)
    return value


def _table(data, widths, *, align_right=(), header=True):
    rows = []
    for index, row in enumerate(data):
        style = WRAP_HEADER if (header and index == 0) else WRAP_STYLE
        rows.append([_wrap(cell, style) for cell in row])
    table = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE),
    ]
    if header:
        style += [
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.5),
            ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
            ("BACKGROUND", (0, 0), (-1, 0), PANEL),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
        ]
    for column in align_right:
        style.append(("ALIGN", (column, 0), (column, -1), "RIGHT"))
    table.setStyle(TableStyle(style))
    return table


class ReportCanvas:
    """Page furniture: who this belongs to, and which page you are holding."""

    def __init__(self, organization_name: str):
        self.organization_name = organization_name

    def __call__(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 12 * mm, f"{self.organization_name} · Campy AI")
        canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, f"Page {doc.page}")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.4)
        canvas.line(18 * mm, 16 * mm, A4[0] - 18 * mm, 16 * mm)
        canvas.restoreState()


def build_analysis_report(
    organization,
    *,
    analytics: list[str] | None = None,
    days: int = 7,
    cameras=None,
    sites=None,
    evidence_limit: int = 12,
    generated_by: str = "",
    title: str = "",
) -> bytes:
    """Render an analysis report. Returns the PDF as bytes.

    One analytic gives a report about that feature; several give a combined one
    with a comparison across them. The shape is otherwise the same, so a reader
    who knows one knows both.
    """
    from django.db.models import Count, Q

    end = timezone.now()
    start = end - timedelta(days=max(int(days or 7), 1))
    keys = [k for k in (analytics or []) if k in ANALYTIC_LABELS] or list(ANALYTIC_LABELS)

    events = (Event.objects.filter(organization=organization, occurred_at__range=(start, end),
                                   analytic__in=keys)
              .select_related("camera", "zone", "employee", "snapshot"))
    if cameras:
        events = events.filter(camera__in=cameras)
    elif sites:
        events = events.filter(camera__site__in=sites)

    combined = len(keys) > 1
    styles = _styles()
    story = []

    scope = ", ".join(ANALYTIC_LABELS[k] for k in keys)
    heading = title or (
        "Combined analysis report" if combined else f"{ANALYTIC_LABELS[keys[0]]} report"
    )

    # -- cover -------------------------------------------------------------
    story.append(Paragraph("CAMPY AI · ANALYSIS REPORT", styles["eyebrow"]))
    story.append(Paragraph(heading, styles["title"]))
    story.append(Paragraph(organization.name, styles["subtitle"]))

    where = "All cameras"
    if cameras:
        names = [c.name for c in cameras]
        where = ", ".join(names[:4]) + (f" and {len(names) - 4} more" if len(names) > 4 else "")
    elif sites:
        where = ", ".join(s.name for s in sites)

    story.append(_table([
        ["Period", f"{timezone.localtime(start):%d %b %Y, %H:%M} — "
                   f"{timezone.localtime(end):%d %b %Y, %H:%M}  ({days} days)"],
        ["Scope", scope],
        ["Cameras", where],
        ["Produced", f"{timezone.localtime(end):%d %b %Y, %H:%M}"
                     + (f" by {generated_by}" if generated_by else "")],
    ], [30 * mm, 140 * mm], header=False))
    story.append(Spacer(1, 6 * mm))

    # -- what was found ----------------------------------------------------
    total = events.count()
    attention = events.filter(severity__in=["high", "critical"]).count()
    resolved = events.filter(status=Event.Status.RESOLVED).count()
    false_alarms = events.filter(status=Event.Status.FALSE_POSITIVE).count()
    reporting = events.values("camera").distinct().count()

    story.append(Paragraph("What was found", styles["h2"]))
    if not total:
        story.append(Paragraph(
            "Nothing was detected in this period for the selected cameras and features. "
            "An empty report is a result: it says the cameras were watched and had "
            "nothing to report, which is not the same as nobody having looked.",
            styles["body"]))
    else:
        story.append(_table([
            ["Findings", "Needed attention", "Resolved", "Called a false alarm", "Cameras reporting"],
            [f"{total}", f"{attention}", f"{resolved}",
             f"{false_alarms} ({false_alarms / total * 100:.0f}%)", f"{reporting}"],
        ], [34 * mm, 34 * mm, 28 * mm, 40 * mm, 34 * mm], align_right=(0, 1, 2, 3, 4)))

    # -- comparison across features ---------------------------------------
    if combined and total:
        story.append(Paragraph("By feature", styles["h2"]))
        rows = [["Feature", "Findings", "Needed attention", "False alarms"]]
        for row in (events.values("analytic")
                    .annotate(n=Count("id"),
                              urgent=Count("id", filter=Q(severity__in=["high", "critical"])),
                              wrong=Count("id", filter=Q(status=Event.Status.FALSE_POSITIVE)))
                    .order_by("-n")):
            rows.append([ANALYTIC_LABELS.get(row["analytic"], row["analytic"]),
                         str(row["n"]), str(row["urgent"]), str(row["wrong"])])
        story.append(_table(rows, [70 * mm, 28 * mm, 36 * mm, 32 * mm], align_right=(1, 2, 3)))

    # -- a section per feature --------------------------------------------
    for key in keys:
        subset = events.filter(analytic=key)
        if combined and not subset.exists():
            continue
        story.append(Paragraph(ANALYTIC_LABELS[key], styles["h2"]))
        story.append(Paragraph(ANALYTIC_NOTES.get(key, ""), styles["note"]))

        count = subset.count()
        if not count:
            story.append(Paragraph("No findings in this period.", styles["body"]))
            continue

        by_camera = (subset.values("camera__name").annotate(n=Count("id")).order_by("-n")[:6])
        rows = [["Camera", "Findings"]] + [[r["camera__name"] or "—", str(r["n"])] for r in by_camera]
        story.append(_table(rows, [100 * mm, 30 * mm], align_right=(1,)))

        by_type = (subset.values("event_type").annotate(n=Count("id")).order_by("-n")[:6])
        if by_type:
            rows = [["What happened", "Times"]] + [
                [humanise(r["event_type"]), str(r["n"])] for r in by_type
            ]
            story.append(Spacer(1, 3 * mm))
            story.append(_table(rows, [100 * mm, 30 * mm], align_right=(1,)))

    # -- evidence ----------------------------------------------------------
    if total and evidence_limit:
        story.append(PageBreak())
        story.append(Paragraph("Evidence", styles["h2"]))
        story.append(Paragraph(
            "The frame that produced each finding, with the detection outlined and, "
            "where one applies, the zone it crossed. Most serious first.",
            styles["note"]))

        ordered = sorted(
            events.exclude(snapshot__isnull=True)[: evidence_limit * 3],
            key=lambda e: (-_severity_rank(e.severity), -e.occurred_at.timestamp()),
        )[:evidence_limit]

        shown = 0
        for event in ordered:
            picture = annotate_event(event)
            if picture is None:
                continue
            block = [_evidence_image(picture), _evidence_caption(event, styles)]
            story.append(KeepTogether(block))
            shown += 1

        if not shown:
            story.append(Paragraph(
                "No frames were stored for these findings. Evidence capture is set per "
                "camera, and stored frames are deleted on the retention schedule.",
                styles["body"]))

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "Findings are produced by automated analysis and are prompts for a person to "
        "review, not conclusions. Theft-related findings describe behaviour, never "
        "intent. This document may contain images of identifiable people — handle it "
        "under your own data-protection policy.",
        styles["note"]))

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=22 * mm,
        title=f"{heading} — {organization.name}", author="Campy AI",
        subject=f"{scope}; {timezone.localtime(start):%d %b %Y} to {timezone.localtime(end):%d %b %Y}",
    )
    furniture = ReportCanvas(organization.name)
    document.build(story, onFirstPage=furniture, onLaterPages=furniture)
    return buffer.getvalue()


def humanise(event_type: str) -> str:
    """`fire_detected` is a database value; a report is read by people."""
    return str(event_type or "detection").replace("_", " ").strip().capitalize()


def _severity_rank(severity: str) -> int:
    return {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(severity, 0)


def _evidence_image(jpeg: bytes) -> RLImage:
    from PIL import Image

    with Image.open(io.BytesIO(jpeg)) as probe:
        width, height = probe.size
    display_width = 120 * mm
    return RLImage(io.BytesIO(jpeg), width=display_width,
                   height=display_width * height / max(width, 1))


def _evidence_caption(event, styles) -> Paragraph:
    when = timezone.localtime(event.occurred_at)
    parts = [f"<b>{event.title}</b>", f"{event.camera.name}"]
    if event.zone_id:
        parts.append(event.zone.name)
    if event.employee_id:
        parts.append(f"identified as {event.employee.display_name}")
    parts.append(f"{when:%d %b %Y, %H:%M:%S}")
    parts.append(f"{ANALYTIC_LABELS.get(event.analytic, event.analytic)}")
    parts.append(f"confidence {event.confidence * 100:.0f}%")
    if event.occurrence_count > 1:
        parts.append(f"seen {event.occurrence_count} times")
    parts.append(event.get_status_display())
    return Paragraph(" · ".join(parts), styles["caption"])
