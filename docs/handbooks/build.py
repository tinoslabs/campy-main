"""Build the two handbooks as PDFs.

    pip install playwright pymupdf requests && playwright install chromium
    python docs/handbooks/build.py

Each book is rendered twice — a full-bleed cover with no margins or running
footer, then the body with both — and the two are merged, because Chromium
applies one set of print margins to a whole document.

Web fonts are downloaded and inlined as data URIs on the first run
(``fonts.css``, gitignored). The HTML also links Google Fonts directly, so the
sources still look right opened in a browser without building anything.
"""
from __future__ import annotations

import base64
import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent

BOOKS = [
    ("technical.html", "Campy-AI-Technical-Handbook.pdf", "Technical Handbook",
     "Campy AI — Technical Handbook",
     "Connecting cameras and what happens to every frame: topology, protocols, "
     "the per-frame pipeline, capacity planning and operations."),
    ("plain-english.html", "Campy-AI-Plain-English-Handbook.pdf", "Plain English Handbook",
     "Campy AI — Plain English Handbook",
     "How your existing CCTV cameras start paying attention, explained without jargon."),
]

FAMILIES = {
    "Archivo": "family=Archivo:wght@500;600;700",
    "JetBrains Mono": "family=JetBrains+Mono:wght@400;500;700",
    "Source Serif 4":
        "family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400",
}
BROWSER_UA = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120 Safari/537.36"
}

FOOTER = """
<div style="width:100%;font-size:7pt;font-family:'JetBrains Mono',monospace;
            color:#7b8f8c;padding:0 18mm;display:flex;justify-content:space-between;
            letter-spacing:.06em;-webkit-print-color-adjust:exact;">
  <span>CAMPY AI &middot; __TITLE__</span><span class="pageNumber"></span>
</div>"""
NO_HEADER = '<div style="display:none"></div>'
PRINT_MARGIN = {"top": "20mm", "bottom": "16mm", "left": "18mm", "right": "18mm"}


def embed_fonts(target: pathlib.Path) -> None:
    """Inline the Latin faces so the PDF does not depend on a network at print time."""
    if target.exists():
        print(f"  fonts    {target.name} (cached)")
        return
    import requests

    faces = []
    for name, query in FAMILIES.items():
        css = requests.get(f"https://fonts.googleapis.com/css2?{query}&display=swap",
                           headers=BROWSER_UA, timeout=30).text
        blocks = re.findall(r"/\* (latin[^*]*) \*/\s*(@font-face \{.*?\})", css, re.S)
        kept = 0
        for label, block in blocks or [("all", b) for b in
                                       re.findall(r"(@font-face \{.*?\})", css, re.S)]:
            if "latin-ext" in label:
                continue
            url = re.search(r"url\((https://[^)]+)\)", block)
            if not url:
                continue
            payload = requests.get(url.group(1), headers=BROWSER_UA, timeout=30).content
            uri = "data:font/woff2;base64," + base64.b64encode(payload).decode()
            faces.append(re.sub(r"url\(https://[^)]+\)", f"url({uri})", block))
            kept += 1
        print(f"  fonts    {name}: {kept} faces")
    target.write_text("\n".join(faces))


def bookmarks(document) -> list[list]:
    """One entry per chapter opening page, titled by its heading.

    The chapter eyebrow is letter-spaced, so it extracts as "C H A P T E R 1";
    squashing whitespace is what makes it matchable.
    """
    entries = [[1, "Cover", 1], [1, "Contents", 2]]
    for number, page in enumerate(document, start=1):
        lines = [line.strip() for line in page.get_text("text").splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        squashed = "".join(lines[0].split()).upper()
        if squashed.startswith("CHAPTER"):
            entries.append([1, f"{squashed[len('CHAPTER'):] or '?'}. {lines[1]}"[:80], number])
    return entries


def render(page, source: pathlib.Path, body_class: str, out: pathlib.Path, footer=None) -> None:
    page.goto(source.as_uri(), wait_until="load", timeout=60000)
    page.evaluate("document.fonts.ready")
    page.evaluate("cls => document.body.className = cls", body_class)
    if body_class == "only-cover":
        # prefer_css_page_size honours the @page margins too, which would inset
        # the full-bleed cover. A later rule of equal specificity wins.
        page.evaluate("() => { const s = document.createElement('style');"
                      "        s.textContent = '@page { size: A4; margin: 0 }';"
                      "        document.head.appendChild(s); }")
    page.wait_for_timeout(900)
    page.emulate_media(media="print")
    options = {"path": str(out), "format": "A4", "print_background": True,
               "prefer_css_page_size": True}
    if footer:
        options.update(display_header_footer=True, header_template=NO_HEADER,
                       footer_template=footer, margin=PRINT_MARGIN)
    else:
        options["margin"] = {"top": "0", "bottom": "0", "left": "0", "right": "0"}
    page.pdf(**options)


def main() -> int:
    try:
        import pymupdf
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        print(f"missing dependency: {exc}\n"
              "  pip install playwright pymupdf requests && playwright install chromium")
        return 1

    embed_fonts(HERE / "fonts.css")
    launch = {}
    if os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE"):
        launch["executable_path"] = os.environ["PLAYWRIGHT_CHROMIUM_EXECUTABLE"]

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(**launch)
        for source, dest, short, title, subject in BOOKS:
            page = browser.new_page()
            cover, body = HERE / "_cover.pdf", HERE / "_body.pdf"
            render(page, HERE / source, "only-cover", cover)
            render(page, HERE / source, "no-cover", body,
                   FOOTER.replace("__TITLE__", short.upper()))
            page.close()

            merged = pymupdf.open()
            with pymupdf.open(cover) as opened:
                merged.insert_pdf(opened, to_page=0)     # the cover is always one page
            with pymupdf.open(body) as opened:
                merged.insert_pdf(opened)
            merged.set_metadata({"title": title, "author": "Campy AI", "subject": subject,
                                 "keywords": "CCTV, video analytics, ONVIF, RTSP, safety"})
            merged.set_toc(bookmarks(merged))
            merged.save(HERE / dest, garbage=4, deflate=True)
            print(f"  built    {dest}  {merged.page_count} pages  "
                  f"{(HERE / dest).stat().st_size / 1024:.0f} KB")
            merged.close()
            cover.unlink()
            body.unlink()
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
