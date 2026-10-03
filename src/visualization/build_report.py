"""Build the report PDFs: Markdown -> self-contained HTML (pandoc) -> PDF (headless Edge / Chrome)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from src.utils import get_logger, setup_logging

log = get_logger(__name__)

BROWSERS = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "msedge", "google-chrome", "chromium", "chromium-browser",
)


def find_browser() -> str:
    """First available Chromium-based browser (used only for its print-to-PDF)."""
    for b in BROWSERS:
        if Path(b).exists() or shutil.which(b):
            return b
    raise FileNotFoundError("No Edge/Chrome/Chromium found for PDF printing")


def build(md: Path, css: Path, out_pdf: Path, title: str) -> Path:
    """Convert one Markdown report to PDF; figures are resolved relative to the Markdown file."""
    html = out_pdf.with_suffix(".html")
    try:
        subprocess.run(
            ["pandoc", str(md), "-s", "--embed-resources", "--css", str(css), "--resource-path", str(md.parent),
             "--metadata", f"pagetitle={title}", "-o", str(html)],
            check=True, capture_output=True, text=True,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError("pandoc not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"pandoc failed for {md}: {exc.stderr}") from exc

    with tempfile.TemporaryDirectory() as profile:
        subprocess.run(
            [find_browser(), "--headless=new", "--disable-gpu", f"--user-data-dir={profile}",
             "--no-pdf-header-footer", f"--print-to-pdf={out_pdf.resolve()}", html.resolve().as_uri()],
            check=True, capture_output=True, text=True, timeout=300,
        )
    if not out_pdf.exists() or out_pdf.stat().st_size < 10_000:
        raise RuntimeError(f"PDF was not written: {out_pdf}")
    log.info("%s -> %s (%.0f KB)", md, out_pdf, out_pdf.stat().st_size / 1024)
    return out_pdf


def main() -> None:
    """CLI: build reports/report.pdf (English) and reports/report_es.pdf (Spanish) when present."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    setup_logging("logs", "build_report", "INFO")
    root = Path(args.reports)
    css = root / "report.css"
    for name, title in (("report", "SliceGAN with Vision Transformer discriminators"),
                        ("report_es", "SliceGAN con discriminadores Vision Transformer")):
        md = root / f"{name}.md"
        if md.exists():
            build(md, css, root / f"{name}.pdf", title)


if __name__ == "__main__":
    main()
