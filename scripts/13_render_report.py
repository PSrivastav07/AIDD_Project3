"""Step 13 - render reports/final_report.md to a self-contained HTML file and a PDF.

Images referenced with relative paths are embedded as base64 data URIs, so the HTML can be shared as a
single file. The PDF is printed with a headless Chromium browser (Microsoft Edge or Google Chrome),
which is available on stock Windows; pass --browser to point at another executable.
"""
from __future__ import annotations

import argparse
import base64
import re
import shutil
import subprocess
from pathlib import Path

import markdown

from solpred.config import ROOT

CSS = """
body { font-family: "Segoe UI", Helvetica, Arial, sans-serif; max-width: 980px; margin: 2.2em auto;
       padding: 0 1.2em; line-height: 1.55; color: #1d1d1f; background: #ffffff; font-size: 15px; }
h1 { font-size: 1.9em; border-bottom: 2px solid #4C72B0; padding-bottom: .25em; }
h2 { font-size: 1.45em; margin-top: 1.8em; border-bottom: 1px solid #d0d7de; padding-bottom: .2em; }
h3 { font-size: 1.15em; margin-top: 1.4em; }
table { border-collapse: collapse; margin: 1em 0; font-size: 0.86em; display: block; overflow-x: auto; }
th, td { border: 1px solid #d0d7de; padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: #f3f5f8; }
tr:nth-child(even) td { background: #fafbfc; }
code { background: #f3f5f8; padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }
pre { background: #f3f5f8; padding: .8em; overflow-x: auto; border-radius: 4px; }
img { max-width: 100%; height: auto; display: block; margin: 0.8em auto; }
blockquote { border-left: 4px solid #4C72B0; margin: 1em 0; padding: .2em 1em; color: #444; background: #f7f9fc; }
@media print { body { max-width: none; margin: 0; font-size: 11pt; } h2 { page-break-after: avoid; }
               img, table { page-break-inside: avoid; } }
"""


def embed_images(html: str, base: Path) -> str:
    def repl(m: re.Match) -> str:
        src = m.group(2)
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        f = (base / src).resolve()
        if not f.exists():
            print(f"warning: missing image {src}")
            return m.group(0)
        mime = "image/png" if f.suffix.lower() == ".png" else "image/svg+xml" if f.suffix.lower() == ".svg" else "image/jpeg"
        data = base64.b64encode(f.read_bytes()).decode()
        return f'{m.group(1)}data:{mime};base64,{data}{m.group(3)}'

    return re.sub(r'(<img[^>]*src=")([^"]+)(")', repl, html)


def find_browser(explicit: str | None) -> str | None:
    cands = [explicit] if explicit else [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        shutil.which("msedge") or "", shutil.which("google-chrome") or "", shutil.which("chromium") or ""]
    for c in cands:
        if c and Path(c).exists():
            return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="reports/final_report.md")
    ap.add_argument("--browser", default=None)
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    src = ROOT / args.input
    text = src.read_text(encoding="utf-8")
    title = next((ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("# ")), "Report")
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"])
    body = embed_images(body, src.parent)
    html = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'><title>{title}</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    out_html = src.with_suffix(".html")
    out_html.write_text(html, encoding="utf-8")
    print(f"wrote {out_html.relative_to(ROOT)} ({out_html.stat().st_size / 1e6:.1f} MB)")
    if args.no_pdf:
        return
    browser = find_browser(args.browser)
    if not browser:
        print("no Chromium-based browser found; skipping PDF")
        return
    out_pdf = src.with_suffix(".pdf")
    cmd = [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
           f"--print-to-pdf={out_pdf}", out_html.resolve().as_uri()]
    subprocess.run(cmd, check=False, timeout=180, capture_output=True)
    print(f"wrote {out_pdf.relative_to(ROOT)}" if out_pdf.exists() else "PDF printing failed")


if __name__ == "__main__":
    main()
