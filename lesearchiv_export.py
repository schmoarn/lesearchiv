#!/usr/bin/env python3
"""
Lesearchiv → Obsidian / OneNote Exporter
==========================================
Benötigt:
    pip install anthropic --no-deps
    pip install httpx anyio sniffio typing-extensions
    pkg install poppler  (Termux)  oder  choco install poppler  (Windows)

Windows poppler alternativ: https://github.com/oschwartz10612/poppler-windows/releases
"""

import os, sys, json, base64, time, re, subprocess, tempfile, shutil
from pathlib import Path
from datetime import datetime
from collections import Counter

try:
    import anthropic
except ImportError:
    print("Fehlt: pip install anthropic --no-deps && pip install httpx anyio sniffio typing-extensions")
    sys.exit(1)

# ── Debug-Modus ───────────────────────────────────────────────────────────
DEBUG = "--debug" in sys.argv

def dprint(msg):
    if DEBUG:
        print(f"  [DEBUG] {msg}")

# ── Konfiguration ──────────────────────────────────────────────────────────
RENDER_DPI    = 150
SKETCH_DPI    = 200
PAUSE_BETWEEN = 2
MAX_RETRIES   = 3

PROMPT = """Du analysierst eine handgeschriebene Notizbuchseite eines österreichischen Lesers.
Themen: Geschichte, Medizingeschichte, Kulturgeschichte, Entdeckungsreisen, Literatur.
Antworte NUR mit gültigem JSON (kein Markdown, keine Erklärungen):
{
  "transkription": "vollständiger transkribierter Text der Seite",
  "jahreszahlen": [1492, 1607],
  "tags": ["Kolumbus", "Entdeckung"],
  "epoche": "15. Jahrhundert",
  "seitenzahlen": "S.12, S.16",
  "ist_skizze": false,
  "ist_titelseite": false,
  "kurzzusammenfassung": "Ein Satz Zusammenfassung"
}
Regeln:
- transkription: ganzer lesbarer Text, Zeilenumbrüche mit \\n erhalten
- jahreszahlen: alle Jahreszahlen als Integer-Array, leer wenn keine
- tags: 3-6 deutsche thematische Schlagwörter
- epoche: Hauptepoche (z.B. "15. Jahrhundert", "Frühe Neuzeit")
- seitenzahlen: im Buch referenzierte Seiten (z.B. "S.12, S.16")
- ist_skizze: true wenn Seite hauptsächlich Karte/Zeichnung/Skizze
- ist_titelseite: true wenn erste Seite mit Buchinformation
- kurzzusammenfassung: ein einziger deutscher Satz
- transkription: Zeilenumbrüche innerhalb eines Seitenzahl-Eintrags zu Fließtext zusammenführen. Nur echte Absätze (neuer Seitenzahl-Eintrag oder expliziter Absatz im Original) als \n\n trennen. Kein harter Zeilenumbruch nach jeder Notizbuchzeile.
"""

# ── Hilfsfunktionen ────────────────────────────────────────────────────────

def safe_filename(s):
    return re.sub(r'[\\/*?:"<>|]', '', s).strip()

def parse_pages(s, total):
    if not s.strip():
        return list(range(1, total + 1))
    result = set()
    for part in s.split(","):
        m = re.match(r'^(\d+)-(\d+)$', part.strip())
        if m:
            result.update(range(int(m.group(1)), int(m.group(2)) + 1))
        elif part.strip().isdigit():
            result.add(int(part.strip()))
    return sorted(p for p in result if 1 <= p <= total)

# Verfügbare Backends ermitteln
try:
    import fitz  # PyMuPDF
    HAVE_PYMUPDF = True
    dprint("Backend: PyMuPDF")
except ImportError:
    HAVE_PYMUPDF = False
    dprint("Backend: pdftoppm (PyMuPDF nicht gefunden)")


def get_page_count(pdf_path):
    # Versuch 1: PyMuPDF
    if HAVE_PYMUPDF:
        try:
            doc = fitz.open(str(pdf_path))
            count = doc.page_count
            doc.close()
            return count
        except Exception as e:
            dprint(f"PyMuPDF Seitenanzahl fehlgeschlagen: {e}")
    # Versuch 2: pdfinfo
    try:
        out = subprocess.check_output(
            ["pdfinfo", str(pdf_path)], stderr=subprocess.DEVNULL
        ).decode()
        for line in out.splitlines():
            if line.startswith("Pages:"):
                return int(line.split(":")[1].strip())
    except Exception as e:
        dprint(f"pdfinfo fehlgeschlagen: {e}")
    return None


def render_page_jpeg(pdf_path, page_num, dpi=150):
    # Versuch 1: PyMuPDF
    if HAVE_PYMUPDF:
        try:
            doc = fitz.open(str(pdf_path))
            page = doc[page_num - 1]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            jpeg_bytes = pix.tobytes("jpeg")
            doc.close()
            return jpeg_bytes
        except Exception as e:
            dprint(f"PyMuPDF Rendering fehlgeschlagen: {e}")
    # Versuch 2: pdftoppm
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "pg")
        subprocess.run([
            "pdftoppm", "-jpeg", "-r", str(dpi),
            "-f", str(page_num), "-l", str(page_num),
            str(pdf_path), prefix
        ], check=True, stderr=subprocess.DEVNULL)
        files = sorted(Path(tmpdir).glob("*.jpg"))
        if not files:
            raise RuntimeError(f"Kein Bild für Seite {page_num}")
        return files[0].read_bytes()

def transcribe_page(client, jpeg_bytes, page_num):
    b64 = base64.standard_b64encode(jpeg_bytes).decode()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64
                        }},
                        {"type": "text", "text": PROMPT}
                    ]
                }]
            )
            raw = response.content[0].text.strip()
            dprint(f"Raw API response Seite {page_num}:\n{raw[:500]}")
            raw = re.sub(r'^```json\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"\n    ⚠ JSON-Fehler Seite {page_num}, Versuch {attempt}: {e}")
            if DEBUG:
                print(f"  [DEBUG] Roher Text der fehlschlug:\n---\n{raw}\n---")
            if attempt == MAX_RETRIES:
                return None
            time.sleep(3)
        except Exception as e:
            print(f"\n    ⚠ API-Fehler Seite {page_num}, Versuch {attempt}: {e}")
            if DEBUG:
                import traceback
                traceback.print_exc()
            if attempt == MAX_RETRIES:
                return None
            time.sleep(5 * attempt)
    return None


KORREKTUR_PROMPT = """Du bekommst eine rohe Transkription handgeschriebener Lesenotizen eines österreichischen Lesers.
Die Notizen wurden per KI aus Handschrift erkannt und enthalten daher moeglicherweise:
- Falsch erkannte Woerter durch schwer lesbare Handschrift
- Grammatikalische Fehler
- Unvollstaendige Saetze
- Fehlende Satzzeichen

Deine Aufgabe:
1. Korrigiere offensichtliche Erkennungsfehler zu grammatikalisch korrektem Deutsch
2. Behalte den inhaltlichen Stil und die persoenliche Ausdrucksweise des Autors bei
3. Erfinde KEINEN Inhalt - wenn etwas unleserlich ist, markiere es mit [?]
4. Behalte Seitenzahlen (z.B. "S.12", "S.165") exakt so wie sie sind
5. Behalte Eigennamen, Jahreszahlen und Fachbegriffe exakt wie im Original
6. Kuerze NICHT - gib den vollstaendigen korrigierten Text zurueck
7. Fuehre Zeilenumbrueche innerhalb eines Seitenzahl-Eintrags zu Fliestext zusammen. Nur bei neuem Seitenzahl-Eintrag (z.B. "S.12", "S.165") oder bei einem expliziten Absatz im Original einen doppelten Zeilenumbruch setzen. Kein harter Zeilenumbruch nach jeder Notizbuchzeile.

Antworte NUR mit dem korrigierten Text, ohne Erklaerungen oder Kommentare."""


def correct_text(client, raw_text, page_num, book="", author=""):
    """Zweiter API-Call: grammatikalische Korrektur der Transkription."""
    if not raw_text or not raw_text.strip():
        return raw_text
    kontext = ""
    if book or author:
        kontext = 'Kontext: Dies sind handgeschriebene Lesenotizen zu "' + book + '"'
        if author:
            kontext += f" von {author}"
        kontext += ". Nutze dieses Wissen um Eigennamen, Orte und Fachbegriffe korrekt zu schreiben.\n\n"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": KORREKTUR_PROMPT + "\n\n" + kontext + "Rohe Transkription:\n" + raw_text
                }]
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"\n    Korrektur-Fehler Seite {page_num}, Versuch {attempt}: {e}")
            if attempt == MAX_RETRIES:
                return raw_text
            time.sleep(3 * attempt)
    return raw_text


# __ Obsidian Export ────────────────────────────────────────────────────────

def export_obsidian(book, author, read_date, out_dir, pages_data, all_years, all_tags):
    """
    Erzeugt eine einzige .md pro Buch mit allen Notizen.
    Skizzen kommen als JPGs in einen _Bilder/ Unterordner.
    """
    bilder_dir = out_dir / "_Bilder"
    md_filename = safe_filename(book) + ".md"
    md_path = out_dir / md_filename

    lines = []

    # YAML Frontmatter
    lines.append("---")
    lines.append(f'buch: "{book}"')
    if author:
        lines.append(f'autor: "{author}"')
    lines.append(f'gelesen: "{read_date}"')
    lines.append(f'erfasst: "{datetime.now().strftime("%Y-%m-%d")}"')
    if all_years:
        lines.append(f'jahreszahlen: {json.dumps(sorted(all_years))}')
        lines.append(f'zeitspanne: "{min(all_years)} – {max(all_years)}"')
    if all_tags:
        top = [t for t, _ in all_tags.most_common(12)]
        lines.append(f'tags: [{", ".join(chr(34)+t+chr(34) for t in top)}]')
    lines.append("---")
    lines.append("")

    # Kopfzeile
    lines.append(f"# {book}")
    lines.append("")
    meta_parts = []
    if author:
        meta_parts.append(f"**Autor:** {author}")
    meta_parts.append(f"**Gelesen:** {read_date}")
    if all_years:
        meta_parts.append(f"**Zeitspanne:** {min(all_years)} – {max(all_years)}")
    lines.append("  \n".join(meta_parts))
    lines.append("")
    lines.append("---")
    lines.append("")

    # Notizen
    for entry in pages_data:
        result     = entry["result"]
        pn         = entry["pdf_page"]
        jpeg_bytes = entry["jpeg"]
        is_sketch  = result.get("ist_skizze", False)
        is_title   = result.get("ist_titelseite", False)
        ref_pages  = result.get("seitenzahlen", "").strip()
        years      = result.get("jahreszahlen") or []
        tags       = result.get("tags") or []
        summary    = result.get("kurzzusammenfassung", "").strip()
        text       = result.get("transkription", "").strip()
        epoche     = result.get("epoche", "")

        # Überschrift
        if is_title:
            heading = "Titelseite"
        elif ref_pages:
            heading = ref_pages
        else:
            heading = f"Notizseite {pn}"

        sketch_label = " 🗺" if is_sketch else ""
        lines.append(f"## {heading}{sketch_label}")
        lines.append("")

        # Meta-Zeile
        meta = []
        if epoche:
            meta.append(f"*{epoche}*")
        if years:
            meta.append(" · ".join(f"**{y}**" for y in years))
        if tags:
            meta.append(" ".join(f"#{t.replace(' ','_')}" for t in tags))
        if meta:
            lines.append("  \n".join(meta))
            lines.append("")

        # Zusammenfassung
        if summary:
            lines.append(f"> {summary}")
            lines.append("")

        # Text
        if text:
            lines.append(text)
            lines.append("")

        # Skizze speichern und einbetten
        if is_sketch and jpeg_bytes:
            bilder_dir.mkdir(exist_ok=True)
            sketch_name = f"{safe_filename(book)}_S{pn:03d}.jpg"
            sketch_path = bilder_dir / sketch_name
            # Hochauflösend speichern
            try:
                hq_jpeg = render_page_jpeg_cached(entry.get("pdf_path"), pn, SKETCH_DPI)
                sketch_path.write_bytes(hq_jpeg)
            except Exception:
                sketch_path.write_bytes(jpeg_bytes)
            lines.append(f"![[_Bilder/{sketch_name}]]")
            lines.append("")

        lines.append("---")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path

_render_cache = {}
def render_page_jpeg_cached(pdf_path, page_num, dpi):
    key = (str(pdf_path), page_num, dpi)
    if key not in _render_cache:
        _render_cache[key] = render_page_jpeg(pdf_path, page_num, dpi)
    return _render_cache[key]

# ── OneNote Export ─────────────────────────────────────────────────────────

def export_onenote(book, author, read_date, out_dir, pages_data, all_years, all_tags):
    """
    Erzeugt eine einzelne HTML-Datei die OneNote direkt importieren kann.
    Skizzen werden als Base64 eingebettet – kein separater Bildordner nötig.
    Importieren: OneNote öffnen → Datei → Dokument drucken / Als Anhang einfügen
    Oder: Datei im Browser öffnen → alles auswählen → in OneNote einfügen.
    """
    html_filename = safe_filename(book) + ".html"
    html_path = out_dir / html_filename

    tag_str = ", ".join(f"#{t.replace(' ','_')}" for t, _ in all_tags.most_common(8)) if all_tags else ""
    year_str = f"{min(all_years)} – {max(all_years)}" if all_years else ""

    sections = []
    for entry in pages_data:
        result     = entry["result"]
        pn         = entry["pdf_page"]
        jpeg_bytes = entry["jpeg"]
        is_sketch  = result.get("ist_skizze", False)
        is_title   = result.get("ist_titelseite", False)
        ref_pages  = result.get("seitenzahlen", "").strip()
        years      = result.get("jahreszahlen") or []
        tags       = result.get("tags") or []
        summary    = result.get("kurzzusammenfassung", "").strip()
        text       = result.get("transkription", "").strip()
        epoche     = result.get("epoche", "")

        heading = "Titelseite" if is_title else (ref_pages if ref_pages else f"Notizseite {pn}")
        sketch_icon = " 🗺" if is_sketch else ""

        meta_parts = []
        if epoche:
            meta_parts.append(f"<em>{epoche}</em>")
        if years:
            meta_parts.append(" · ".join(f"<strong>{y}</strong>" for y in years))
        if tags:
            meta_parts.append(" ".join(f"<span style='color:#8B6914'>#{t}</span>" for t in tags))
        meta_html = " &nbsp;|&nbsp; ".join(meta_parts)

        # Skizze einbetten
        img_html = ""
        if is_sketch and jpeg_bytes:
            try:
                hq = render_page_jpeg_cached(entry.get("pdf_path"), pn, SKETCH_DPI)
            except Exception:
                hq = jpeg_bytes
            b64 = base64.standard_b64encode(hq).decode()
            img_html = f'<p><img src="data:image/jpeg;base64,{b64}" style="max-width:600px;border:1px solid #ccc;border-radius:4px"></p>'

        text_html = text.replace("\n", "<br>") if text else ""

        sections.append(f"""
    <div style="border-bottom:2px solid #c8922a;margin-bottom:28px;padding-bottom:20px">
      <h2 style="font-family:Georgia,serif;color:#1c1a17;margin:0 0 6px">{heading}{sketch_icon}</h2>
      <p style="font-size:13px;color:#666;margin:0 0 10px">{meta_html}</p>
      {"<blockquote style='border-left:3px solid #c8922a;margin:0 0 12px;padding:6px 14px;color:#555;font-style:italic'>" + summary + "</blockquote>" if summary else ""}
      <p style="font-family:Georgia,serif;line-height:1.7;white-space:pre-wrap">{text_html}</p>
      {img_html}
    </div>""")

    html = f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>{book}</title>
<style>
  body{{font-family:Georgia,serif;max-width:800px;margin:40px auto;padding:0 24px;background:#fff;color:#1c1a17}}
  h1{{font-size:2rem;color:#1c1a17;border-bottom:3px solid #c8922a;padding-bottom:10px}}
  .meta-header{{color:#666;font-size:14px;margin-bottom:32px;line-height:2}}
</style>
</head>
<body>
<h1>{book}</h1>
<div class="meta-header">
  {"<strong>Autor:</strong> " + author + "<br>" if author else ""}
  <strong>Gelesen:</strong> {read_date}<br>
  {"<strong>Zeitspanne:</strong> " + year_str + "<br>" if year_str else ""}
  {"<strong>Tags:</strong> " + tag_str + "<br>" if tag_str else ""}
</div>
{"".join(sections)}
</body>
</html>"""

    html_path.write_text(html, encoding="utf-8")
    return html_path

# ── Eingabe-Abfrage ────────────────────────────────────────────────────────

def ask(prompt, default=None):
    suffix = f" [{default}]" if default else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val or default or ""

def choose(prompt, options):
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    while True:
        val = input("Auswahl: ").strip()
        if val.isdigit() and 1 <= int(val) <= len(options):
            return int(val) - 1
        print(f"Bitte 1–{len(options)} eingeben.")

# ── Hauptprogramm ──────────────────────────────────────────────────────────

def main():
    print("═" * 52)
    print("  Lesearchiv Exporter")
    print("  Obsidian & OneNote Edition")
    print("  Tipp: python lesearchiv_export.py --debug  fuer ausfuehrliche Fehlermeldungen")
    print("═" * 52 + "\n")
    if DEBUG:
        print("  [DEBUG-MODUS aktiv]\n")

    # API Key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        api_key = ask("Anthropic API Key (sk-ant-...)")
    if not api_key:
        print("Kein API Key – Abbruch."); sys.exit(1)

    # PDF
    pdf_path = ask("\nPfad zur PDF-Datei").strip('"').strip("'")
    if not os.path.exists(pdf_path):
        print(f"Nicht gefunden: {pdf_path}"); sys.exit(1)

    total = get_page_count(pdf_path)
    if not total:
        print("Seitenanzahl konnte nicht ermittelt werden."); sys.exit(1)
    print(f"  → {total} Seiten erkannt")

    # Buchinfos
    default_name = Path(pdf_path).stem.replace("_", " ").replace(" - ", " – ")
    book      = ask("\nBuchtitel", default_name)
    author    = ask("Autor")
    read_date = ask("Lesedatum (leer = heute)") or datetime.now().strftime("%Y-%m-%d")

    # Zielformat
    fmt_idx = choose("Zielformat wählen:", ["Obsidian (Markdown)", "OneNote (HTML)", "Beide"])
    do_obsidian = fmt_idx in (0, 2)
    do_onenote  = fmt_idx in (1, 2)

    # Ausgabeordner
    if do_obsidian:
        default_obs = str(Path.home() / "Obsidian" / "Lesearchiv")
        obs_root = Path(ask("Obsidian Vault-Ordner", default_obs))
        obs_out  = obs_root / safe_filename(book)
        obs_out.mkdir(parents=True, exist_ok=True)

    if do_onenote:
        default_one = str(Path.home() / "Documents" / "Lesearchiv")
        one_root = Path(ask("Ausgabeordner für OneNote-HTML", default_one))
        one_out  = one_root / safe_filename(book)
        one_out.mkdir(parents=True, exist_ok=True)

    # Seiten
    pages_in  = ask("\nSeiten verarbeiten (leer = alle, z.B. '2-10' oder '2,5,8')")
    page_nums = parse_pages(pages_in, total)
    print(f"\n  → {len(page_nums)} Seiten werden verarbeitet\n")

    # Transkription
    client = anthropic.Anthropic(api_key=api_key)
    pages_data = []
    skipped    = []
    all_years  = set()
    all_tags   = Counter()

    for i, pn in enumerate(page_nums, 1):
        print(f"[{i:2}/{len(page_nums)}] Seite {pn:2} … ", end="", flush=True)
        try:
            jpeg = render_page_jpeg(pdf_path, pn, RENDER_DPI)
        except Exception as e:
            print(f"✗ Render-Fehler: {e}")
            skipped.append(pn); continue

        result = transcribe_page(client, jpeg, pn)
        if result is None:
            print("✗ Transkription fehlgeschlagen")
            skipped.append(pn); continue

        # Korrektur-Schritt
        raw_text = result.get("transkription", "")
        if raw_text.strip() and not result.get("ist_skizze"):
            print("✎ ", end="", flush=True)
            result["transkription"] = correct_text(client, raw_text, pn, book=book, author=author)
            time.sleep(1)

        years = result.get("jahreszahlen") or []
        tags  = result.get("tags") or []
        all_years.update(years)
        all_tags.update(tags)

        pages_data.append({
            "pdf_page": pn,
            "result":   result,
            "jpeg":     jpeg,
            "pdf_path": pdf_path
        })

        sketch_icon = " 🗺" if result.get("ist_skizze") else ""
        print(f"✓{sketch_icon}  {result.get('seitenzahlen','') or f'Notizseite {pn}'}  {years or ''}  {tags[:2]}")

        if i < len(page_nums):
            time.sleep(PAUSE_BETWEEN)

    if not pages_data:
        print("\nKeine Seiten erfolgreich transkribiert."); sys.exit(1)

    # Export
    print("\nExportiere …")

    if do_obsidian:
        md_path = export_obsidian(book, author, read_date, obs_out, pages_data, all_years, all_tags)
        print(f"  ✓ Obsidian: {md_path}")

    if do_onenote:
        html_path = export_onenote(book, author, read_date, one_out, pages_data, all_years, all_tags)
        print(f"  ✓ OneNote:  {html_path}")
        print(f"     → Im Browser öffnen, alles auswählen (Strg+A), in OneNote einfügen")

    print("\n" + "═" * 52)
    print(f"  Fertig! {len(pages_data)} Seiten transkribiert", end="")
    if skipped:
        print(f", {len(skipped)} fehlgeschlagen: {skipped}", end="")
    print("\n" + "═" * 52)

if __name__ == "__main__":
    main()
