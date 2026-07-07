"""
Lesearchiv – Streamlit App
==========================
Deployment: Streamlit Cloud
Secrets:    ANTHROPIC_API_KEY in st.secrets

Lokale Installation:
    pip install streamlit anthropic pymupdf
    streamlit run lesearchiv_app.py
"""

import os, json, base64, time, re, io, zipfile, tempfile, subprocess
from pathlib import Path
from datetime import datetime
from collections import Counter

import streamlit as st

try:
    import anthropic
except ImportError:
    st.error("Fehlt: pip install anthropic")
    st.stop()

# PyMuPDF oder pdftoppm
try:
    import fitz
    HAVE_PYMUPDF = True
except ImportError:
    HAVE_PYMUPDF = False

# ── Konfiguration ──────────────────────────────────────────────────────────
RENDER_DPI  = 150
SKETCH_DPI  = 200
MAX_RETRIES = 3

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
- transkription: ganzer lesbarer Text, Zeilenumbrüche innerhalb eines Eintrags zu Fließtext zusammenführen
- Nur bei neuem Seitenzahl-Eintrag oder explizitem Absatz einen doppelten Zeilenumbruch setzen
- jahreszahlen: alle Jahreszahlen als Integer-Array, leer wenn keine
- tags: 3-6 deutsche thematische Schlagwörter
- epoche: Hauptepoche (z.B. "15. Jahrhundert", "Frühe Neuzeit")
- seitenzahlen: im Buch referenzierte Seiten (z.B. "S.12, S.16")
- ist_skizze: true wenn Seite hauptsächlich Karte/Zeichnung/Skizze
- ist_titelseite: true wenn erste Seite mit Buchinformation
- kurzzusammenfassung: ein einziger deutscher Satz
"""

KORREKTUR_PROMPT = """Du bekommst eine rohe Transkription handgeschriebener Lesenotizen eines österreichischen Lesers.
Die Notizen wurden per KI aus Handschrift erkannt und enthalten daher möglicherweise Erkennungsfehler.

Deine Aufgabe:
1. Korrigiere offensichtliche Erkennungsfehler zu grammatikalisch korrektem Deutsch
2. Behalte den inhaltlichen Stil und die persönliche Ausdrucksweise des Autors bei
3. Erfinde KEINEN Inhalt – wenn etwas unleserlich ist, markiere es mit [?]
4. Behalte Seitenzahlen (z.B. "S.12", "S.165") exakt so wie sie sind
5. Behalte Eigennamen, Jahreszahlen und Fachbegriffe exakt wie im Original
6. Kürze NICHT – gib den vollständigen korrigierten Text zurück
7. Führe Zeilenumbrüche innerhalb eines Seitenzahl-Eintrags zu Fließtext zusammen

Antworte NUR mit dem korrigierten Text, ohne Erklärungen."""

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

def get_page_count_from_bytes(pdf_bytes):
    if HAVE_PYMUPDF:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        count = doc.page_count
        doc.close()
        return count
    # Fallback: pdftoppm via tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        tmp = f.name
    try:
        out = subprocess.check_output(["pdfinfo", tmp], stderr=subprocess.DEVNULL).decode()
        for line in out.splitlines():
            if line.startswith("Pages:"):
                return int(line.split(":")[1].strip())
    finally:
        os.unlink(tmp)
    return None

def render_page_from_bytes(pdf_bytes, page_num, dpi=150):
    if HAVE_PYMUPDF:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[page_num - 1]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        jpeg = pix.tobytes("jpeg")
        doc.close()
        return jpeg
    # Fallback: pdftoppm
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        tmp = f.name
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "pg")
            subprocess.run([
                "pdftoppm", "-jpeg", "-r", str(dpi),
                "-f", str(page_num), "-l", str(page_num),
                tmp, prefix
            ], check=True, stderr=subprocess.DEVNULL)
            files = sorted(Path(tmpdir).glob("*.jpg"))
            if files:
                return files[0].read_bytes()
    finally:
        os.unlink(tmp)
    raise RuntimeError(f"Rendering fehlgeschlagen für Seite {page_num}")

def transcribe_page(client, jpeg_bytes, page_num):
    b64 = base64.standard_b64encode(jpeg_bytes).decode()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": PROMPT}
                ]}]
            )
            raw = response.content[0].text.strip()
            raw = re.sub(r'^```json\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            return json.loads(raw)
        except json.JSONDecodeError:
            if attempt == MAX_RETRIES:
                return None
            time.sleep(3)
        except Exception:
            if attempt == MAX_RETRIES:
                return None
            time.sleep(5 * attempt)
    return None

def correct_text(client, raw_text, book="", author=""):
    if not raw_text or not raw_text.strip():
        return raw_text
    kontext = ""
    if book or author:
        kontext = 'Kontext: Lesenotizen zu "' + book + '"'
        if author:
            kontext += f" von {author}"
        kontext += ". Eigennamen korrekt schreiben.\n\n"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                messages=[{"role": "user",
                           "content": KORREKTUR_PROMPT + "\n\n" + kontext + "Rohe Transkription:\n" + raw_text}]
            )
            return response.content[0].text.strip()
        except Exception:
            if attempt == MAX_RETRIES:
                return raw_text
            time.sleep(3 * attempt)
    return raw_text

# ── Export: Obsidian ───────────────────────────────────────────────────────

def build_obsidian_zip(book, author, read_date, pages_data, all_years, all_tags):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        lines = ["---"]
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
        lines += ["---", "", f"# {book}", ""]
        meta = []
        if author:
            meta.append(f"**Autor:** {author}")
        meta.append(f"**Gelesen:** {read_date}")
        if all_years:
            meta.append(f"**Zeitspanne:** {min(all_years)} – {max(all_years)}")
        lines.append("  \n".join(meta))
        lines += ["", "---", ""]

        for entry in pages_data:
            result    = entry["result"]
            pn        = entry["pdf_page"]
            is_sketch = result.get("ist_skizze", False)
            is_title  = result.get("ist_titelseite", False)
            ref_pages = result.get("seitenzahlen", "").strip()
            years     = result.get("jahreszahlen") or []
            tags      = result.get("tags") or []
            summary   = result.get("kurzzusammenfassung", "").strip()
            text      = result.get("transkription", "").strip()
            epoche    = result.get("epoche", "")

            heading = "Titelseite" if is_title else (ref_pages if ref_pages else f"Notizseite {pn}")
            lines.append(f"## {heading}{' 🗺' if is_sketch else ''}")
            lines.append("")
            meta2 = []
            if epoche: meta2.append(f"*{epoche}*")
            if years:  meta2.append(" · ".join(f"**{y}**" for y in years))
            if tags:   meta2.append(" ".join(f"#{t.replace(' ','_')}" for t in tags))
            if meta2:
                lines.append("  \n".join(meta2))
                lines.append("")
            if summary:
                lines += [f"> {summary}", ""]
            if text:
                lines += [text, ""]
            if is_sketch and entry.get("jpeg"):
                img_name = f"_Bilder/{safe_filename(book)}_S{pn:03d}.jpg"
                zf.writestr(img_name, entry["jpeg"])
                lines += [f"![[{img_name}]]", ""]
            lines += ["---", ""]

        md_name = safe_filename(book) + ".md"
        zf.writestr(md_name, "\n".join(lines))
    buf.seek(0)
    return buf.getvalue()

# ── Export: OneNote ────────────────────────────────────────────────────────

def build_onenote_html(book, author, read_date, pages_data, all_years, all_tags):
    tag_str  = ", ".join(f"#{t.replace(' ','_')}" for t, _ in all_tags.most_common(8)) if all_tags else ""
    year_str = f"{min(all_years)} – {max(all_years)}" if all_years else ""
    sections = []

    for entry in pages_data:
        result    = entry["result"]
        pn        = entry["pdf_page"]
        is_sketch = result.get("ist_skizze", False)
        is_title  = result.get("ist_titelseite", False)
        ref_pages = result.get("seitenzahlen", "").strip()
        years     = result.get("jahreszahlen") or []
        tags      = result.get("tags") or []
        summary   = result.get("kurzzusammenfassung", "").strip()
        text      = result.get("transkription", "").strip()
        epoche    = result.get("epoche", "")

        heading = "Titelseite" if is_title else (ref_pages if ref_pages else f"Notizseite {pn}")
        meta_parts = []
        if epoche: meta_parts.append(f"<em>{epoche}</em>")
        if years:  meta_parts.append(" · ".join(f"<strong>{y}</strong>" for y in years))
        if tags:   meta_parts.append(" ".join(f"<span style='color:#8B6914'>#{t}</span>" for t in tags))
        meta_html = " &nbsp;|&nbsp; ".join(meta_parts)

        img_html = ""
        if is_sketch and entry.get("jpeg"):
            b64 = base64.standard_b64encode(entry["jpeg"]).decode()
            img_html = f'<p><img src="data:image/jpeg;base64,{b64}" style="max-width:600px;border:1px solid #ccc;border-radius:4px"></p>'

        text_html = text.replace("\n\n", "</p><p>").replace("\n", " ") if text else ""
        sections.append(f"""
  <div style="border-bottom:2px solid #c8922a;margin-bottom:28px;padding-bottom:20px">
    <h2 style="font-family:Georgia,serif;color:#1c1a17;margin:0 0 6px">{heading}{' 🗺' if is_sketch else ''}</h2>
    <p style="font-size:13px;color:#666;margin:0 0 10px">{meta_html}</p>
    {"<blockquote style='border-left:3px solid #c8922a;margin:0 0 12px;padding:6px 14px;color:#555;font-style:italic'>" + summary + "</blockquote>" if summary else ""}
    <p style="font-family:Georgia,serif;line-height:1.7">{text_html}</p>
    {img_html}
  </div>""")

    return f"""<!DOCTYPE html>
<html lang="de"><head><meta charset="UTF-8"><title>{book}</title>
<style>
  body{{font-family:Georgia,serif;max-width:800px;margin:40px auto;padding:0 24px;background:#fff;color:#1c1a17}}
  h1{{font-size:2rem;color:#1c1a17;border-bottom:3px solid #c8922a;padding-bottom:10px}}
  .meta{{color:#666;font-size:14px;margin-bottom:32px;line-height:2}}
</style></head><body>
<h1>{book}</h1>
<div class="meta">
  {"<strong>Autor:</strong> " + author + "<br>" if author else ""}
  <strong>Gelesen:</strong> {read_date}<br>
  {"<strong>Zeitspanne:</strong> " + year_str + "<br>" if year_str else ""}
  {"<strong>Tags:</strong> " + tag_str + "<br>" if tag_str else ""}
</div>
{"".join(sections)}
</body></html>"""

# ── Streamlit UI ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Lesearchiv",
    page_icon="📖",
    layout="centered"
)

st.markdown("""
<style>
  .block-container { max-width: 780px; }
  h1 { font-family: Georgia, serif; }
  .stProgress > div > div { background-color: #c8922a; }
</style>
""", unsafe_allow_html=True)

st.title("📖 Lesearchiv")
st.caption("Handgeschriebene Lesenotizen → Obsidian & OneNote")

# ── API Key ────────────────────────────────────────────────────────────────
api_key = st.secrets.get("ANTHROPIC_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
if not api_key:
    api_key = st.text_input("Anthropic API Key", type="password",
                             help="Wird nicht gespeichert. Besser: in Streamlit Secrets hinterlegen.")
if not api_key:
    st.info("Bitte API Key eingeben um fortzufahren.")
    st.stop()

client = anthropic.Anthropic(api_key=api_key)

# ── Eingaben ───────────────────────────────────────────────────────────────
st.divider()
col1, col2 = st.columns(2)
with col1:
    book   = st.text_input("Buchtitel", placeholder="Tony Horwitz – Es war nicht Columbus")
    author = st.text_input("Autor", placeholder="Tony Horwitz")
with col2:
    read_date  = st.date_input("Lesedatum", value=datetime.today()).strftime("%Y-%m-%d")
    export_fmt = st.selectbox("Exportformat", ["Obsidian (Markdown + ZIP)", "OneNote (HTML)", "Beide"])

pdf_file = st.file_uploader("PDF hochladen", type=["pdf"])

if pdf_file:
    pdf_bytes = pdf_file.read()
    try:
        total = get_page_count_from_bytes(pdf_bytes)
    except Exception as e:
        st.error(f"PDF konnte nicht gelesen werden: {e}")
        st.stop()

    st.success(f"PDF geladen: **{total} Seiten**")

    pages_input = st.text_input(
        "Seiten verarbeiten",
        value="",
        placeholder="Leer = alle  |  z.B. 2-10  oder  2,5,8"
    )
    page_nums = parse_pages(pages_input, total)
    st.caption(f"{len(page_nums)} Seiten ausgewählt · geschätzte Kosten: ~${len(page_nums)*0.02:.2f}")

    if not book:
        st.warning("Bitte Buchtitel eingeben.")
        st.stop()

    if st.button("▶ Transkription starten", type="primary", use_container_width=True):

        pages_data = []
        skipped    = []
        all_years  = set()
        all_tags   = Counter()

        progress  = st.progress(0)
        status    = st.status("Transkription läuft …", expanded=True)
        log       = st.empty()
        log_lines = []

        for i, pn in enumerate(page_nums):
            frac = i / len(page_nums)
            progress.progress(frac)
            status.update(label=f"Seite {pn} von {total} … ({i+1}/{len(page_nums)})")

            # Render
            try:
                jpeg = render_page_from_bytes(pdf_bytes, pn, RENDER_DPI)
            except Exception as e:
                log_lines.append(f"✗ Seite {pn}: Render-Fehler – {e}")
                log.code("\n".join(log_lines))
                skipped.append(pn)
                continue

            # Transkription
            result = transcribe_page(client, jpeg, pn)
            if result is None:
                log_lines.append(f"✗ Seite {pn}: Transkription fehlgeschlagen")
                log.code("\n".join(log_lines))
                skipped.append(pn)
                continue

            # Korrektur
            raw_text = result.get("transkription", "")
            if raw_text.strip() and not result.get("ist_skizze"):
                result["transkription"] = correct_text(client, raw_text, book=book, author=author)
                time.sleep(1)

            years = result.get("jahreszahlen") or []
            tags  = result.get("tags") or []
            all_years.update(years)
            all_tags.update(tags)

            sketch = " 🗺" if result.get("ist_skizze") else ""
            ref    = result.get("seitenzahlen") or f"Notizseite {pn}"
            log_lines.append(f"✓{sketch} {ref}  {years or ''}  {tags[:2]}")
            log.code("\n".join(log_lines))

            pages_data.append({
                "pdf_page": pn,
                "result":   result,
                "jpeg":     jpeg if result.get("ist_skizze") else None,
            })

            if i < len(page_nums) - 1:
                time.sleep(2)

        progress.progress(1.0)
        status.update(label=f"Fertig – {len(pages_data)} Seiten transkribiert", state="complete")

        if not pages_data:
            st.error("Keine Seiten konnten transkribiert werden.")
            st.stop()

        st.divider()
        st.subheader("Export")

        do_obsidian = export_fmt in ("Obsidian (Markdown + ZIP)", "Beide")
        do_onenote  = export_fmt in ("OneNote (HTML)", "Beide")

        if do_obsidian:
            zip_bytes = build_obsidian_zip(book, author, read_date, pages_data, all_years, all_tags)
            st.download_button(
                label="⬇ Obsidian ZIP herunterladen",
                data=zip_bytes,
                file_name=f"{safe_filename(book)}_Obsidian.zip",
                mime="application/zip",
                use_container_width=True
            )

        if do_onenote:
            html_str = build_onenote_html(book, author, read_date, pages_data, all_years, all_tags)
            st.download_button(
                label="⬇ OneNote HTML herunterladen",
                data=html_str.encode("utf-8"),
                file_name=f"{safe_filename(book)}_OneNote.html",
                mime="text/html",
                use_container_width=True
            )
            st.caption("HTML im Browser öffnen → Strg+A → in OneNote einfügen")

        if skipped:
            st.warning(f"Fehlgeschlagen: Seiten {skipped}")

        st.success(f"✓ {len(pages_data)} Seiten · {len(all_years)} Jahreszahlen · {len(all_tags)} Tags")
