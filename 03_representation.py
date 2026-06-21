"""
Tahap 2 — Case Representation
==============================
Input  : data/processed/clean/pasal_362/*.txt dan pasal_363/*.txt
Output : data/processed/cases.csv dan cases.json

Kolom CSV:
    case_id, filename, label_pasal, label_vonis,
    no_perkara, tanggal, pengadilan, terdakwa,
    pasal_dakwaan, amar_putusan, vonis_bulan,
    ringkasan_fakta, argumen_hukum, word_count, text_full

Jalankan:
    python 03_representation.py

Requirements:
    pip install pandas tqdm openpyxl
"""

import re
import json
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────
#  KONFIGURASI
# ─────────────────────────────────────────────
CLEAN_DIR    = Path("data/processed/clean")
OUTPUT_DIR   = Path("data/processed")
LOG_DIR      = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "representation.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  FUNGSI EKSTRAKSI METADATA
# ─────────────────────────────────────────────
def ekstrak_no_perkara(text: str) -> str:
    """Ekstrak nomor perkara. Contoh: 45/Pid.B/2023/PN Mlg"""
    patterns = [
        r"\d+\s*/\s*pid\.b\s*/\s*\d{4}\s*/\s*pn\s*[\w\.]+",
        r"\d+\s*/\s*pid\.sus\s*/\s*\d{4}\s*/\s*pn\s*[\w\.]+",
        r"nomor\s*[:\s]+(\d+[/\-][\w\.]+[/\-]\d{4}[/\-][\w\.]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(0).strip()
    return ""


def ekstrak_tanggal(text: str) -> str:
    """Ekstrak tanggal putusan."""
    # Pola: 'diucapkan pada hari ... tanggal DD bulan YYYY'
    bulan_map = {
        "januari": "01", "februari": "02", "maret": "03", "april": "04",
        "mei": "05", "juni": "06", "juli": "07", "agustus": "08",
        "september": "09", "oktober": "10", "november": "11", "desember": "12"
    }
    patterns = [
        r"(\d{1,2})\s+(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)\s+(\d{4})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            hari, bln, thn = m.group(1), m.group(2).lower(), m.group(3)
            return f"{thn}-{bulan_map.get(bln, '00')}-{int(hari):02d}"
    return ""


def ekstrak_terdakwa(text: str) -> str:
    """Ekstrak nama terdakwa."""
    patterns = [
        r"terdakwa\s*[:\s]+([A-Z][a-zA-Z\s]+?)(?:\s*,|\s*alias|\s*bin|\s*binti|\n)",
        r"nama\s*lengkap\s*[:\s]+([A-Z][a-zA-Z\s]+?)(?:\s*;|\s*,|\n)",
        r"nama\s*[:\s]+([A-Z][a-zA-Z\s\.]+?)(?:\s*;|\s*,|\s*\n)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            nama = m.group(1).strip()
            # Bersihkan kata-kata tidak relevan
            if len(nama) > 3 and len(nama) < 60:
                return nama.title()
    return ""


def ekstrak_pasal_dakwaan(text: str) -> str:
    """Ekstrak pasal yang didakwakan."""
    found = []
    patterns = [
        r"pasal\s+3[56789]\d?\s+(?:jo\.?\s+pasal\s+\d+\s+)?kuhp",
        r"pasal\s+362\s+kuhp",
        r"pasal\s+363\s+(?:ayat\s+\(\d\)\s+)?kuhp",
        r"pasal\s+364\s+kuhp",
        r"pasal\s+365\s+kuhp",
    ]
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        found.extend(matches)
    return "; ".join(set(found)) if found else ""


def ekstrak_amar_putusan(text: str) -> str:
    """
    Ekstrak amar putusan (bersalah/bebas/lepas).
    Cari bagian MENGADILI dalam teks.
    """
    # Cari blok MENGADILI
    m = re.search(r"m\s*e\s*n\s*g\s*a\s*d\s*i\s*l\s*i(.{50,800}?)(?=menimbang|menetapkan|demikian|$)",
                  text, re.IGNORECASE | re.DOTALL)
    if m:
        blok = m.group(1).lower().strip()

        if re.search(r"terbukti.*bersalah|menyatakan.*terdakwa.*bersalah", blok):
            return "bersalah"
        if re.search(r"tidak terbukti|membebaskan terdakwa|bebas dari", blok):
            return "bebas"
        if re.search(r"melepaskan terdakwa|lepas dari", blok):
            return "lepas"

        # Kembalikan cuplikan amar jika tidak cocok pola di atas
        return blok[:200].replace("\n", " ").strip()
    return ""


def ekstrak_vonis(text: str) -> tuple[int, str]:
    """
    Ekstrak lama hukuman penjara dalam bulan.
    Return: (total_bulan, label_vonis)
    """
    tahun_m = re.search(r"pidana penjara selama\s+(\d+)\s*\([\w\s]+\)\s*tahun", text, re.IGNORECASE)
    bulan_m = re.search(r"pidana penjara selama\s+(\d+)\s*\([\w\s]+\)\s*bulan", text, re.IGNORECASE)

    # Coba pola alternatif
    if not tahun_m and not bulan_m:
        tahun_m = re.search(r"penjara\s+(\d+)\s+tahun", text, re.IGNORECASE)
        bulan_m = re.search(r"penjara\s+(\d+)\s+bulan", text, re.IGNORECASE)

    total = 0
    if tahun_m:
        total += int(tahun_m.group(1)) * 12
    if bulan_m:
        total += int(bulan_m.group(1))

    if total == 0:
        return 0, "tidak_diketahui"
    elif total < 12:
        return total, "ringan"
    elif total <= 60:
        return total, "sedang"
    else:
        return total, "berat"


def ekstrak_ringkasan_fakta(text: str) -> str:
    """
    Ekstrak ringkasan fakta dari bagian 'menimbang' atau 'fakta hukum'.
    Ambil max 500 karakter pertama dari bagian relevan.
    """
    patterns = [
        r"menimbang.*?bahwa(.{100,600}?)(?=menimbang|mempertimbangkan|mengenai|$)",
        r"fakta.{0,10}hukum(.{100,500}?)(?=menimbang|mempertimbangkan|$)",
        r"bahwa terdakwa(.{100,500}?)(?=bahwa|menimbang|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            snippet = m.group(1).replace("\n", " ").strip()
            snippet = re.sub(r"\s+", " ", snippet)
            return snippet[:500]
    return ""


def ekstrak_argumen_hukum(text: str) -> str:
    """
    Ekstrak argumen hukum utama dari bagian pertimbangan hakim.
    """
    patterns = [
        r"mempertimbangkan(.{100,600}?)(?=menimbang|mengadili|menetapkan|$)",
        r"pertimbangan hukum(.{100,500}?)(?=mengadili|menetapkan|$)",
        r"unsur.unsur(.{100,500}?)(?=menimbang|mengadili|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            snippet = m.group(1).replace("\n", " ").strip()
            snippet = re.sub(r"\s+", " ", snippet)
            return snippet[:500]
    return ""


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("TAHAP 2 — CASE REPRESENTATION")
    log.info("=" * 60)

    txt_files = sorted(CLEAN_DIR.rglob("*.txt"))
    log.info(f"Total file clean ditemukan: {len(txt_files)}")

    rows = []

    for txt_path in tqdm(txt_files, desc="Ekstraksi metadata"):
        label_folder = txt_path.parent.name  # pasal_362 / pasal_363
        text = txt_path.read_text(encoding="utf-8", errors="ignore")

        # Ekstrak semua metadata
        no_perkara      = ekstrak_no_perkara(text)
        tanggal         = ekstrak_tanggal(text)
        terdakwa        = ekstrak_terdakwa(text)
        pasal_dakwaan   = ekstrak_pasal_dakwaan(text)
        amar_putusan    = ekstrak_amar_putusan(text)
        vonis_bulan, label_vonis = ekstrak_vonis(text)
        ringkasan_fakta = ekstrak_ringkasan_fakta(text)
        argumen_hukum   = ekstrak_argumen_hukum(text)
        word_count      = len(text.split())

        row = {
            "case_id":          txt_path.stem,
            "filename":         txt_path.name,
            "label_pasal":      label_folder,
            "label_vonis":      label_vonis,
            "no_perkara":       no_perkara,
            "tanggal":          tanggal,
            "pengadilan":       "PN Malang",
            "terdakwa":         terdakwa,
            "pasal_dakwaan":    pasal_dakwaan,
            "amar_putusan":     amar_putusan,
            "vonis_bulan":      vonis_bulan,
            "ringkasan_fakta":  ringkasan_fakta,
            "argumen_hukum":    argumen_hukum,
            "word_count":       word_count,
            "text_full":        text[:3000],  # simpan 3000 char pertama
        }
        rows.append(row)

        log.info(
            f"OK: {txt_path.name} | pasal={label_folder} | "
            f"vonis={label_vonis}({vonis_bulan}bln) | amar={amar_putusan[:30]}"
        )

    # ── Simpan ke CSV ──
    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "cases.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info(f"CSV disimpan: {csv_path} ({len(df)} baris)")

    # ── Simpan ke JSON ──
    json_path = OUTPUT_DIR / "cases.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    log.info(f"JSON disimpan: {json_path}")

    # ── Summary distribusi label ──
    log.info("=" * 60)
    log.info("DISTRIBUSI LABEL")
    log.info(f"  label_pasal:\n{df['label_pasal'].value_counts().to_string()}")
    log.info(f"  label_vonis:\n{df['label_vonis'].value_counts().to_string()}")
    log.info(f"  amar_putusan:\n{df['amar_putusan'].value_counts().head(5).to_string()}")
    log.info(f"  Rata-rata word count: {df['word_count'].mean():.0f} kata")
    log.info("=" * 60)

    # ── Preview 3 baris pertama ──
    print("\n=== PREVIEW cases.csv ===")
    preview_cols = ["case_id", "label_pasal", "label_vonis", "no_perkara",
                    "tanggal", "amar_putusan", "word_count"]
    print(df[preview_cols].head(3).to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
