"""
Tahap 1A — Konversi PDF ke Plain Text
======================================
Input  : data/raw/pasal_362/*.pdf dan data/raw/pasal_363/*.pdf
Output : data/raw/pasal_362/*.txt dan data/raw/pasal_363/*.txt

Jalankan:
    python 01_pdf_to_text.py

Requirements:
    pip install pdfminer.six tqdm
"""

import re
import logging
from pathlib import Path
from tqdm import tqdm
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError

# ─────────────────────────────────────────────
#  KONFIGURASI
# ─────────────────────────────────────────────
RAW_DIR = Path("data/raw")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Minimum karakter agar teks dianggap valid (80% isi putusan)
MIN_CHARS = 500

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "cleaning.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  FUNGSI EKSTRAKSI
# ─────────────────────────────────────────────
def pdf_to_text(pdf_path: Path) -> str | None:
    """Ekstrak teks dari PDF menggunakan pdfminer."""
    try:
        text = extract_text(str(pdf_path))
        if not text or len(text.strip()) < MIN_CHARS:
            log.warning(f"Teks terlalu pendek ({len(text.strip() if text else '')} char): {pdf_path.name}")
            return None
        return text
    except PDFSyntaxError as e:
        log.error(f"PDF corrupt/tidak valid: {pdf_path.name} — {e}")
        return None
    except Exception as e:
        log.error(f"Error ekstraksi {pdf_path.name}: {e}")
        return None


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("TAHAP 1A — KONVERSI PDF → TXT")
    log.info("=" * 60)

    # Kumpulkan semua PDF
    pdf_files = sorted(RAW_DIR.rglob("*.pdf"))
    log.info(f"Total PDF ditemukan: {len(pdf_files)}")

    success = 0
    failed  = 0
    skipped = 0

    for pdf_path in tqdm(pdf_files, desc="Konversi PDF"):
        txt_path = pdf_path.with_suffix(".txt")

        # Skip jika sudah ada
        if txt_path.exists():
            log.info(f"Skip (sudah ada): {txt_path.name}")
            skipped += 1
            continue

        text = pdf_to_text(pdf_path)
        if text is None:
            failed += 1
            continue

        txt_path.write_text(text, encoding="utf-8")
        log.info(f"OK: {txt_path.name} ({len(text):,} chars)")
        success += 1

    log.info("=" * 60)
    log.info("SUMMARY KONVERSI")
    log.info(f"  Berhasil : {success}")
    log.info(f"  Gagal    : {failed}")
    log.info(f"  Skip     : {skipped}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
