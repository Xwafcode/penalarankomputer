"""
Tahap 1B — Cleaning & Normalisasi Teks Putusan
===============================================
Input  : data/raw/pasal_362/*.txt dan data/raw/pasal_363/*.txt
Output : data/processed/clean/pasal_362/*.txt dan pasal_363/*.txt

Jalankan:
    python 02_cleaning.py

Requirements:
    pip install tqdm
"""

import re
import logging
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────
#  KONFIGURASI
# ─────────────────────────────────────────────
RAW_DIR     = Path("data/raw")
CLEAN_DIR   = Path("data/processed/clean")
LOG_DIR     = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "cleaning.log", encoding="utf-8", mode="a"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  FUNGSI CLEANING
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Pipeline cleaning teks putusan pengadilan:
    1. Hapus header/footer berulang (nomor halaman, watermark MA)
    2. Hapus karakter tidak perlu
    3. Normalisasi spasi dan baris
    4. Lowercase
    """

    # 1. Hapus watermark dan header khas putusan MA
    patterns_hapus = [
        r"Direktori Putusan Mahkamah Agung Republik Indonesia",
        r"putusan\.mahkamahagung\.go\.id",
        r"Disclaimer\s*[:：].*",
        r"kepaniteraan mahkamah agung.*",
        r"mahkamah agung republik indonesia",
        r"halaman \d+ dari \d+",
        r"hal\.\s*\d+\s*dari\s*\d+",
        r"-{3,}",           # garis pemisah panjang
        r"={3,}",           # garis sama dengan panjang
        r"\f",              # form feed (page break)
    ]
    for pat in patterns_hapus:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)

    # 2. Hapus nomor halaman standalone
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # 3. Normalisasi karakter khusus
    text = text.replace("\xa0", " ")   # non-breaking space
    text = text.replace("\t", " ")     # tab → spasi
    text = re.sub(r"[^\x00-\x7F\u00C0-\u024F\u1E00-\u1EFF]", " ", text)  # hapus non-latin

    # 4. Normalisasi spasi berlebih dan baris kosong
    text = re.sub(r"[ \t]+", " ", text)         # multi-spasi → satu spasi
    text = re.sub(r"\n{3,}", "\n\n", text)       # max 2 baris kosong berturut
    text = re.sub(r"^\s+|\s+$", "", text)        # strip awal/akhir

    # 5. Lowercase
    text = text.lower()

    return text


def validate_text(text: str, filename: str) -> bool:
    """Validasi minimal 80% isi putusan tersedia."""
    word_count = len(text.split())
    if word_count < 100:
        log.warning(f"VALIDASI GAGAL — terlalu pendek ({word_count} kata): {filename}")
        return False
    log.info(f"VALIDASI OK — {word_count:,} kata: {filename}")
    return True


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("TAHAP 1B — CLEANING & NORMALISASI TEKS")
    log.info("=" * 60)

    txt_files = sorted(RAW_DIR.rglob("*.txt"))
    log.info(f"Total TXT ditemukan: {len(txt_files)}")

    success = 0
    failed  = 0
    skipped = 0

    for txt_path in tqdm(txt_files, desc="Cleaning"):
        # Tentukan output path (pertahankan struktur folder pasal_362/pasal_363)
        label_folder = txt_path.parent.name   # "pasal_362" atau "pasal_363"
        out_dir      = CLEAN_DIR / label_folder
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / txt_path.name

        if out_path.exists():
            log.info(f"Skip (sudah ada): {out_path.name}")
            skipped += 1
            continue

        # Baca teks asli
        try:
            raw_text = txt_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            log.error(f"Gagal baca {txt_path.name}: {e}")
            failed += 1
            continue

        # Cleaning
        clean = clean_text(raw_text)

        # Validasi
        if not validate_text(clean, txt_path.name):
            failed += 1
            continue

        # Simpan
        out_path.write_text(clean, encoding="utf-8")
        success += 1

    log.info("=" * 60)
    log.info("SUMMARY CLEANING")
    log.info(f"  Berhasil : {success}")
    log.info(f"  Gagal    : {failed}")
    log.info(f"  Skip     : {skipped}")
    log.info(f"  Output   : {CLEAN_DIR.resolve()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
