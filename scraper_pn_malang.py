"""
Scraper Putusan Pencurian - PN Malang
Direktori Mahkamah Agung RI (putusan3.mahkamahagung.go.id)

Target   : 60 putusan (30 Pasal 362 + 30 Pasal 363)
Output   : /data/raw/pasal_362/*.pdf dan /data/raw/pasal_363/*.pdf
Jalankan : python scraper_pn_malang.py

Requirements:
    pip install requests beautifulsoup4 lxml tqdm
"""

import os
import re
import time
import json
import random
import logging
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm

# ─────────────────────────────────────────────
#  KONFIGURASI
# ─────────────────────────────────────────────
BASE_URL   = "https://putusan3.mahkamahagung.go.id"
PENGADILAN = "pn-malang"
KATEGORI   = "pencurian-1"

TARGET = {
    "pasal_362": 30,
    "pasal_363": 30,
}

OUTPUT_DIR = Path("data/raw")
LOG_DIR    = Path("logs")

DELAY_MIN = 2.0
DELAY_MAX = 4.0

# ─────────────────────────────────────────────
#  COOKIE — update nilai ini dari browsermu
# ─────────────────────────────────────────────
CF_CLEARANCE = "SPW0tAg1uCRhfWtgefZQQlEtqaWkUmbpCqqOfDueAEE-1782046380-1.2.1.1-9kFmqjCp0EbHBVDQ51eNknStfK_XdrSkh.RubbzYz4P_WuyJXu2VADz9pmkMJNR1kxXGwr8EwErMszyovqdDSGVvLK2eJXDd6okD0OpYIR2lqle3AijbmjyNBIx2w_4G4TkOHD4APzg9C7UrruMWOeKmEvdww0AoSrVojJ_ztBnIKvOo2i5knZE58o4Bg_RnR6Aw_ngYlHcZHuiog_f_9Zbzwns51PfnfiwOI1rJiBM_PQC_Sri2kN9H6wwbMr7CPdj0XVVLCkFfeRZ0_ItqFJXUVbkOTq6fJHqIo4BNHsHcirRJkAxmXPiHOKAazVjxeMwYQrMKKWHASMqOHL3bkPYGeSiVjHh6p9_vuS2xL867f6S9V0WalcebRxyoj2lRfEa.0c1Xy7cGfy7mtBfxnN0ehiqbbD0BBHYUV6rKsY_xtHtCXBb0sRDqBGXUSpNL"
PHPSESSID    = ""  # isi jika ada
USER_AGENT   = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"

_cookie_parts = [f"cf_clearance={CF_CLEARANCE}"]
if PHPSESSID:
    _cookie_parts.append(f"PHPSESSID={PHPSESSID}")
COOKIE_STRING = "; ".join(_cookie_parts)

# ─────────────────────────────────────────────
#  SETUP LOGGING
# ─────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "scraper.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  HTTP SESSION
# ─────────────────────────────────────────────
session = requests.Session()
session.headers.update({
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Referer": BASE_URL + "/",
    "Cookie": COOKIE_STRING,
})


def delay():
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))


# ─────────────────────────────────────────────
#  FUNGSI LISTING
# ─────────────────────────────────────────────
def get_listing_page(page: int) -> BeautifulSoup | None:
    url = (
        f"{BASE_URL}/direktori/index/pengadilan/{PENGADILAN}"
        f"/kategori/{KATEGORI}/page/{page}.html"
    )
    try:
        r = session.get(url, timeout=20)
        r.raise_for_status()
        return BeautifulSoup(r.text, "lxml")
    except requests.RequestException as e:
        log.error(f"Gagal ambil halaman listing {page}: {e}")
        return None


def extract_case_links(soup: BeautifulSoup) -> list[dict]:
    """
    Ekstrak semua link putusan dari halaman listing.
    Direktori MA: setiap item putusan ada di dalam tag <h2> atau <a> dengan
    href mengandung /putusan/ atau /direktori/putusan/
    """
    cases = []
    seen_urls = set()

    all_links = soup.find_all("a", href=True)
    for a in all_links:
        href = a["href"]
        # Filter hanya link ke halaman detail putusan
        if not ("/putusan/" in href or "/direktori/putusan/" in href):
            continue

        title = a.get_text(strip=True)
        if not title or len(title) < 5:
            continue

        full_url = href if href.startswith("http") else BASE_URL + href
        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        cases.append({
            "title": title,
            "url": full_url,
        })

    return cases


# ─────────────────────────────────────────────
#  KLASIFIKASI PASAL — dari halaman detail
# ─────────────────────────────────────────────
def get_pasal_from_detail(case_url: str) -> tuple[str | None, str | None]:
    """
    Buka halaman detail putusan.
    Return: (label_pasal, pdf_url)
    label_pasal: 'pasal_362', 'pasal_363', atau None
    pdf_url    : URL download PDF atau None
    """
    try:
        r = session.get(case_url, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        full_text = soup.get_text(" ", strip=True).lower()

        # ── Klasifikasi pasal ──
        label = None
        # Cek 363 dulu (lebih spesifik, mencakup subset 362)
        if re.search(r'pasal\s*3\s*6\s*3', full_text) or "dengan pemberatan" in full_text:
            label = "pasal_363"
        elif re.search(r'pasal\s*3\s*6\s*2', full_text):
            label = "pasal_362"
        elif "pencurian dengan pemberatan" in full_text:
            label = "pasal_363"
        elif "pencurian" in full_text:
            label = "pasal_362"  # default: pencurian biasa

        # ── Cari URL PDF ──
        pdf_url = None
        pdf_patterns = [
            lambda s: s.find("a", href=re.compile(r"\.pdf", re.I)),
            lambda s: s.find("a", string=re.compile(r"download|unduh|pdf", re.I)),
            lambda s: s.find("a", href=re.compile(r"/download/", re.I)),
            lambda s: s.find("a", href=re.compile(r"/pdf/", re.I)),
            lambda s: s.find("a", href=re.compile(r"putusan.*\.pdf", re.I)),
        ]
        for pattern in pdf_patterns:
            tag = pattern(soup)
            if tag:
                href = tag["href"]
                pdf_url = href if href.startswith("http") else BASE_URL + href
                break

        # Coba iframe PDF
        if not pdf_url:
            iframe = soup.find("iframe", src=re.compile(r"\.pdf", re.I))
            if iframe:
                src = iframe["src"]
                pdf_url = src if src.startswith("http") else BASE_URL + src

        # Log preview untuk debug
        if label:
            log.debug(f"Label: {label} | PDF: {pdf_url} | URL: {case_url}")
        else:
            log.warning(f"Label tidak terdeteksi: {case_url}")

        return label, pdf_url

    except requests.RequestException as e:
        log.error(f"Gagal ambil detail {case_url}: {e}")
        return None, None


# ─────────────────────────────────────────────
#  DOWNLOAD PDF
# ─────────────────────────────────────────────
def download_pdf(pdf_url: str, save_path: Path) -> bool:
    try:
        r = session.get(pdf_url, timeout=60, stream=True)
        r.raise_for_status()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        size_kb = save_path.stat().st_size / 1024
        if size_kb < 5:
            log.warning(f"File terlalu kecil ({size_kb:.1f} KB), mungkin bukan PDF: {save_path.name}")
            save_path.unlink(missing_ok=True)
            return False

        log.info(f"TERSIMPAN: {save_path.name} ({size_kb:.1f} KB)")
        return True

    except requests.RequestException as e:
        log.error(f"Gagal download {pdf_url}: {e}")
        return False


def main():
    log.info("=" * 60)
    log.info("SCRAPER PUTUSAN PN MALANG — PENCURIAN")
    log.info(f"Target: {TARGET}")
    log.info("=" * 60)

    collected  = {"pasal_362": 0, "pasal_363": 0}
    metadata_list = []
    page = 1
    max_pages = 150

    with tqdm(total=sum(TARGET.values()), desc="Total putusan") as pbar:
        while page <= max_pages:
            if all(collected[k] >= TARGET[k] for k in TARGET):
                log.info("Semua target terpenuhi!")
                break

            log.info(f"--- Halaman listing {page} ---")
            soup = get_listing_page(page)

            if soup is None:
                log.warning(f"Halaman {page} gagal, lewati.")
                page += 1
                delay()
                continue

            cases = extract_case_links(soup)
            if not cases:
                log.info(f"Tidak ada link di halaman {page}, berhenti.")
                break

            log.info(f"Ditemukan {len(cases)} link di halaman {page}")

            for case in cases:
                if all(collected[k] >= TARGET[k] for k in TARGET):
                    break

                # Ambil pasal & PDF URL dari halaman detail
                delay()
                label, pdf_url = get_pasal_from_detail(case["url"])

                if label is None:
                    log.warning(f"Skip (label None): {case['title'][:60]}")
                    continue

                if collected[label] >= TARGET[label]:
                    log.info(f"Kuota {label} penuh, skip.")
                    continue

                if pdf_url is None:
                    log.warning(f"PDF tidak ditemukan: {case['url']}")
                    continue

                idx       = collected[label] + 1
                filename  = f"case_{label}_{idx:03d}.pdf"
                save_path = OUTPUT_DIR / label / filename

                if save_path.exists():
                    log.info(f"Sudah ada: {filename}, lewati.")
                    collected[label] += 1
                    pbar.update(1)
                    continue

                success = download_pdf(pdf_url, save_path)
                if success:
                    collected[label] += 1
                    pbar.update(1)
                    metadata_list.append({
                        "case_id":    f"{label}_{idx:03d}",
                        "filename":   filename,
                        "label":      label,
                        "title":      case["title"],
                        "source_url": case["url"],
                        "pdf_url":    pdf_url,
                    })
                    log.info(
                        f"Progress → 362: {collected['pasal_362']}/{TARGET['pasal_362']} | "
                        f"363: {collected['pasal_363']}/{TARGET['pasal_363']}"
                    )

            page += 1
            delay()

    # Simpan metadata
    meta_path = OUTPUT_DIR / "metadata_raw.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)
    log.info(f"Metadata disimpan: {meta_path}")

    # Summary
    log.info("=" * 60)
    log.info("SUMMARY AKHIR")
    for label, count in collected.items():
        status = "SELESAI" if count >= TARGET[label] else "KURANG"
        log.info(f"  [{status}] {label}: {count}/{TARGET[label]}")
    log.info(f"  Total metadata: {len(metadata_list)} entri")
    log.info(f"  File PDF: {OUTPUT_DIR.resolve()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
