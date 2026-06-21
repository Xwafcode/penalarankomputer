# CBR Putusan Pencurian - Pengadilan Negeri Malang

Sistem *Case-Based Reasoning* (CBR) untuk memprediksi pasal dan vonis tindak pidana pencurian (Pasal 362 vs Pasal 363 KUHP) berdasarkan ekstraksi teks putusan dari Pengadilan Negeri Malang. 

## 📂 Struktur Proyek
- `01_pdf_to_text.py` : Konversi file putusan PDF mentah menjadi teks.
- `02_cleaning.py` : Membersihkan teks dari format MA, header, dan karakter khusus.
- `03_representation.py` : Mengekstrak fitur (ringkasan fakta, vonis, dll) menjadi representasi terstruktur (`cases.json`).
- `04_retrieval.py` : Membangun model Retrieval menggunakan TF-IDF (+ SVM/NB) dan IndoBERT (+ Cosine Similarity).
- `05_predict.py` : Tahap Case Solution Reuse (Majority & Weighted Voting).
- `06_evaluation.py` : Tahap Evaluasi untuk mengukur performa prediksi.
- `app.py` : Antarmuka web menggunakan Streamlit untuk mencoba klasifikasi kasus baru.
- `data/` : Menyimpan dataset mentah, teks bersih, dan hasil prediksi evaluasi.
- `models/` : Menyimpan vektor TF-IDF dan bobot model pre-trained yang digunakan.

## 🚀 Instalasi

Pastikan Python 3.9+ sudah terinstall. Jalankan perintah berikut untuk menginstall dependensi:
```bash
pip install -r requirements.txt
```

## ⚙️ Cara Menjalankan

### Menjalankan Pipeline Lengkap
Jika Anda ingin me-rebuild seluruh dataset dan melatih ulang model dari awal:
```bash
python 01_pdf_to_text.py
python 02_cleaning.py
python 03_representation.py
python 04_retrieval.py
python 05_predict.py
python 06_evaluation.py
```

### Menjalankan Antarmuka Web (Streamlit)
Untuk langsung menguji sistem dengan teks putusan baru, jalankan aplikasi web:
```bash
streamlit run app.py
```
Aplikasi akan otomatis terbuka di browser pada alamat `http://localhost:8501`.

## 📊 Hasil Evaluasi Singkat
- **Label Pasal (362 vs 363):** TF-IDF + SVM adalah metode terbaik dengan F1-Score **0.9161**. TF-IDF sangat sensitif terhadap *keyword* spesifik (misal: "malam hari", "merusak kunci") yang sering muncul pada Pasal 363.
- **Label Vonis (Ringan, Sedang, Berat):** IndoBERT + Cosine Similarity memiliki performa yang sedikit lebih baik dengan F1-Score **0.7083** dibandingkan TF-IDF, karena dapat menangkap kemiripan kronologis secara semantik.

## 👥 Kredit
Proyek Praktikum Penalaran Komputer - Fakultas Teknik Informatika UMM.
