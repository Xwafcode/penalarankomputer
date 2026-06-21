### FAKULTAS TEKNIK
INFORMATIKA
informatika.umm.ac.id | informatika@umm.ac.id
Mata Kuliah Penalaran Komputer
Semester Genap Tahun Akademik 2025/2026
[CPMK-1] Mahasiswa mampu menerapkan beragam metode akuisisi, analisis, serta pengolahan data
secara tepat yang sesuai dengan konsep penalaran komputer
[SubCPMK-3] Mahasiswa mampu menerapkan siklus dalam case-based reasoning
menggunakan dataset tertentu serta metode yang optimal.
Skor
Tugas:
1. Deskripsi Proyek
Mahasiswa akan merancang dan mengimplementasikan sistem Case-Based Reasoning
(CBR) sederhana berbasis Python untuk mendukung analisis putusan pengadilan.
Dengan memanfaatkan data putusan yang dipublikasikan di Direktori Putusan
Mahkamah Agung Republik Indonesia, merujuk pada siklus CBR, sistem ini memiliki
fitur:
a. Membangun Case Base
i. Mengunduh dan melakukan preprocessing teks putusan (≥ 30 dokumen) dari
satu jenis perkara (misal: pidana khusus Narkotika & Psikotropika).
ii. Membersihkan teks (hapus header/ footer, normalisasi karakter, tokenisasi) agar
siap diproses selanjutnya.
b. Case Representation
i. Mengekstrak metadata penting seperti nomor perkara, tanggal, jenis perkara,
pasal digunakan, nama pihak dan informasi lain dari metadata putusan.
ii. Menghasilkan text feature (ringkasan fakta, argumen hukum utama) dan
menyimpannya dalam format terstruktur (.csv/ .xlsx/ .json).
c. Case Retrieval
i. Membangun vektor representasi dokumen dengan memilih salah satu dari dua
pendekatan:
1. Pendekatan statistik dengan menggunakan metode seperti Term Frequency-
Inverse Document Frequency (TF-IDF) atau metode lain.
2. Pendekatan Text Embedding seperti model Transformer (BERT) atau model
lain.
ii. Menggunakan model untuk proses retrieval:
1. Menggunakan model machine learning seperti Support Vector Machine
(SVM) atau Naive Bayes pada representasi TF-IDF untuk classification/
retrieval.
2. Menggunakan model transformer (BERT/ RoBERTa/ IndoBERT/ dll) untuk
retrieval pada hasil embedding.
d. Case/ Solution Reuse
Dari top-k kasus (misal 5 kasus termirip), mengambil elemen putusan—misalnya
amar putusan atau dakwaan—sebagai “solusi” atas kasus baru.
e. Revisi & Retain (Opsional)
Menambahkan kasus baru yang terbukti solusi-nya tepat ke dalam case base untuk
iterasi berikutnya.
f. Evaluasi Model
i. Hasil Retrieval: Accuracy, Precision, Recall, F1-score (apabila tersedia label
hasil putusan).
ii. Analisis kegagalan model (Rejection) dan rekomendasi perbaikan.
2. Ruang Lingkup
a. Tim: Satu tim maksimal terdiri atas 2 Mahasiswa.
b. Jenis Perkara: pilih satu domain hukum (contoh: perdata gugatan wanprestasi,
pidana khusus narkotika & psikotropika). Di satu kelas penalaran komputer,
maksimal ada 2 domain hukum yang sama, selebihnya akan di diskualifikasi.
c. Volume Data: minimal 30 putusan.
25

---

### FAKULTAS TEKNIK
INFORMATIKA
informatika.umm.ac.id | informatika@umm.ac.id
d. Bahasa & Tools: Python, Jupyter Notebook, pandas, scikit-learn, transformers,
BeautifulSoup/requests dll.
3. Output yang Diharapkan
a. Codebase: notebook dan/atau script Python untuk tiap tahap CBR.
b. Github Repository: Link GitHub Repository yang memuat project CBR.
4. Spesifikasi Tugas per Tahap CBR
Berikut uraian tugas, langkah kerja, dan output yang harus disiapkan pada tiap tahap
siklus CBR:
a. Tahap 1: Membangun Case Base
Tujuan: mengumpulkan (scraping) dan menyiapkan (cleaning) corpus putusan yang
bersih.
Langkah Kerja sebagai berikut:
i. Seleksi & Unduh
1. Pilih satu domain perkara (misal: Perdata wanprestasi).
2. Unduh ≥ 30 dokumen putusan (PDF/HTML) dari Direktori MA RI.
ii. Konversi & Ekstraksi Teks
Ubah PDF → plain text (pdfminer, pdftotext) atau HTML → text
(BeautifulSoup).
iii. Pembersihan
1. Hapus header/ footer, nomor halaman, watermark.
2. Normalisasi spasi dan karakter (lower-case, remove punctuation if needed).
3. Simpan hasil di folder:
/data/raw/
├─ case_001.txt
├─ case_002.txt
└─ …
iv. Validasi
1. Periksa keutuhan teks (minimal 80% isi putusan tersedia).
2. Catat log file pembersihan: /logs/cleaning.log. (opsional)
v. Output:
1. Folder /data/raw/*.txt
2. cleaning.log (data history pembersihan) (opsional)
b. Tahap 2 : Case Representation
Tujuan: Representasikan setiap putusan dalam struktur data terorganisir
Langkah Kerja:
i. Ekstraksi Metadata
Ambil metadata dari setiap putusan seperti Nomor Perkara, Tanggal, Jenis
Perkara, Pasal, Pihak Penggugat & Tergugat, dll.
ii. Ekstraksi Konten Kunci
1. Ringkasan fakta (misal barang bukti dan dakwaan).
2. Argumen hukum utama (misal putusan dan pasal yang diputuskan).
i. Feature Engineering
Hitung length (jumlah kata), bag-of-words, atau buat QA-pairs
sederhana.
ii. Penyimpanan
Simpan ke format terstruktur:
• CSV: /data/processed/cases.csv, atau
• JSON: /data/processed/cases.json
Contoh kolom file CSV:
case_id no_perkara tanggal ringkasan_fakta pasal pihak text_full
1 123/Pdt.G/… 2023-02-10 “Pada tanggal…” 124 KUHPer A vs. B …
iii. Output: Data kasus dalam format .csv atau .json
c. Tahap 3 : Case Retrieval
Tujuan: Temukan kasus lama yang paling mirip dengan query kasus baru

---

### FAKULTAS TEKNIK
INFORMATIKA
informatika.umm.ac.id | informatika@umm.ac.id
Langkah Kerja:
i. Representasi Vektor
1. TF-IDF: sklearn.feature_extraction.text.TfidfVectorizer
2. BERT Embedding: transformers → model pre-trained (misal
indobenchmark/indobert-base-p1)
ii. Splitting Data
1. Lakukan splitting data untuk membagi data menjadi data train dan data test
2. Rasio perbandingan data dapat berdasarkan kebutuhan atau merujuk pada
artikel penelitian, missal 70:30 atau 80:20.
iii. Model Retrieval dengan memilih pendekatan berikut:
1. Gunakan model machine learning seperti Support Vector Machine (SVM)
atau Naive Bayes pada representasi TF-IDF untuk classification/ retrieval.
2. Gunakan model transformer (BERT/ RoBERTa/ IndoBERT/ dll) untuk
retrieval pada hasil embedding.
iv. Fungsi Retrieval
def retrieve(query: str, k: int = 5) -> List[case_id]:
# 1) Pre-process query
# 2) Hitung vektor query
# 3) Hitung cosine‐similarity dengan semua case vectors
# 4) Kembalikan top-k case_id
v. Pengujian Awal:
1. Siapkan 5–10 query uji beserta ground-truth case_id.
2. Simpan di /data/eval/queries.json.
vi. Output:
1. Script 03_retrieval.py / notebook tahap retrieval
2. Fungsi retrieve() teruji
3. File /data/eval/queries.json
d. Tahap 4 : Case Solution Reuse
Tujuan: Gunakan putusan lama sebagai dasar pencarian untuk kasus baru
Langkah Kerja:
i. Ekstrak Solusi:
1. Dari kasus top-k, ambil amar putusan atau ringkasan dakwaan.
2. Simpan di struktur: {case_id: solusi_text}
ii. Algoritma Prediksi
1. Majority vote: pilih solusi yang paling banyak muncul.
2. Weighted similarity: bobot = skor similarity.
iii. Implementasi Fungsi
def predict_outcome(query: str) -> str:
top_k = retrieve(query, k=5)
solutions = [case_solutions[c] for c in top_k]
# Terapkan voting / weighting → pilih satu ringkasan
return predicted_solution
iv. Demo Manual
1. Siapkan 5 contoh kasus baru → jalankan predict_outcome() → bandingkan
dengan putusan sebenarnya.
v. Output:
1. Script 04_predict.py / notebook
2. File /data/results/predictions.csv berisi:
query_id predicted_solution top_5_case_ids
e. Tahap 5: Model Evaluation
Tujuan: Ukur dan analisis performa retrieval & prediksi
Langkah Kerja:
i. Evaluasi Retrieval
1. Ukur Accuracy, Precision, Recall, F1-score.
2. Gunakan sklearn.metrics.

---

### FAKULTAS TEKNIK
INFORMATIKA
informatika.umm.ac.id | informatika@umm.ac.id
3. Implementasi di 05_evaluation.py:
def eval_retrieval(queries, ground_truth, k):
# loop setiap query → hitung metrics
ii. Visualisasi & Laporan
1. Tabel metrik per model (Misal: TF-IDF vs. BERT vs. SVM).
2. Plot bar chart performance (opsional).
3. Diskusikan kasus kegagalan (error analysis).
iii. Output:
1. File /data/eval/retrieval_metrics.csv
2. File /data/eval/prediction_metrics.csv
3. Bagian evaluasi lengkap di laporan
4.
5. Timeline Proyek
Proyek CBR dapat mulai dikerjakan pasca UTS
Minggu Kegiatan
1 Seleksi dan pembangungan Case Base
2 Ekstraksi metadata & fitur serta Case Representation
3 
Case Retrieval & Evaluasi hasil retrieval (Accuracy, Precision, Recall, F1-
Score) serta perbaikan hasil evaluasi retrieval
Petunjuk Pengumpulan:
Mahasiswa wajib mengupload Code Repository dengan komposisi sebagai berikut:
i. Platform: GitHub (dengan akses public untuk dapat dinilai).
ii. Upload link repository Github melalui LMS.
iii. Struktur:
/data/ # Raw & processed data
/notebooks/ # Jupyter notebooks per tahap
README.md # Petunjuk instalasi & eksekusi
iv. Konten README.md:
1. Bahasa: Inggris atau Indonesia
2. Detail: cara instal (requirements.txt), cara menjalankan pipeline end-to-end, contoh perintah,
dll.
Tidak mengerjakan tugas = Nilai D
Rubrik Penilaian:
No Komponen Penilaian Bobot 85-100 80-84 70-79 60-69
1 Membangun Case
Base (Data
Acquisition &
Preprocessing)
15% ≥30 dokumen,
preprocessing
lengkap,
bersih,
terstruktur,
siap
digunakan
≥30 dokumen,
preprocessing
baik,
kekurangan
minor
≥30 dokumen,
preprocessing
sebagian
berhasil
Dokumen
kurang dari
ketentuan
atau
preprocessing
kurang
2 Case Representation
(Metadata & Feature
Engineering)
20% Metadata
lengkap, fitur
relevan,
tersimpan
terstruktur
(.csv/.json),
siap
digunakan
model
Metadata dan
fitur sebagian
besar lengkap
Metadata
tersedia tetapi
feature
terbatas
Metadata
tidak lengkap
atau tidak
optimal
3 Case Retrieval (Model
dan Fungsi Retrieval)
25% Model
retrieval
berjalan
sempurna,
Model berjalan
baik dengan
kekurangan
minor
Model
berjalan tetapi
performa
terbatas
Model
retrieval
tidak stabil
atau hanya

---

### FAKULTAS TEKNIK
INFORMATIKA
informatika.umm.ac.id | informatika@umm.ac.id
fungsi
retrieve()
benar, hasil
relevan dan
optimal
sebagian
berhasil
4 Case Solution Reuse
(Prediksi Solusi)
15% Fungsi
prediksi
berjalan benar,
solusi
dihasilkan
logis dan
konsisten
Fungsi
berjalan
dengan
kekurangan
minor
Fungsi
berjalan tetapi
kurang
optimal
Fungsi
terbatas atau
tidak
konsisten
5 Model Evaluation dan
Analisis
15% Evaluasi
lengkap
(Accuracy,
Precision,
Recall, F1),
analisis
mendalam dan
kritis
Evaluasi
lengkap,
analisis cukup
baik
Evaluasi
tersedia tetapi
analisis
terbatas
Evaluasi
minimal
6 Repository GitHub
dan Dokumentasi
10% Repository
lengkap,
struktur jelas,
README
lengkap, dapat
direplikasi
Repository
lengkap
dengan
kekurangan
minor
Repository
tersedia tetapi
kurang rapi
Repository
tidak lengkap
1 Membangun Case
Base (Data
Acquisition &
Preprocessing)
15% ≥30 dokumen,
preprocessing
lengkap,
bersih,
terstruktur,
siap
digunakan
≥30 dokumen,
preprocessing
baik,
kekurangan
minor
≥30 dokumen,
preprocessing
sebagian
berhasil
Dokumen
kurang dari
ketentuan
atau
preprocessing
kurang