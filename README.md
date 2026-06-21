# CBR Putusan Pencurian Pengadilan Negeri Malang

Sistem **Case-Based Reasoning (CBR)** untuk analisis putusan pengadilan tindak pidana pencurian (Pasal 362 vs Pasal 363 KUHP) dari Pengadilan Negeri Malang.

Sistem ini mengimplementasikan siklus CBR lengkap: **Retrieve → Reuse → Revise → Retain**, menggunakan dua pendekatan retrieval (TF-IDF dan IndoBERT).

## Struktur Proyek

```
├── data/
│   ├── raw/                    # PDF dan TXT putusan mentah
│   │   ├── pasal_362/          # 30 putusan Pasal 362
│   │   └── pasal_363/          # 30 putusan Pasal 363
│   ├── processed/
│   │   ├── clean/              # Teks putusan setelah cleaning
│   │   ├── cases.csv           # Data kasus terstruktur
│   │   └── cases.json          # Data kasus terstruktur (JSON)
│   ├── eval/
│   │   ├── queries.json        # Query uji beserta ground-truth
│   │   ├── retrieval_metrics.csv
│   │   ├── prediction_metrics.csv
│   │   └── evaluation_chart.png
│   └── results/
│       └── predictions.csv     # Hasil prediksi
├── notebooks/
│   ├── 01_membangun_case_base.ipynb    # Tahap 1: Scraping, PDF to Text, Cleaning
│   ├── 02_case_representation.ipynb    # Tahap 2: Ekstraksi Metadata & Fitur
│   ├── 03_case_retrieval.ipynb         # Tahap 3: TF-IDF & IndoBERT Retrieval
│   ├── 04_case_solution_reuse.ipynb    # Tahap 4: Prediksi Solusi (Voting)
│   └── 05_model_evaluation.ipynb       # Tahap 5: Evaluasi & Analisis
├── models/                     # Model terlatih (TF-IDF, SVM, embeddings)
├── logs/                       # Log proses scraping, cleaning, retrieval
├── app.py                      # Antarmuka web Streamlit
├── requirements.txt
└── README.md
```

## Instalasi

Pastikan **Python 3.9+** sudah terinstall.

```bash
pip install -r requirements.txt
```

### Dependensi Utama

| Library | Fungsi |
|---|---|
| `pandas`, `numpy`, `scikit-learn` | Data processing, TF-IDF, SVM, Naive Bayes |
| `transformers`, `torch` | IndoBERT embedding |
| `pdfminer.six` | Konversi PDF ke teks |
| `beautifulsoup4`, `requests` | Web scraping |
| `streamlit` | Antarmuka web |
| `matplotlib` | Visualisasi chart evaluasi |

## Cara Menjalankan

### Opsi 1: Menjalankan via Notebook (Direkomendasikan)

Buka folder `notebooks/` dan jalankan notebook secara berurutan:

1. `01_membangun_case_base.ipynb` — Scraping, PDF to Text, Cleaning
2. `02_case_representation.ipynb` — Ekstraksi metadata dan fitur
3. `03_case_retrieval.ipynb` — Training model retrieval (TF-IDF + IndoBERT)
4. `04_case_solution_reuse.ipynb` — Prediksi solusi menggunakan voting
5. `05_model_evaluation.ipynb` — Evaluasi performa dan analisis kegagalan

```bash
cd notebooks
jupyter notebook
```

### Opsi 2: Menjalankan via Script Python

```bash
python 01_pdf_to_text.py
python 02_cleaning.py
python 03_representation.py
python 04_retrieval.py
python 05_predict.py
python 06_evaluation.py
```

### Menjalankan Antarmuka Web (Streamlit)

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`.

Fitur antarmuka web:
- **Tab 1**: Prediksi pasal dan vonis untuk kasus baru (paste teks putusan)
- **Tab 2**: Pengujian dengan contoh kasus dari dataset
- **Tab 3**: Visualisasi metrik evaluasi model

## Metodologi

### Pendekatan Retrieval

| Pendekatan | Model | Metode |
|---|---|---|
| TF-IDF | SVM, Naive Bayes | Klasifikasi langsung |
| TF-IDF | Cosine Similarity (k=5) | Retrieval top-k + voting |
| IndoBERT | Cosine Similarity (k=5) | Retrieval top-k + voting |

### Algoritma Prediksi (Solution Reuse)

- **Majority Vote**: memilih label yang paling sering muncul di top-k
- **Weighted Vote**: bobot = similarity^5 (kasus paling mirip mendominasi)

## Hasil Evaluasi

### Retrieval (label_pasal)

| Model | Accuracy | F1-Score |
|---|---|---|
| TF-IDF + SVM | **0.9167** | **0.9161** |
| TF-IDF + NB | 0.8333 | 0.8333 |
| TF-IDF + Cosine (k=5) | 0.8333 | 0.8333 |
| IndoBERT + Cosine (k=5) | 0.9167 | 0.9161 |

### Prediction (label_pasal)

| Model | Accuracy | F1-Score |
|---|---|---|
| TF-IDF + WeightedVote | **0.90** | **0.8990** |
| IndoBERT + WeightedVote | 0.60 | 0.5833 |

### Temuan

1. **TF-IDF + SVM** mencapai F1 tertinggi (0.9161) untuk klasifikasi pasal karena teks hukum mengandung keyword spesifik (misal: "malam hari", "merusak kunci" untuk Pasal 363)
2. **Weighted Vote** (sim^5) lebih akurat daripada Majority Vote karena kasus paling mirip langsung mendominasi prediksi
3. IndoBERT memerlukan fine-tuning untuk domain hukum Indonesia agar performanya optimal
