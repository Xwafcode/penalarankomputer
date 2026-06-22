import json
import pickle
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

DATA_DIR   = Path("data/processed")
EVAL_DIR   = Path("data/eval")
MODEL_DIR  = Path("models")
RESULT_DIR = Path("data/results")

st.set_page_config(
    page_title="CBR Putusan Pencurian - PN Malang",
    page_icon="a]",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #e94560;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .metric-card h1 {
        margin: 0.3rem 0 0 0;
        font-size: 1.8rem;
    }
    .result-box {
        background: #f8f9fa;
        border-left: 4px solid #4CAF50;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .result-box-warn {
        background: #fff8e1;
        border-left: 4px solid #ff9800;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .sim-bar {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        margin-top: 4px;
    }
    .case-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTextArea textarea {
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_cases():
    with open(DATA_DIR / "cases.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_tfidf():
    with open(MODEL_DIR / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    cases = load_cases()
    texts = [clean_text(c["text_full"]) for c in cases]
    vectors = vectorizer.transform(texts)
    return vectorizer, vectors


@st.cache_data
def load_embeddings():
    return np.load(MODEL_DIR / "indobert_embeddings.npy")


@st.cache_resource
def load_indobert():
    import torch
    from transformers import AutoModel, AutoTokenizer
    MODEL_NAME = "indobenchmark/indobert-base-p1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return tokenizer, model, device


INDO_COMMON_WORDS = {
    "ada", "adalah", "adanya", "agar", "akan", "akibat", "akhir",
    "alasan", "amar", "anak", "antara", "apa", "apabila",
    "atas", "atau", "ayat",
    "bagi", "bagian", "bahwa", "baik", "barang", "baru",
    "batu", "beberapa", "belum", "benar", "berapa",
    "berdasarkan", "berdiri", "berikut", "berkas", "bermaksud",
    "bersalah", "bersama", "berupa", "besar", "biasa", "biaya",
    "bila", "bisa", "boleh", "buah", "bukan", "bulan",
    "cara", "cukup",
    "dahulu", "dalam", "dan", "dapat", "dari", "datang",
    "dengan", "demi", "demikian", "demikianlah",
    "denda", "depan", "desa", "dimana", "dimaksud",
    "diatur", "diancam", "didakwa", "didakwakan", "dijatuhkan",
    "dinyatakan", "diperoleh", "dipersidangan",
    "duduk", "dusun",
    "empat", "enam",
    "fakta",
    "hak", "hal", "halaman", "hanya", "hari", "hakim",
    "harus", "haruslah", "hasil", "hendak", "hukum", "hukuman",
    "ini", "islam", "itu",
    "jadi", "jalan", "jaksa", "jam", "jenis",
    "jika", "juga", "jumlah",
    "kabupaten", "kalau", "kami", "karena", "karenanya",
    "kasus", "kata", "keadilan", "kebangsaan",
    "kecamatan", "kedua", "kelamin",
    "kelurahan", "keluarga", "kemudian", "kepada",
    "kepadanya", "kepentingan", "keperluan",
    "kerja", "kerugian", "kesadaran",
    "keterangan", "ketentuan", "ketiga", "ketika",
    "ketua", "ketuhanan",
    "kewajiban", "khususnya",
    "korban", "kota", "kuasa", "kunci", "kurang",
    "lagi", "lahir", "lain", "lalu", "lama",
    "langsung", "lebih", "lengkap", "lima",
    "maha", "maka", "maksud", "malang",
    "mampu", "mana", "masa", "masing",
    "masih", "masuk", "masyarakat", "maupun",
    "majelis", "meja",
    "melakukan", "melanggar", "melawan",
    "melepaskan", "memang", "membebaskan",
    "membawa", "membayar", "memberi", "memberikan",
    "memenuhi", "memiliki",
    "memohon", "mempertimbangkan", "memperhatikan",
    "memutuskan",
    "mendapat", "mendekati", "mendorong",
    "mendengar", "menerangkan",
    "mengadili", "mengajukan", "mengakui", "mengalami",
    "mengambil", "mengenai", "mengetahui",
    "menggunakan", "mengingat", "menguasai",
    "menimbang", "menikmati", "menjadi",
    "menjatuhkan", "menuju",
    "menurut", "menyatakan", "menyesal", "menyesali",
    "meresahkan", "merupakan", "meskipun",
    "milik", "miliki", "motor", "mulai",
    "nama", "namun", "negeri", "nomor",
    "orang", "oleh",
    "pada", "paling", "panitera", "para", "parkir",
    "pasal", "pekerjaan",
    "pelaku", "pemaaf", "pembenar",
    "pembacaan", "pemeriksaan",
    "pencurian", "pendapat", "penetapan",
    "pengganti", "pengetahuan",
    "penjara", "penjeraan",
    "penuntut", "penunjukan",
    "penyesuaian", "penyidik",
    "peradilan", "peraturan",
    "perbuatan", "perbuatannya",
    "perkara", "perlu", "pernah", "persidangan",
    "pertama", "pertimbangan", "perundang",
    "pidana", "pihak", "pokoknya",
    "pula", "pukul", "putusan",
    "rasa", "rekaman", "republik", "resto",
    "ringan", "rupiah",
    "saat", "saja", "saksi", "salah",
    "sama", "sampai", "sangat", "sarana",
    "satu", "sebagai", "sebagaimana",
    "sebagian", "sebelum", "sebelumnya",
    "secara", "sedang", "sedangkan",
    "sehingga", "sejak", "sejumlah", "sekira", "sekitar",
    "selain", "selaku", "selama", "selanjutnya",
    "seluruh", "seluruhnya",
    "semua", "sendiri", "sependapat",
    "sepeda", "sepengetahuan",
    "seperti", "sepatutnya",
    "serta", "sesuai", "sesuatu",
    "setelah", "setiap", "setimpal",
    "sifat", "sopan",
    "suatu", "sudah", "sumpah", "supaya", "surat",
    "tahun", "tanggal", "tanpa", "telah",
    "tempat", "tentang", "tentunya",
    "terbukti", "terhadap",
    "terdakwa", "terlebih", "termasuk", "ternyata",
    "tersebut", "terungkap",
    "tetap", "tetapi",
    "tidak", "tindak", "tinggal",
    "tujuan", "tulang", "tuntutan",
    "tujuh", "tiga", "tingkat",
    "umum", "umur", "undang",
    "unsur", "untuk", "utama",
    "wajib", "waktu", "walaupun", "warna", "wib",
}


def _remove_page_boundaries(raw_text):
    import re
    lines = raw_text.split('\n')
    content_lines = []
    in_disclaimer = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower() == 'disclaimer':
            in_disclaimer = True
            continue
        if in_disclaimer:
            if 'ext.318)' in stripped.lower():
                in_disclaimer = False
            continue
        if re.match(r'^halaman\s+\d+$', stripped, re.IGNORECASE):
            continue
        stripped = re.sub(
            r'\s*putusan\s+nomor\s+\d+/pid\.\w+/\d{4}/pn\s+\w+\s*$',
            '', stripped, flags=re.IGNORECASE
        )
        if stripped:
            content_lines.append(stripped)

    return ' '.join(content_lines)


def _split_merged_words(text, word_set, min_token_len=6, min_part_len=3):
    import re
    tokens = text.split()
    result = []
    for token in tokens:
        alpha_only = re.sub(r'[^a-z]', '', token)
        if len(alpha_only) <= min_token_len or alpha_only in word_set:
            result.append(token)
            continue
        best_split = None
        best_score = -1
        for i in range(min_part_len, len(alpha_only) - min_part_len + 1):
            left = alpha_only[:i]
            right = alpha_only[i:]
            if left in word_set and right in word_set:
                score = len(left) + len(right) + min(len(left), len(right))
                if score > best_score:
                    best_score = score
                    best_split = (left, right)
        if best_split:
            result.append(best_split[0])
            sub = _split_merged_words(best_split[1], word_set, min_token_len, min_part_len)
            result.append(sub)
        else:
            result.append(token)
    return ' '.join(result)


def clean_text(text):
    import re
    text = _remove_page_boundaries(text)
    text = text.lower()

    def fix_spaced_chars(m):
        return m.group(0).replace(' ', '')
    text = re.sub(r'(?<!\w)([a-z] ){3,}[a-z](?!\w)', fix_spaced_chars, text)

    text = re.sub(r';', '; ', text)
    text = re.sub(r'(?<=[a-z])\.(?=[a-z])', '. ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    text = _split_merged_words(text, INDO_COMMON_WORDS)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def get_embedding(text, tokenizer, model, device):
    import torch
    # Tokenize completely without truncation
    tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    
    chunk_size = 510  # Max length 512 minus 2 (for CLS and SEP)
    chunks = [tokens[j:j+chunk_size] for j in range(0, len(tokens), chunk_size)]
    
    chunk_embs = []
    for chunk in chunks:
        input_ids = torch.cat([
            torch.tensor([tokenizer.cls_token_id]), 
            chunk, 
            torch.tensor([tokenizer.sep_token_id])
        ]).unsqueeze(0).to(device)
        
        attention_mask = torch.ones_like(input_ids).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
        mask = attention_mask.unsqueeze(-1).float()
        emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        chunk_embs.append(emb)
        
    if chunk_embs:
        return torch.stack(chunk_embs).mean(dim=0).cpu().numpy()[0]
    else:
        return np.zeros(model.config.hidden_size)


def retrieve_tfidf(query, vectorizer, all_vectors, cases, k=5):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, all_vectors)[0]
    top_k = np.argsort(sims)[::-1][:k]
    return [(cases[i], float(sims[i])) for i in top_k]


def retrieve_bert(query, tokenizer, model, device, all_embeddings, cases, k=5):
    q_emb = get_embedding(query, tokenizer, model, device).reshape(1, -1)
    sims = cosine_similarity(q_emb, all_embeddings)[0]
    top_k = np.argsort(sims)[::-1][:k]
    return [(cases[i], float(sims[i])) for i in top_k]


def majority_vote(results, key):
    values = [c.get(key, "") for c, _ in results if c.get(key)]
    if not values:
        return "tidak_diketahui"
    return Counter(values).most_common(1)[0][0]


def weighted_vote(results, key):
    scores = {}
    for c, sim in results:
        val = c.get(key, "")
        if val:
            scores[val] = scores.get(val, 0.0) + (sim ** 5)
    if not scores:
        return "tidak_diketahui"
    return max(scores, key=scores.get)


def predict_vonis_bulan(results, voting_method):
    if not results:
        return "?"
    if voting_method == "Majority Vote":
        # Average of all k results
        total_months = sum(c.get("vonis_bulan", 0) for c, _ in results)
        return f"{int(round(total_months / len(results)))} bulan"
    else:
        # Weighted average using sim^5
        total_weight = sum((sim ** 5) for _, sim in results)
        if total_weight == 0:
            return "?"
        weighted_months = sum(c.get("vonis_bulan", 0) * (sim ** 5) for c, sim in results)
        return f"{int(round(weighted_months / total_weight))} bulan"


def render_case_card(case, sim, rank):
    pasal_color = "#4CAF50" if case["label_pasal"] == "pasal_362" else "#2196F3"
    vonis_bulan = case.get('vonis_bulan', 0)
    # Gradien warna untuk bulan (semakin lama semakin merah)
    if vonis_bulan < 12:
        vonis_color = "#8BC34A"
    elif vonis_bulan <= 36:
        vonis_color = "#FF9800"
    else:
        vonis_color = "#F44336"
        
    bar_width = max(int(sim * 100), 5)

    st.markdown(f"""
    <div class="case-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <strong>#{rank}</strong> &nbsp;
                <code>{case['case_id']}</code> &nbsp;
                <span style="background:{pasal_color};color:white;padding:2px 8px;border-radius:12px;font-size:0.75rem;">
                    {case['label_pasal']}
                </span>
                <span style="background:{vonis_color};color:white;padding:2px 8px;border-radius:12px;font-size:0.75rem;margin-left:4px;">
                    vonis: {vonis_bulan} bln
                </span>
            </div>
            <div style="font-weight:600;font-size:1.1rem;color:#333;">
                {sim:.4f}
            </div>
        </div>
        <div style="margin-top:6px;">
            <div style="background:#e0e0e0;height:8px;border-radius:4px;">
                <div style="width:{bar_width}%;height:8px;border-radius:4px;background:linear-gradient(90deg,#667eea,#764ba2);"></div>
            </div>
        </div>
        <div style="margin-top:8px;font-size:0.8rem;color:#666;">
            <strong>No Perkara:</strong> {case.get('no_perkara','-')} &nbsp;|&nbsp;
            <strong>Amar:</strong> {case.get('amar_putusan','-')}
        </div>
        <details style="margin-top:6px;">
            <summary style="cursor:pointer;font-size:0.8rem;color:#667eea;">Lihat ringkasan fakta</summary>
            <p style="font-size:0.78rem;color:#555;margin-top:4px;">
                {case.get('ringkasan_fakta','Tidak tersedia')[:500]}
            </p>
        </details>
    </div>
    """, unsafe_allow_html=True)


def page_predict():
    st.markdown('<div class="main-header">Case-Based Reasoning: Putusan Pencurian</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Sistem CBR untuk analisis putusan pidana pencurian PN Malang<br>'
                'Pasal 362 (Pencurian Biasa) vs Pasal 363 (Pencurian dengan Pemberatan)</div>',
                unsafe_allow_html=True)

    cases = load_cases()
    vectorizer, all_vectors = load_tfidf()
    all_embeddings = load_embeddings()

    with st.sidebar:
        st.header("Pengaturan")
        k = st.slider("Jumlah kasus serupa (k)", 1, 10, 5)
        method = st.radio("Metode Retrieval", ["TF-IDF", "IndoBERT", "Keduanya"], index=2)
        voting = st.radio("Metode Voting", ["Majority Vote", "Weighted Vote"], index=1)

        st.divider()
        st.header("Info Case Base")
        st.metric("Total Kasus", len(cases))
        pasal_362 = sum(1 for c in cases if c["label_pasal"] == "pasal_362")
        pasal_363 = sum(1 for c in cases if c["label_pasal"] == "pasal_363")
        col1, col2 = st.columns(2)
        col1.metric("Pasal 362", pasal_362)
        col2.metric("Pasal 363", pasal_363)

    tab1, tab2, tab3 = st.tabs(["Prediksi Kasus Baru", "Contoh Pengujian", "Metrik Evaluasi"])

    with tab1:
        st.subheader("Pilih Contoh Kasus (Opsional)")
        contoh_options = ["-- Ketik/Upload Sendiri --"]
        
        import random
        random.seed(42)
        s_363 = random.sample([c for c in cases if c['label_pasal'] == 'pasal_363'], 2)
        s_362 = random.sample([c for c in cases if c['label_pasal'] == 'pasal_362'], 2)
        samples = s_363 + s_362
        
        sample_dict = {}
        for i, s in enumerate(samples, 1):
            label = f"Contoh {i}: {s['label_pasal']} (Vonis {s.get('vonis_bulan','?')} bln)"
            contoh_options.append(label)
            sample_dict[label] = s
            
        selected_contoh = st.selectbox("Gunakan kasus dari data latih sebagai contoh:", contoh_options)
        
        default_val = ""
        if selected_contoh != "-- Ketik/Upload Sendiri --":
            s_data = sample_dict[selected_contoh]
            default_val = s_data["text_full"]
            st.info(f"**Prediksi Seharusnya:** {s_data['label_pasal']} | **Vonis:** {s_data.get('vonis_bulan','?')} bulan")
            
        st.subheader("Masukkan Teks Putusan")
        query_text = st.text_area(
            "Paste teks putusan pengadilan di sini:",
            height=250,
            value=default_val,
            placeholder="Contoh: PUTUSAN Nomor xxx/Pid.B/2024/PN Mlg DEMI KEADILAN BERDASARKAN KETUHANAN YANG MAHA ESA ..."
        )

        uploaded_file = st.file_uploader("Atau upload file .txt", type=["txt"])
        if uploaded_file is not None:
            query_text = uploaded_file.read().decode("utf-8", errors="ignore")
            st.success(f"File '{uploaded_file.name}' berhasil dimuat ({len(query_text)} karakter)")

        if st.button("Analisis Kasus", type="primary", use_container_width=True):
            if not query_text or len(query_text.strip()) < 50:
                st.error("Teks terlalu pendek. Masukkan minimal 50 karakter teks putusan.")
                return

            cleaned = clean_text(query_text)

            tfidf_results = []
            bert_results = []

            with st.spinner("Menganalisis kasus..."):
                if method in ["TF-IDF", "Keduanya"]:
                    tfidf_results = retrieve_tfidf(cleaned, vectorizer, all_vectors, cases, k)

                if method in ["IndoBERT", "Keduanya"]:
                    tokenizer, model, device = load_indobert()
                    bert_results = retrieve_bert(cleaned, tokenizer, model, device, all_embeddings, cases, k)

            vote_fn = majority_vote if voting == "Majority Vote" else weighted_vote

            col1, col2, col3, col4 = st.columns(4)
            if tfidf_results:
                pred_pasal = vote_fn(tfidf_results, "label_pasal")
                pred_vonis = predict_vonis_bulan(tfidf_results, voting)
                col1.markdown(f"""<div class="metric-card"><h3>TF-IDF - Pasal</h3><h1>{pred_pasal}</h1></div>""", unsafe_allow_html=True)
                col2.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#f093fb,#f5576c);"><h3>TF-IDF - Vonis</h3><h1>{pred_vonis}</h1></div>""", unsafe_allow_html=True)

            if bert_results:
                pred_pasal_b = vote_fn(bert_results, "label_pasal")
                pred_vonis_b = predict_vonis_bulan(bert_results, voting)
                col3.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#4facfe,#00f2fe);"><h3>IndoBERT - Pasal</h3><h1>{pred_pasal_b}</h1></div>""", unsafe_allow_html=True)
                col4.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#43e97b,#38f9d7);"><h3>IndoBERT - Vonis</h3><h1>{pred_vonis_b}</h1></div>""", unsafe_allow_html=True)

            st.divider()

            if method == "Keduanya":
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("TF-IDF Top-K")
                    for i, (case, sim) in enumerate(tfidf_results):
                        render_case_card(case, sim, i + 1)
                with c2:
                    st.subheader("IndoBERT Top-K")
                    for i, (case, sim) in enumerate(bert_results):
                        render_case_card(case, sim, i + 1)
            elif tfidf_results:
                st.subheader("TF-IDF Top-K")
                for i, (case, sim) in enumerate(tfidf_results):
                    render_case_card(case, sim, i + 1)
            elif bert_results:
                st.subheader("IndoBERT Top-K")
                for i, (case, sim) in enumerate(bert_results):
                    render_case_card(case, sim, i + 1)

    with tab2:
        st.subheader("Contoh Pengujian dengan Data dari Case Base")
        st.info("Pilih salah satu kasus dari case base untuk melihat bagaimana sistem bekerja. "
                "Kasus yang dipilih akan digunakan sebagai query dan dibandingkan dengan seluruh case base.")

        case_ids = [c["case_id"] for c in cases]
        selected_id = st.selectbox("Pilih kasus:", case_ids, index=0)
        selected_case = next(c for c in cases if c["case_id"] == selected_id)

        col1, col2 = st.columns(2)
        col1.metric("Pasal", selected_case["label_pasal"])
        col2.metric("Vonis (bulan)", f"{selected_case.get('vonis_bulan', '?')} bulan")

        with st.expander("Lihat teks putusan (500 karakter pertama)"):
            st.text(selected_case["text_full"][:500])

        if st.button("Jalankan Pencarian Serupa", type="primary"):
            cleaned = clean_text(selected_case["text_full"])

            with st.spinner("Menganalisis..."):
                tfidf_res = retrieve_tfidf(cleaned, vectorizer, all_vectors, cases, k=5)
                tokenizer, model, device = load_indobert()
                bert_res = retrieve_bert(cleaned, tokenizer, model, device, all_embeddings, cases, k=5)

            st.success(f"Ground Truth: **{selected_case['label_pasal']}** | "
                       f"Vonis: **{selected_case.get('vonis_bulan','?')} bulan**")

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("TF-IDF Top-5")
                for i, (case, sim) in enumerate(tfidf_res):
                    render_case_card(case, sim, i + 1)
            with c2:
                st.subheader("IndoBERT Top-5")
                for i, (case, sim) in enumerate(bert_res):
                    render_case_card(case, sim, i + 1)

    with tab3:
        st.subheader("Metrik Evaluasi Model")

        if (EVAL_DIR / "retrieval_metrics.csv").exists():
            df_ret = pd.read_csv(EVAL_DIR / "retrieval_metrics.csv")
            st.markdown("#### Retrieval Metrics (Tahap 3)")
            st.dataframe(df_ret, use_container_width=True, hide_index=True)

        if (EVAL_DIR / "prediction_metrics.csv").exists():
            df_pred = pd.read_csv(EVAL_DIR / "prediction_metrics.csv")
            st.markdown("#### Prediction Metrics (Tahap 4-5)")
            st.dataframe(df_pred, use_container_width=True, hide_index=True)

        if (EVAL_DIR / "evaluation_chart.png").exists():
            st.markdown("#### Grafik Perbandingan Model")
            st.image(str(EVAL_DIR / "evaluation_chart.png"), use_container_width=True)

        st.markdown("#### Ringkasan Temuan")
        st.markdown("""
        | Aspek | Temuan |
        |---|---|
        | **Model terbaik (Pasal)** | TF-IDF + SVM (F1=0.9161) |
        | **Rekomendasi** | Gunakan TF-IDF untuk teks ringkasan, perbaiki ekstraksi data untuk model IndoBERT |
        """)


if __name__ == "__main__":
    page_predict()
