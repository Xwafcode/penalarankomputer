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


def clean_text(text):
    import re
    text = text.lower().strip()
    # Hapus disclaimer MA
    text = re.sub(r"disclaimer.*?ext\.318\)", "", text, flags=re.DOTALL)
    text = re.sub(r"halaman \d+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_embedding(text, tokenizer, model, device):
    import torch
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=512, padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
    return emb.cpu().numpy()[0]


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
            scores[val] = scores.get(val, 0.0) + sim
    if not scores:
        return "tidak_diketahui"
    return max(scores, key=scores.get)


def render_case_card(case, sim, rank):
    pasal_color = "#4CAF50" if case["label_pasal"] == "pasal_362" else "#2196F3"
    vonis_map = {"ringan": "#8BC34A", "sedang": "#FF9800", "berat": "#F44336", "tidak_diketahui": "#9E9E9E"}
    vonis_color = vonis_map.get(case.get("label_vonis", ""), "#9E9E9E")
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
                    vonis: {case.get('label_vonis','?')}
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
            <strong>Vonis:</strong> {case.get('vonis_bulan','-')} bulan &nbsp;|&nbsp;
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
        st.subheader("Masukkan Teks Putusan")
        query_text = st.text_area(
            "Paste teks putusan pengadilan di sini:",
            height=250,
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
                pred_vonis = vote_fn(tfidf_results, "label_vonis")
                col1.markdown(f"""<div class="metric-card"><h3>TF-IDF - Pasal</h3><h1>{pred_pasal}</h1></div>""", unsafe_allow_html=True)
                col2.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#f093fb,#f5576c);"><h3>TF-IDF - Vonis</h3><h1>{pred_vonis}</h1></div>""", unsafe_allow_html=True)

            if bert_results:
                pred_pasal_b = vote_fn(bert_results, "label_pasal")
                pred_vonis_b = vote_fn(bert_results, "label_vonis")
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

        col1, col2, col3 = st.columns(3)
        col1.metric("Pasal", selected_case["label_pasal"])
        col2.metric("Vonis", selected_case.get("label_vonis", "?"))
        col3.metric("Vonis (bulan)", selected_case.get("vonis_bulan", "?"))

        with st.expander("Lihat teks putusan (500 karakter pertama)"):
            st.text(selected_case["text_full"][:500])

        if st.button("Jalankan Retrieval", key="demo_btn", use_container_width=True):
            cleaned = clean_text(selected_case["text_full"])

            with st.spinner("Menganalisis..."):
                tfidf_res = retrieve_tfidf(cleaned, vectorizer, all_vectors, cases, k=5)
                tokenizer, model, device = load_indobert()
                bert_res = retrieve_bert(cleaned, tokenizer, model, device, all_embeddings, cases, k=5)

            st.success(f"Ground Truth: **{selected_case['label_pasal']}** | "
                       f"Vonis: **{selected_case.get('label_vonis','?')}**")

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
        | **Model terbaik (Vonis)** | IndoBERT + Cosine (F1=0.7083) |
        | **Kelemahan utama** | Imbalance kelas vonis (sedang=33, ringan=8, berat=1) |
        | **Rekomendasi** | Augmentasi data vonis, perbaiki regex ekstraksi amar putusan |
        """)


if __name__ == "__main__":
    page_predict()
