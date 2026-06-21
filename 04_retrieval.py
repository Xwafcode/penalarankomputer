"""
Tahap 3 — Case Retrieval
=========================
Input  : data/processed/cases.json
Output :
    - data/eval/queries.json
    - data/eval/retrieval_metrics.csv
    - models/tfidf_vectorizer.pkl
    - models/svm_pasal.pkl, nb_pasal.pkl
    - models/svm_vonis.pkl, nb_vonis.pkl
    - models/indobert_embeddings.npy

Jalankan:
    python 04_retrieval.py
"""

import json
import logging
import pickle
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

DATA_PATH = Path("data/processed/cases.json")
EVAL_DIR = Path("data/eval")
MODEL_DIR = Path("models")
LOG_DIR = Path("logs")

for d in [EVAL_DIR, MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "retrieval.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

_tfidf_vectorizer = None
_tfidf_case_vectors = None
_tfidf_case_ids = None
_tfidf_case_labels = None

_bert_tokenizer = None
_bert_model = None
_bert_case_embeddings = None
_bert_case_ids = None
_bert_case_labels = None


def load_cases():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_text(text):
    import re
    text = text.lower().strip()
    text = re.sub(r"disclaimer.*?ext\.318\)", "", text, flags=re.DOTALL)
    text = re.sub(r"halaman \d+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mean_pooling(model_output, attention_mask):
    import torch
    token_embeddings = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask_expanded, 1)
    clamped = torch.clamp(mask_expanded.sum(1), min=1e-9)
    return summed / clamped


def get_indobert_embeddings(texts, tokenizer, model, batch_size=4):
    import torch
    device = get_device()
    model = model.to(device)
    model.eval()
    
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embs = []
        
        for text in batch:
            # Tokenize without truncation to get all tokens
            tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            
            chunk_size = 510  # 512 - 2 (for CLS and SEP)
            chunks = [tokens[j:j+chunk_size] for j in range(0, len(tokens), chunk_size)]
            
            chunk_embs = []
            for chunk in chunks:
                # Add [CLS] (2) and [SEP] (3) for IndoBERT
                input_ids = torch.cat([
                    torch.tensor([tokenizer.cls_token_id]), 
                    chunk, 
                    torch.tensor([tokenizer.sep_token_id])
                ]).unsqueeze(0).to(device)
                
                attention_mask = torch.ones_like(input_ids).to(device)
                
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Mean pooling for this chunk
                pooled = mean_pooling(outputs, attention_mask)
                chunk_embs.append(pooled)
            
            # Average embeddings of all chunks for this document
            if chunk_embs:
                doc_emb = torch.stack(chunk_embs).mean(dim=0)
            else:
                # Fallback for empty text
                doc_emb = torch.zeros((1, model.config.hidden_size)).to(device)
                
            batch_embs.append(doc_emb.cpu().numpy())
            
        all_emb.append(np.vstack(batch_embs))
        
        if (i // batch_size) % 5 == 0:
            log.info(f"  Embedding batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
            
    return np.vstack(all_emb)


def retrieval_predict(query_vectors, train_vectors, train_labels, k=5):
    predictions = []
    all_top_k = []
    for i in range(query_vectors.shape[0]):
        if hasattr(query_vectors, "toarray"):
            q = query_vectors[i : i + 1]
        else:
            q = query_vectors[i : i + 1]
        sims = cosine_similarity(q, train_vectors).flatten()
        top_k_idx = sims.argsort()[::-1][:k]
        top_k_labels = [train_labels[j] for j in top_k_idx]
        vote = Counter(top_k_labels).most_common(1)[0][0]
        predictions.append(vote)
        all_top_k.append(top_k_idx.tolist())
    return predictions, all_top_k


def evaluate_model(y_true, y_pred, model_name, label_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    log.info(f"\n{'=' * 50}")
    log.info(f"Model: {model_name} | Label: {label_name}")
    log.info(f"  Accuracy : {acc:.4f}")
    log.info(f"  Precision: {prec:.4f}")
    log.info(f"  Recall   : {rec:.4f}")
    log.info(f"  F1-Score : {f1:.4f}")
    log.info(f"\n{classification_report(y_true, y_pred, zero_division=0)}")

    return {
        "model": model_name,
        "label": label_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
    }


def retrieve(query, k=5, method="tfidf"):
    query = preprocess_text(query)

    if method == "tfidf":
        if _tfidf_vectorizer is None:
            raise RuntimeError("TF-IDF model belum di-load. Jalankan main() terlebih dahulu.")
        q_vec = _tfidf_vectorizer.transform([query])
        sims = cosine_similarity(q_vec, _tfidf_case_vectors).flatten()
        top_k_idx = sims.argsort()[::-1][:k]
        return [
            {
                "case_id": _tfidf_case_ids[i],
                "label": _tfidf_case_labels[i],
                "similarity": round(float(sims[i]), 4),
            }
            for i in top_k_idx
        ]

    elif method == "indobert":
        if _bert_model is None:
            raise RuntimeError("IndoBERT model belum di-load. Jalankan main() terlebih dahulu.")
        q_emb = get_indobert_embeddings([query], _bert_tokenizer, _bert_model)
        sims = cosine_similarity(q_emb, _bert_case_embeddings).flatten()
        top_k_idx = sims.argsort()[::-1][:k]
        return [
            {
                "case_id": _bert_case_ids[i],
                "label": _bert_case_labels[i],
                "similarity": round(float(sims[i]), 4),
            }
            for i in top_k_idx
        ]

    else:
        raise ValueError(f"Method tidak dikenal: {method}")


def main():
    global _tfidf_vectorizer, _tfidf_case_vectors, _tfidf_case_ids, _tfidf_case_labels
    global _bert_tokenizer, _bert_model, _bert_case_embeddings, _bert_case_ids, _bert_case_labels

    log.info("=" * 60)
    log.info("TAHAP 3 — CASE RETRIEVAL")
    log.info("=" * 60)

    cases = load_cases()
    log.info(f"Total kasus: {len(cases)}")

    for c in cases:
        c["text_processed"] = preprocess_text(c["text_full"])

    labels_pasal = [c["label_pasal"] for c in cases]
    train_cases, test_cases = train_test_split(
        cases, test_size=0.2, stratify=labels_pasal, random_state=42
    )
    log.info(f"Train: {len(train_cases)}, Test: {len(test_cases)}")

    train_pasal_dist = Counter([c["label_pasal"] for c in train_cases])
    test_pasal_dist = Counter([c["label_pasal"] for c in test_cases])
    log.info(f"  Train distribusi pasal: {dict(train_pasal_dist)}")
    log.info(f"  Test distribusi pasal : {dict(test_pasal_dist)}")

    train_texts = [c["text_processed"] for c in train_cases]
    test_texts = [c["text_processed"] for c in test_cases]
    train_labels_pasal = [c["label_pasal"] for c in train_cases]
    test_labels_pasal = [c["label_pasal"] for c in test_cases]

    train_vonis_cases = [c for c in train_cases if c["label_vonis"] not in ["tidak_diketahui"]]
    test_vonis_cases = [c for c in test_cases if c["label_vonis"] not in ["tidak_diketahui"]]
    has_vonis_data = len(train_vonis_cases) >= 4 and len(test_vonis_cases) >= 2

    if has_vonis_data:
        log.info(f"  Vonis data: train={len(train_vonis_cases)}, test={len(test_vonis_cases)}")
        train_vonis_texts = [c["text_processed"] for c in train_vonis_cases]
        test_vonis_texts = [c["text_processed"] for c in test_vonis_cases]
        train_labels_vonis = [c["label_vonis"] for c in train_vonis_cases]
        test_labels_vonis = [c["label_vonis"] for c in test_vonis_cases]
    else:
        log.warning("Data vonis tidak cukup untuk evaluasi terpisah")

    all_metrics = []

    # ═══════════════════════════════════════════════════════════
    #  PENDEKATAN 1: TF-IDF + SVM / Naive Bayes
    # ═══════════════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("PENDEKATAN 1: TF-IDF + Machine Learning")
    log.info("=" * 60)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    log.info(f"TF-IDF shape: train={X_train.shape}, test={X_test.shape}")

    all_texts = [c["text_processed"] for c in cases]
    _tfidf_vectorizer = vectorizer
    _tfidf_case_vectors = vectorizer.transform(all_texts)
    _tfidf_case_ids = [c["case_id"] for c in cases]
    _tfidf_case_labels = [c["label_pasal"] for c in cases]

    with open(MODEL_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    log.info("TF-IDF vectorizer disimpan")

    # ── SVM (label_pasal) ──
    log.info("\n--- SVM Classification (label_pasal) ---")
    svm_pasal = LinearSVC(max_iter=10000, random_state=42, C=1.0)
    svm_pasal.fit(X_train, train_labels_pasal)
    svm_pred = svm_pasal.predict(X_test)
    m = evaluate_model(test_labels_pasal, svm_pred, "TF-IDF + SVM", "label_pasal")
    all_metrics.append(m)
    with open(MODEL_DIR / "svm_pasal.pkl", "wb") as f:
        pickle.dump(svm_pasal, f)

    # ── Naive Bayes (label_pasal) ──
    log.info("\n--- Naive Bayes Classification (label_pasal) ---")
    nb_pasal = MultinomialNB(alpha=1.0)
    nb_pasal.fit(X_train, train_labels_pasal)
    nb_pred = nb_pasal.predict(X_test)
    m = evaluate_model(test_labels_pasal, nb_pred, "TF-IDF + NB", "label_pasal")
    all_metrics.append(m)
    with open(MODEL_DIR / "nb_pasal.pkl", "wb") as f:
        pickle.dump(nb_pasal, f)

    # ── TF-IDF Cosine Retrieval (label_pasal) ──
    log.info("\n--- TF-IDF Cosine Retrieval (label_pasal, k=5) ---")
    tfidf_ret_pred, _ = retrieval_predict(X_test, X_train, train_labels_pasal, k=5)
    m = evaluate_model(test_labels_pasal, tfidf_ret_pred, "TF-IDF + Cosine (k=5)", "label_pasal")
    all_metrics.append(m)

    # ═══════════════════════════════════════════════════════════
    #  PENDEKATAN 2: IndoBERT Embedding
    # ═══════════════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("PENDEKATAN 2: IndoBERT Embedding")
    log.info("=" * 60)

    MODEL_NAME = "indobenchmark/indobert-base-p1"
    log.info(f"Loading model: {MODEL_NAME}...")

    from transformers import AutoModel, AutoTokenizer
    import torch

    device = get_device()
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    _bert_tokenizer = tokenizer
    _bert_model = model
    log.info("Model loaded")

    log.info("Generating train embeddings...")
    train_emb = get_indobert_embeddings(train_texts, tokenizer, model)
    log.info(f"Train embeddings: {train_emb.shape}")

    log.info("Generating test embeddings...")
    test_emb = get_indobert_embeddings(test_texts, tokenizer, model)
    log.info(f"Test embeddings: {test_emb.shape}")

    log.info("Generating all case embeddings...")
    all_emb = get_indobert_embeddings(all_texts, tokenizer, model)
    _bert_case_embeddings = all_emb
    _bert_case_ids = [c["case_id"] for c in cases]
    _bert_case_labels = [c["label_pasal"] for c in cases]
    np.save(MODEL_DIR / "indobert_embeddings.npy", all_emb)
    log.info(f"All embeddings: {all_emb.shape}, disimpan")

    # ── IndoBERT Cosine Retrieval (label_pasal) ──
    log.info("\n--- IndoBERT Cosine Retrieval (label_pasal, k=5) ---")
    bert_ret_pred, _ = retrieval_predict(test_emb, train_emb, train_labels_pasal, k=5)
    m = evaluate_model(test_labels_pasal, bert_ret_pred, "IndoBERT + Cosine (k=5)", "label_pasal")
    all_metrics.append(m)

    # ── IndoBERT (label_vonis) ──
    if has_vonis_data:
        log.info("\n--- IndoBERT Cosine Retrieval (label_vonis, k=5) ---")
        pass

    # ═══════════════════════════════════════════════════════════
    #  GENERATE queries.json
    # ═══════════════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("GENERATING queries.json")
    log.info("=" * 60)

    queries = []
    for i, tc in enumerate(test_cases[:10]):
        tfidf_results = retrieve(tc["text_processed"], k=5, method="tfidf")
        bert_results = retrieve(tc["text_processed"], k=5, method="indobert")

        q = {
            "query_id": f"q{i + 1:02d}",
            "source_case_id": tc["case_id"],
            "ground_truth_label_pasal": tc["label_pasal"],
            "ground_truth_label_vonis": tc["label_vonis"],
            "query_text": tc["text_processed"][:500],
            "tfidf_top5": tfidf_results,
            "indobert_top5": bert_results,
        }
        queries.append(q)
        log.info(
            f"  {q['query_id']}: {tc['case_id']} ({tc['label_pasal']}) → "
            f"TF-IDF top1={tfidf_results[0]['case_id']} | "
            f"BERT top1={bert_results[0]['case_id']}"
        )

    with open(EVAL_DIR / "queries.json", "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    log.info(f"queries.json disimpan: {len(queries)} queries")

    # ═══════════════════════════════════════════════════════════
    #  SAVE METRICS
    # ═══════════════════════════════════════════════════════════
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(EVAL_DIR / "retrieval_metrics.csv", index=False)
    log.info(f"retrieval_metrics.csv disimpan")

    print("\n" + "=" * 70)
    print("RETRIEVAL METRICS SUMMARY")
    print("=" * 70)
    print(metrics_df.to_string(index=False))
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════
    #  DEMO retrieve()
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("DEMO: retrieve() function")
    print("=" * 70)

    demo_query = test_cases[0]["text_processed"][:300]
    print(f"\nQuery (300 char): {demo_query[:100]}...")
    print(f"Ground truth: {test_cases[0]['label_pasal']}")

    print("\n--- TF-IDF Retrieval ---")
    results = retrieve(demo_query, k=5, method="tfidf")
    for r in results:
        print(f"  {r['case_id']} | label={r['label']} | sim={r['similarity']}")

    print("\n--- IndoBERT Retrieval ---")
    results = retrieve(demo_query, k=5, method="indobert")
    for r in results:
        print(f"  {r['case_id']} | label={r['label']} | sim={r['similarity']}")

    log.info("\n" + "=" * 60)
    log.info("TAHAP 3 SELESAI")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
