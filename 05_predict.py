import json
import pickle
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

DATA_DIR   = Path("data/processed")
EVAL_DIR   = Path("data/eval")
MODEL_DIR  = Path("models")
RESULT_DIR = Path("data/results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

def load_cases():
    with open(DATA_DIR / "cases.json", encoding="utf-8") as f:
        return json.load(f)

def load_models():
    with open(MODEL_DIR / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    embeddings = np.load(MODEL_DIR / "indobert_embeddings.npy")
    return vectorizer, embeddings

def get_device():
    try:
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        return None

def get_indobert_embedding(text, tokenizer, model, device, max_length=512):
    import torch
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    emb  = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
    return emb.cpu().numpy()[0]

def retrieve_tfidf(query_text, vectorizer, all_vectors, cases, k=5):
    q_vec = vectorizer.transform([query_text])
    sims  = cosine_similarity(q_vec, all_vectors)[0]
    top_k = np.argsort(sims)[::-1][:k]
    return [(cases[i]["case_id"], sims[i]) for i in top_k]

def retrieve_bert(query_emb, all_embeddings, cases, k=5):
    q = query_emb.reshape(1, -1)
    sims  = cosine_similarity(q, all_embeddings)[0]
    top_k = np.argsort(sims)[::-1][:k]
    return [(cases[i]["case_id"], sims[i]) for i in top_k]

def extract_solution(case):
    return {
        "label_pasal":      case.get("label_pasal", ""),
        "label_vonis":      case.get("label_vonis", ""),
        "amar_putusan":     case.get("amar_putusan", ""),
        "vonis_bulan":      case.get("vonis_bulan", 0),
        "ringkasan_fakta":  case.get("ringkasan_fakta", "")[:300],
        "no_perkara":       case.get("no_perkara", ""),
    }

def majority_vote(solutions, key):
    values = [s[key] for s in solutions if s.get(key)]
    if not values:
        return "tidak_diketahui"
    return Counter(values).most_common(1)[0][0]

def weighted_vote(top_k_with_sim, case_map, key):
    scores = {}
    for cid, sim in top_k_with_sim:
        val = case_map[cid].get(key, "")
        if val:
            scores[val] = scores.get(val, 0.0) + sim
    if not scores:
        return "tidak_diketahui"
    return max(scores, key=scores.get)

def predict_outcome(query_text, vectorizer, all_vectors, all_embeddings,
                    cases, case_map, tokenizer=None, model=None, device=None, k=5):
    tfidf_top_k = retrieve_tfidf(query_text, vectorizer, all_vectors, cases, k)
    tfidf_solutions = [extract_solution(case_map[cid]) for cid, _ in tfidf_top_k]

    tfidf_mv_pasal  = majority_vote(tfidf_solutions, "label_pasal")
    tfidf_mv_vonis  = majority_vote(tfidf_solutions, "label_vonis")
    tfidf_wv_pasal  = weighted_vote(tfidf_top_k, case_map, "label_pasal")
    tfidf_wv_vonis  = weighted_vote(tfidf_top_k, case_map, "label_vonis")

    bert_mv_pasal = bert_mv_vonis = bert_wv_pasal = bert_wv_vonis = None
    bert_top_k = []
    if tokenizer is not None and model is not None:
        query_emb = get_indobert_embedding(query_text, tokenizer, model, device)
        bert_top_k = retrieve_bert(query_emb, all_embeddings, cases, k)
        bert_solutions = [extract_solution(case_map[cid]) for cid, _ in bert_top_k]
        bert_mv_pasal  = majority_vote(bert_solutions, "label_pasal")
        bert_mv_vonis  = majority_vote(bert_solutions, "label_vonis")
        bert_wv_pasal  = weighted_vote(bert_top_k, case_map, "label_pasal")
        bert_wv_vonis  = weighted_vote(bert_top_k, case_map, "label_vonis")

    return {
        "tfidf_top5":        [{"case_id": c, "sim": round(float(s), 4)} for c, s in tfidf_top_k],
        "tfidf_mv_pasal":    tfidf_mv_pasal,
        "tfidf_mv_vonis":    tfidf_mv_vonis,
        "tfidf_wv_pasal":    tfidf_wv_pasal,
        "tfidf_wv_vonis":    tfidf_wv_vonis,
        "bert_top5":         [{"case_id": c, "sim": round(float(s), 4)} for c, s in bert_top_k],
        "bert_mv_pasal":     bert_mv_pasal,
        "bert_mv_vonis":     bert_mv_vonis,
        "bert_wv_pasal":     bert_wv_pasal,
        "bert_wv_vonis":     bert_wv_vonis,
    }

def main():
    cases = load_cases()
    case_map = {c["case_id"]: c for c in cases}
    vectorizer, all_embeddings = load_models()

    texts = [c["text_full"] for c in cases]
    all_vectors = vectorizer.transform(texts)

    device    = get_device()
    tokenizer = None
    bert_model = None

    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        MODEL_NAME = "indobenchmark/indobert-base-p1"
        print(f"Loading IndoBERT: {MODEL_NAME} ...")
        tokenizer   = AutoTokenizer.from_pretrained(MODEL_NAME)
        bert_model  = AutoModel.from_pretrained(MODEL_NAME).to(device)
        bert_model.eval()
        print(f"Device: {device}")
    except Exception as e:
        print(f"IndoBERT tidak tersedia: {e}")

    with open(EVAL_DIR / "queries.json", encoding="utf-8") as f:
        queries = json.load(f)

    rows = []
    for q in queries:
        qid       = q["query_id"]
        gt_pasal  = q["ground_truth_label_pasal"]
        gt_vonis  = q["ground_truth_label_vonis"]
        qtext     = q["query_text"]

        result = predict_outcome(
            qtext, vectorizer, all_vectors, all_embeddings,
            cases, case_map, tokenizer, bert_model, device, k=5
        )

        top5_tfidf_ids = "|".join(x["case_id"] for x in result["tfidf_top5"])
        top5_bert_ids  = "|".join(x["case_id"] for x in result["bert_top5"]) if result["bert_top5"] else ""

        rows.append({
            "query_id":             qid,
            "ground_truth_pasal":   gt_pasal,
            "ground_truth_vonis":   gt_vonis,
            "tfidf_mv_pasal":       result["tfidf_mv_pasal"],
            "tfidf_mv_vonis":       result["tfidf_mv_vonis"],
            "tfidf_wv_pasal":       result["tfidf_wv_pasal"],
            "tfidf_wv_vonis":       result["tfidf_wv_vonis"],
            "bert_mv_pasal":        result["bert_mv_pasal"],
            "bert_mv_vonis":        result["bert_mv_vonis"],
            "bert_wv_pasal":        result["bert_wv_pasal"],
            "bert_wv_vonis":        result["bert_wv_vonis"],
            "top5_tfidf_case_ids":  top5_tfidf_ids,
            "top5_bert_case_ids":   top5_bert_ids,
        })

        print(f"  {qid}: GT_pasal={gt_pasal}, GT_vonis={gt_vonis}")
        print(f"         TF-IDF MV: pasal={result['tfidf_mv_pasal']}, vonis={result['tfidf_mv_vonis']}")
        print(f"         TF-IDF WV: pasal={result['tfidf_wv_pasal']}, vonis={result['tfidf_wv_vonis']}")
        if result["bert_mv_pasal"] is not None:
            print(f"         BERT   MV: pasal={result['bert_mv_pasal']}, vonis={result['bert_mv_vonis']}")
            print(f"         BERT   WV: pasal={result['bert_wv_pasal']}, vonis={result['bert_wv_vonis']}")
        print()

    df = pd.DataFrame(rows)
    out_path = RESULT_DIR / "predictions.csv"
    df.to_csv(out_path, index=False)
    print(f"\nPredictions disimpan: {out_path}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
