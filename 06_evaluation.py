import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score
)

warnings.filterwarnings("ignore")

EVAL_DIR   = Path("data/eval")
RESULT_DIR = Path("data/results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

def compute_metrics(y_true, y_pred, model_name, label_name):
    mask = [i for i, v in enumerate(y_true) if v != "tidak_diketahui"]
    yt = [y_true[i] for i in mask]
    yp = [y_pred[i] for i in mask]
    if not yt:
        return {}
    acc  = accuracy_score(yt, yp)
    prec = precision_score(yt, yp, average="weighted", zero_division=0)
    rec  = recall_score(yt, yp, average="weighted", zero_division=0)
    f1   = f1_score(yt, yp, average="weighted", zero_division=0)
    return {
        "model":     model_name,
        "label":     label_name,
        "accuracy":  round(acc, 4),
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1_score":  round(f1, 4),
        "n_samples": len(yt),
    }

def print_report(y_true, y_pred, model_name, label_name):
    mask = [i for i, v in enumerate(y_true) if v != "tidak_diketahui"]
    yt = [y_true[i] for i in mask]
    yp = [y_pred[i] for i in mask]
    if not yt:
        return
    print(f"\n{'='*60}")
    print(f"  {model_name} | {label_name}")
    print(f"{'='*60}")
    print(classification_report(yt, yp, zero_division=0))

def main():
    retrieval_path  = EVAL_DIR / "retrieval_metrics.csv"
    predictions_path = RESULT_DIR / "predictions.csv"

    df_ret  = pd.read_csv(retrieval_path)
    df_pred = pd.read_csv(predictions_path)

    print("\n" + "="*70)
    print("TAHAP 5: MODEL EVALUATION")
    print("="*70)

    print("\n--- Retrieval Metrics (dari Tahap 3) ---")
    print(df_ret.to_string(index=False))

    gt_pasal = df_pred["ground_truth_pasal"].tolist()
    gt_vonis = df_pred["ground_truth_vonis"].tolist()

    pred_cols = {
        "TF-IDF + MajorityVote": ("tfidf_mv_pasal", "tfidf_mv_vonis"),
        "TF-IDF + WeightedVote": ("tfidf_wv_pasal", "tfidf_wv_vonis"),
    }

    if df_pred["bert_mv_pasal"].notna().any():
        pred_cols["IndoBERT + MajorityVote"] = ("bert_mv_pasal", "bert_mv_vonis")
        pred_cols["IndoBERT + WeightedVote"]  = ("bert_wv_pasal", "bert_wv_vonis")

    pred_metrics = []
    for model_name, (col_pasal, col_vonis) in pred_cols.items():
        y_pasal = df_pred[col_pasal].fillna("tidak_diketahui").tolist()
        y_vonis = df_pred[col_vonis].fillna("tidak_diketahui").tolist()

        m = compute_metrics(gt_pasal, y_pasal, model_name, "label_pasal")
        if m:
            pred_metrics.append(m)
            print_report(gt_pasal, y_pasal, model_name, "label_pasal")

        m = compute_metrics(gt_vonis, y_vonis, model_name, "label_vonis")
        if m:
            pred_metrics.append(m)
            print_report(gt_vonis, y_vonis, model_name, "label_vonis")

    df_pred_metrics = pd.DataFrame(pred_metrics)
    pred_metrics_path = EVAL_DIR / "prediction_metrics.csv"
    df_pred_metrics.to_csv(pred_metrics_path, index=False)

    print("\n\n" + "="*70)
    print("PREDICTION METRICS SUMMARY")
    print("="*70)
    print(df_pred_metrics.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("CBR System Evaluation — Retrieval vs Prediction", fontsize=14, fontweight="bold")

    for ax, target_label, title in [
        (axes[0], "label_pasal", "Label Pasal (362 vs 363)"),
        (axes[1], "label_vonis", "Label Vonis (Berat / Sedang / Ringan)"),
    ]:
        df_r = df_ret[df_ret["label"] == target_label].copy()
        df_p = df_pred_metrics[df_pred_metrics["label"] == target_label].copy()

        df_r["source"] = "Retrieval"
        df_p["source"] = "Prediction"

        df_all = pd.concat([
            df_r[["model", "accuracy", "f1_score", "source"]],
            df_p[["model", "accuracy", "f1_score", "source"]],
        ], ignore_index=True)

        models  = df_all["model"].tolist()
        acc     = df_all["accuracy"].tolist()
        f1      = df_all["f1_score"].tolist()
        x       = np.arange(len(models))
        width   = 0.35

        bars1 = ax.bar(x - width/2, acc, width, label="Accuracy", color="#4C72B0", alpha=0.85)
        bars2 = ax.bar(x + width/2, f1,  width, label="F1-Score",  color="#DD8452", alpha=0.85)

        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.15)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for bar in bars1:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
        for bar in bars2:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)

    plt.tight_layout()
    chart_path = EVAL_DIR / "evaluation_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"\nChart disimpan: {chart_path}")

    print("\n\n" + "="*70)
    print("ERROR ANALYSIS")
    print("="*70)

    for model_name, (col_pasal, col_vonis) in pred_cols.items():
        errors = df_pred[df_pred[col_pasal] != df_pred["ground_truth_pasal"]]
        if not errors.empty:
            print(f"\n[{model_name}] Kesalahan Prediksi label_pasal:")
            for _, row in errors.iterrows():
                print(f"  {row['query_id']}: GT={row['ground_truth_pasal']} -> Pred={row[col_pasal]}")

    print("\n\n" + "="*70)
    print("ANALISIS KEGAGALAN & REKOMENDASI")
    print("="*70)

    df_ret_pasal = df_ret[df_ret["label"] == "label_pasal"]
    best_ret = df_ret_pasal.loc[df_ret_pasal["f1_score"].idxmax()]
    print(f"\nModel Retrieval Terbaik (label_pasal): {best_ret['model']} "
          f"(Accuracy={best_ret['accuracy']:.4f}, F1={best_ret['f1_score']:.4f})")

    df_ret_vonis = df_ret[df_ret["label"] == "label_vonis"]
    best_vonis = df_ret_vonis.loc[df_ret_vonis["f1_score"].idxmax()]
    print(f"Model Retrieval Terbaik (label_vonis): {best_vonis['model']} "
          f"(Accuracy={best_vonis['accuracy']:.4f}, F1={best_vonis['f1_score']:.4f})")

    print("""
Temuan & Rekomendasi:
1. label_pasal: TF-IDF + SVM mencapai F1 tertinggi (0.9161) dibanding IndoBERT (0.8286).
   -> Teks hukum formal dengan fitur kata kunci pasal cenderung cocok untuk TF-IDF.
2. label_vonis: IndoBERT + Cosine lebih baik (F1=0.7083) dibanding TF-IDF (F1=0.4808).
   -> Makna semantik lebih penting untuk memprediksi berat/ringannya vonis.
3. Imbalance kelas vonis (sedang=33, ringan=8, berat=1) menjadi faktor utama
   rendahnya performa label_vonis - rekomendasi: augmentasi data atau oversampling.
4. Kasus "tidak_diketahui" (18 kasus) dikecualikan dari evaluasi vonis.
   -> Ekstraksi regex amar putusan perlu ditingkatkan agar bisa menangkap lebih banyak kasus.
""")

    print(f"\nSemua output tersimpan di:")
    print(f"  {pred_metrics_path}")
    print(f"  {chart_path}")

if __name__ == "__main__":
    main()
