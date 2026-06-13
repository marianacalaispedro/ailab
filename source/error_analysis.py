# ERROR ANALYSIS IMPORTS
from itertools import combinations
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
from typing import List

def build_pred_binary(test_data: pd.DataFrame, pred_col_name: str, emotion_cols: List[str]) -> pd.DataFrame:
    """Build a binary prediction matrix from a column of prediction dictionaries."""
    pred_binary = pd.DataFrame(0, index=test_data.index, columns=emotion_cols)
    for idx, pred in enumerate(test_data[pred_col_name]):
        if isinstance(pred, dict):
            for emotion, val in pred.items():
                if emotion in emotion_cols and val == 1:
                    pred_binary.loc[pred_binary.index[idx], emotion] = 1
    return pred_binary

def analyze_error_combinations(test_data: pd.DataFrame, pred_binary: pd.DataFrame, emotion_cols: List[str]) -> pd.DataFrame:
    """Count how often each TP/FP/FN combination occurs across all sentences."""
    combo_counts = Counter()
    for i in range(len(test_data)):
        true_row = test_data[emotion_cols].iloc[i]
        pred_row = pred_binary.iloc[i]
        tp = int(((pred_row == 1) & (true_row == 1)).sum())
        fp = int(((pred_row == 1) & (true_row == 0)).sum())
        fn = int(((pred_row == 0) & (true_row == 1)).sum())
        if fp > 0 or fn > 0:
            parts = []
            if tp > 0: parts.append(f"{tp}TP")
            if fp > 0: parts.append(f"{fp}FP")
            if fn > 0: parts.append(f"{fn}FN")
            combo_counts["_".join(parts)] += 1
    df = pd.DataFrame(combo_counts.items(), columns=["Combination", "Count"])
    return df.sort_values("Count", ascending=False).reset_index(drop=True)
 
 
def analyze_errors_per_emotion(test_data: pd.DataFrame, pred_binary: pd.DataFrame, emotion_cols: List[str]) -> pd.DataFrame:
    """Calculate TP, FP, FN, and error rate for each emotion."""
    rows = []
    for e in emotion_cols:
        true_col = test_data[e]
        pred_col = pred_binary[e]
        tp = int(((pred_col == 1) & (true_col == 1)).sum())
        fp = int(((pred_col == 1) & (true_col == 0)).sum())
        fn = int(((pred_col == 0) & (true_col == 1)).sum())
        total_true = int(true_col.sum())
        total_pred = int(pred_col.sum())
        errors = fp + fn
        error_rate = errors / (total_true + total_pred) if (total_true + total_pred) > 0 else 0
        rows.append({
            "emotion": e, "TP": tp, "FP": fp, "FN": fn,
            "Total True": total_true, "Total Predicted": total_pred,
            "Errors (abs)": errors, "Error Rate": round(error_rate * 100, 1)
        })
    df = pd.DataFrame(rows).set_index("emotion")
    return df.sort_values("Errors (abs)", ascending=False)
 
 
def analyze_errors_by_cardinality(test_data: pd.DataFrame, pred_binary: pd.DataFrame, emotion_cols: List[str]) -> pd.DataFrame:
    """Analyze how error rates change with the number of true emotions per sentence."""
    true_counts = test_data[emotion_cols].sum(axis=1)
    rows = []
    for n in sorted(true_counts.unique()):
        mask = true_counts == n
        subset_true = test_data[emotion_cols][mask]
        subset_pred = pred_binary[mask]
        tp = int(((subset_pred == 1) & (subset_true == 1)).values.sum())
        fp = int(((subset_pred == 1) & (subset_true == 0)).values.sum())
        fn = int(((subset_pred == 0) & (subset_true == 1)).values.sum())
        n_sentences = int(mask.sum())
        rows.append({
            "True Emotion Count": int(n),
            "Sentences": n_sentences,
            "Total True Emotions": n * n_sentences,
            "TP": tp,
            "TP (%)": round(tp / (tp + fn) * 100, 1) if (tp + fn) > 0 else 0,
            "FP": fp,
            "FP (%)": round(fp / (fp + tp) * 100, 1) if (fp + tp) > 0 else 0,
            "FN": fn,
            "FN (%)": round(fn / (fn + tp) * 100, 1) if (fn + tp) > 0 else 0,
            "Errors (abs)": fp + fn,
            "Error (%)": round((fp + fn) / (tp + fp + fn) * 100, 1) if (tp + fp + fn) > 0 else 0,
            "Errors per Sentence": round((fp + fn) / n_sentences, 2),
        })
    return pd.DataFrame(rows).set_index("True Emotion Count")
 
 
def analyze_errors_by_length(test_data: pd.DataFrame, pred_binary: pd.DataFrame, emotion_cols: List[str]) -> pd.DataFrame:
    """Analyze how error rates change with text length."""
    df = test_data.copy()
    df["text_length"] = df["text"].str.len()
    df["FP"] = ((pred_binary == 1) & (test_data[emotion_cols] == 0)).sum(axis=1)
    df["FN"] = ((pred_binary == 0) & (test_data[emotion_cols] == 1)).sum(axis=1)
    df["errors"] = df["FP"] + df["FN"]
    df["true_emotion_count"] = test_data[emotion_cols].sum(axis=1)
    df["pred_emotion_count"] = pred_binary.sum(axis=1)
    bins = [0, 30, 60, 100, 150, 200, 99999]
    labels = ["<30", "30-60", "60-100", "100-150", "150-200", ">200"]
    df["length_bin"] = pd.cut(df["text_length"], bins=bins, labels=labels)
    return df.groupby("length_bin", observed=True).agg(
        Sentences=("errors", "count"),
        Avg_True_Emotions=("true_emotion_count", "mean"),
        Avg_Pred_Emotions=("pred_emotion_count", "mean"),
        FP_total=("FP", "sum"),
        FN_total=("FN", "sum"),
        Errors_per_Sentence=("errors", "mean"),
        FP_per_Sentence=("FP", "mean"),
        FN_per_Sentence=("FN", "mean"),
    ).round(2)
 
 
def plot_error_analysis(combo_df: pd.DataFrame, emotion_error_df: pd.DataFrame,
                        cardinality_df: pd.DataFrame, length_df: pd.DataFrame,
                        model_name: str, save_dir: str = "plots") -> None:
    """Plot all 4 error analyses for a single model and save to disk."""
    os.makedirs(save_dir, exist_ok=True)
 
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f"Error Analysis: {model_name}", fontsize=16, fontweight="bold")
 
    top_combos = combo_df.head(15)
    axes[0, 0].barh(top_combos["Combination"], top_combos["Count"], color="steelblue")
    axes[0, 0].set_title("Top 15 Error Combinations")
    axes[0, 0].set_xlabel("Count")
    axes[0, 0].invert_yaxis()
 
    top_emotions = emotion_error_df.head(15)
    x = range(len(top_emotions))
    width = 0.3
    axes[0, 1].bar([i - width for i in x], top_emotions["FP"], width=width, label="FP", color="tomato")
    axes[0, 1].bar([i for i in x],         top_emotions["FN"], width=width, label="FN", color="orange")
    axes[0, 1].bar([i + width for i in x], top_emotions["TP"], width=width, label="TP", color="mediumseagreen")
    axes[0, 1].set_xticks(list(x))
    axes[0, 1].set_xticklabels(top_emotions.index, rotation=45, ha="right", fontsize=7)
    axes[0, 1].set_title("Top 15 Emotions: TP / FP / FN")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].legend()
 
    error_rates = cardinality_df["Errors per Sentence"].astype(float)
    axes[1, 0].bar(error_rates.index.astype(str), error_rates.values, color="mediumpurple")
    axes[1, 0].set_title("Errors per Sentence by True Emotion Count")
    axes[1, 0].set_xlabel("Number of True Emotions")
    axes[1, 0].set_ylabel("Errors per Sentence")
 
    axes[1, 1].plot(length_df.index.astype(str), length_df["Errors_per_Sentence"],
                    marker="o", color="steelblue", label="Errors per Sentence")
    axes[1, 1].plot(length_df.index.astype(str), length_df["FP_per_Sentence"],
                    marker="s", color="tomato", linestyle="--", label="FP per Sentence")
    axes[1, 1].plot(length_df.index.astype(str), length_df["FN_per_Sentence"],
                    marker="^", color="orange", linestyle="--", label="FN per Sentence")
    axes[1, 1].set_title("Errors by Text Length")
    axes[1, 1].set_xlabel("Text Length (chars)")
    axes[1, 1].set_ylabel("Errors per Sentence")
    axes[1, 1].legend()
 
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"error_analysis_{model_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
 
 
def run_error_analysis(test_data: pd.DataFrame, pred_cols: List[str], emotion_cols: List[str], save_dir: str = "plots") -> None:
    """Run all 4 error analyses for each prediction column and save plots."""
    for pred_col_name in pred_cols:
        pred_binary    = build_pred_binary(test_data, pred_col_name, emotion_cols)
        combo_df       = analyze_error_combinations(test_data, pred_binary, emotion_cols)
        emotion_err_df = analyze_errors_per_emotion(test_data, pred_binary, emotion_cols)
        cardinality_df = analyze_errors_by_cardinality(test_data, pred_binary, emotion_cols)
        length_df      = analyze_errors_by_length(test_data, pred_binary, emotion_cols)
        plot_error_analysis(combo_df, emotion_err_df, cardinality_df, length_df, pred_col_name, save_dir)


def compare_models_side_by_side(test_data: pd.DataFrame, pred_cols: List[str], emotion_cols: List[str], save_dir: str = "plots") -> None:
    """
    Compare all models side by side across all 4 error analyses.
    Saves all plots and exports results to Excel.
    """
    os.makedirs(save_dir, exist_ok=True)

    all_combos         = {}
    all_emotion_errors = {}
    all_cardinality    = {}
    all_length         = {}
    summary_rows       = []

    for pred_col_name in pred_cols:
        pred_binary = build_pred_binary(test_data, pred_col_name, emotion_cols)
        all_combos[pred_col_name]         = analyze_error_combinations(test_data, pred_binary, emotion_cols)
        all_emotion_errors[pred_col_name] = analyze_errors_per_emotion(test_data, pred_binary, emotion_cols)
        all_cardinality[pred_col_name]    = analyze_errors_by_cardinality(test_data, pred_binary, emotion_cols)
        all_length[pred_col_name]         = analyze_errors_by_length(test_data, pred_binary, emotion_cols)

        fp_total = int(((pred_binary == 1) & (test_data[emotion_cols] == 0)).values.sum())
        fn_total = int(((pred_binary == 0) & (test_data[emotion_cols] == 1)).values.sum())
        tp_total = int(((pred_binary == 1) & (test_data[emotion_cols] == 1)).values.sum())
        errors = fp_total + fn_total
        summary_rows.append({
            "Model": pred_col_name,
            "TP": tp_total,
            "FP": fp_total,
            "FN": fn_total,
            "Total Errors": errors,
            "Error Rate (%)": round(errors / (tp_total + fp_total + fn_total) * 100, 1) if (tp_total + fp_total + fn_total) > 0 else 0
        })

    n = len(pred_cols)
    colors = ["steelblue", "tomato", "mediumpurple", "orange", "mediumseagreen",
              "darkblue", "darkred", "darkorchid", "darkorange"]

    # Error Combinations
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 8))
    if n == 1: axes = [axes]
    fig.suptitle("Error Combinations", fontsize=14, fontweight="bold")
    for ax, pred_col in zip(axes, pred_cols):
        df = all_combos[pred_col].head(15)
        ax.barh(df["Combination"], df["Count"], color="steelblue")
        ax.set_title(pred_col, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_error_combinations.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # Errors per Emotion
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 8), sharey=True)
    if n == 1: axes = [axes]
    fig.suptitle("Errors per Emotion (Top 15)", fontsize=14, fontweight="bold")
    for ax, pred_col in zip(axes, pred_cols):
        df = all_emotion_errors[pred_col].head(15)
        x = range(len(df))
        width = 0.3
        ax.bar([i - width for i in x], df["FP"], width=width, label="FP", color="tomato")
        ax.bar([i for i in x],         df["FN"], width=width, label="FN", color="orange")
        ax.bar([i + width for i in x], df["TP"], width=width, label="TP", color="mediumseagreen")
        ax.set_xticks(list(x))
        ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=7)
        ax.set_title(pred_col, fontsize=9)
        ax.set_ylabel("Count")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_errors_per_emotion.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # Errors by Cardinality
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1: axes = [axes]
    fig.suptitle("Errors per Sentence by True Emotion Count", fontsize=14, fontweight="bold")
    for ax, pred_col in zip(axes, pred_cols):
        df = all_cardinality[pred_col]
        error_rates = df["Errors per Sentence"].astype(float)
        ax.bar(error_rates.index.astype(str), error_rates.values, color="mediumpurple")
        ax.set_title(pred_col, fontsize=9)
        ax.set_xlabel("True Emotion Count")
        ax.set_ylabel("Errors per Sentence")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_errors_by_cardinality.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # Errors by Text Length
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Errors per Sentence by Text Length", fontsize=14, fontweight="bold")
    for i, pred_col in enumerate(pred_cols):
        df = all_length[pred_col]
        ax.plot(df.index.astype(str), df["Errors_per_Sentence"],
                marker="o", label=pred_col, color=colors[i % len(colors)])
    ax.set_xlabel("Text Length (chars)")
    ax.set_ylabel("Errors per Sentence")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison_errors_by_length.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # Export to Excel
    excel_path = os.path.join(save_dir, "error_analysis_results.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).set_index("Model").to_excel(writer, sheet_name="Summary")
        for pred_col in pred_cols:
            short = pred_col[:26]
            all_combos[pred_col].to_excel(writer, sheet_name=f"{short}_cmb", index=False)
            all_emotion_errors[pred_col].to_excel(writer, sheet_name=f"{short}_emo")
            all_cardinality[pred_col].to_excel(writer, sheet_name=f"{short}_card")
            all_length[pred_col].to_excel(writer, sheet_name=f"{short}_len")

def build_cooccurrence_table(true_df, pred_series, emotion_cols):
    
    pred_binary = pd.DataFrame(0, index=pred_series.index, columns=emotion_cols)
    for idx, pred in enumerate(pred_series):
        if isinstance(pred, dict):
            for emotion, val in pred.items():
                if emotion in emotion_cols and val == 1:
                    pred_binary.iloc[idx][emotion] = 1

    all_labels = []
    for e in emotion_cols:
        for outcome in ["TP", "FP", "FN"]:
            all_labels.append(f"{e}_{outcome}")

    matrix = pd.DataFrame(0, index=all_labels, columns=all_labels)

    for i in range(len(true_df)):
        true_row = true_df[emotion_cols].iloc[i]
        pred_row = pred_binary.iloc[i]

        row_labels = []
        for e in emotion_cols:
            t = true_row[e]
            p = pred_row[e]
            if p == 1 and t == 1:
                row_labels.append(f"{e}_TP")
            elif p == 1 and t == 0:
                row_labels.append(f"{e}_FP")
            elif p == 0 and t == 1:
                row_labels.append(f"{e}_FN")

        for a, b in combinations(row_labels, 2):
            matrix.loc[a, b] += 1
            matrix.loc[b, a] += 1

    mask = matrix.sum(axis=1) > 0
    matrix = matrix.loc[mask, mask]

    return matrix, pred_binary  # <-- pred_binary auch zurückgeben