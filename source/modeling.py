from torch.utils.data import Dataset
import torch

def compute_label_language_similarity(
    X,
    y_df,
    max_df=0.8,
    min_df=5
):
    """
    Computes a label–label cosine similarity matrix based on TF-IDF
    representations of review text associated with each label.

    Parameters
    ----------
    X : pd.Series
        Text data (e.g., X_train), index-aligned with y_df.
    y_df : pd.DataFrame
        Binary label matrix (columns = labels).
    max_df : float
        Max document frequency for TF-IDF.
    min_df : int
        Min document frequency for TF-IDF.

    Returns
    -------
    similarity_df : pd.DataFrame
        Label–label cosine similarity matrix.
    """

    # Safety checks
    assert len(X) == len(y_df)
    assert X.index.equals(y_df.index)

    label_texts = []

    for label in y_df.columns:
        texts_for_label = X.loc[y_df[label] == 1]

        # Handle rare / empty labels safely
        combined_text = " ".join(texts_for_label) if len(texts_for_label) > 0 else ""

        label_texts.append({
            "label": label,
            "text": combined_text
        })

    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer(
        max_df=max_df,
        min_df=min_df
    )

    tfidf_matrix = tfidf_vectorizer.fit_transform(
        [item["text"] for item in label_texts]
    )

    # Cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=[item["label"] for item in label_texts],
        columns=[item["label"] for item in label_texts]
    )

    return similarity_df



import pandas as pd
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss
)
from typing import List, Dict

def load_baseline_classifier():
    """
    Loads the Hugging Face text-classification pipeline for the SamLowe model.
    """
    print("Downloading/Loading model 'SamLowe/roberta-base-go_emotions' into memory...")
    return pipeline(
        task="text-classification", 
        model="SamLowe/roberta-base-go_emotions", 
        top_k=None
    )

def extract_top_emotions(predictions, threshold=0.5):
    """
    Helper function to filter predictions based on a probability threshold.
    """
    top_emotions = [p['label'] for p in predictions if p['score'] > threshold]
    # Fallback to the strongest emotion if none are above the threshold
    if not top_emotions:
        top_emotions = [predictions[0]['label']]
    return ", ".join(top_emotions)

'''def apply_baseline_to_dataframe(df: pd.DataFrame, text_column: str = 'text', 
                                threshold: float = 0.5, 
                                output_column: str = 'baseline_predictions') -> pd.DataFrame:
    """
    Applies the baseline model to a text column in the DataFrame and returns the 
    DataFrame with a new column for predictions.
    
    Args:
        df: Pandas DataFrame containing the texts.
        text_column: Name of the column containing the text.
        threshold: Minimum probability (0.0 to 1.0) required for an emotion.
        output_column: Name of the output column for predictions (default: 'baseline_predictions').
    """
    # Create a copy to avoid unintentionally modifying the original DataFrame
    df_result = df.copy()
    
    classifier = load_baseline_classifier()
    
    print(f"Gathering predictions for {len(df)} rows... This might take a moment!")
    
    def predict_row(text):
        try:
            preds = classifier(str(text))[0]
            return extract_top_emotions(preds, threshold=threshold)
        except Exception as e:
            return "error"

    # Use tqdm for a progress bar if installed, otherwise use standard apply
    try:
        from tqdm import tqdm
        tqdm.pandas(desc="Generating predictions")
        df_result[output_column] = df_result[text_column].progress_apply(predict_row)
    except ImportError:
        df_result[output_column] = df_result[text_column].apply(predict_row)
        
    print(f"Predictions completed successfully! Saved to column: '{output_column}'")
    return df_result'''

def apply_baseline_to_dataframe(df: pd.DataFrame, text_column: str = 'text', 
                                threshold: float = 0.5, 
                                output_column: str = 'baseline_predictions',
                                model_name: str = 'SamLowe/roberta-base-go_emotions') -> pd.DataFrame:
    """
    Applies a Hugging Face model to a text column.
    
    Args:
        df: Pandas DataFrame containing the texts.
        text_column: Name of the column containing the text.
        threshold: Minimum probability (0.0 to 1.0) required for an emotion.
        output_column: Name of the output column for predictions.
        model_name: Hugging Face model identifier.
    """
    df_result = df.copy()
    
    print(f"Loading model: {model_name}...")
    classifier = pipeline(
        task="text-classification", 
        model=model_name, 
        top_k=None
    )
    
    print(f"Gathering predictions for {len(df)} rows...")
    
    def predict_row(text):
        try:
            preds = classifier(str(text))[0]
            top_emotions = [p['label'] for p in preds if p['score'] > threshold]
            if not top_emotions:
                top_emotions = [preds[0]['label']]
            return ", ".join(top_emotions)
        except Exception as e:
            return "error"

    try:
        from tqdm import tqdm
        tqdm.pandas(desc="Generating predictions")
        df_result[output_column] = df_result[text_column].progress_apply(predict_row)
    except ImportError:
        df_result[output_column] = df_result[text_column].apply(predict_row)
        
    print(f"Predictions saved to column: '{output_column}'")
    return df_result

def predict_with_model(df: pd.DataFrame, text_column: str, model_name: str, 
                       output_column: str, threshold: float = 0.5,
                       max_length: int = 512) -> pd.DataFrame:
    """
    Apply any Hugging Face model to a text column and add predictions as a new column.
    """
    from dotenv import load_dotenv
    import os
    from transformers import pipeline
    from huggingface_hub import login
    
    # Load token from .env
    load_dotenv()
    token = os.getenv('HUGGINGFACE_TOKEN')
    
    if token:
        print("Using authentication token")
        # Login first
        login(token=token)
    else:
        print("No token found")
    
    df_result = df.copy()
    
    print(f"Loading model: {model_name}...")
    
    # Pass token explicitly to pipeline
    classifier = pipeline(
        task="text-classification", 
        model=model_name, 
        token=token,  # <<< IMPORTANT: Pass token here
        top_k=None,
        truncation=True,
        max_length=max_length
    )
    
    print(f"Generating predictions for {len(df)} rows...")
    
    # Original prediction function with string output
    # def predict_row(text):
    #    try:
    #        preds = classifier(str(text), truncation=True, max_length=max_length)[0]
    #        top_emotions = [p['label'] for p in preds if p['score'] > threshold]
    #        if not top_emotions:
    #            top_emotions = [preds[0]['label']]
    #        return ", ".join(top_emotions)
    #    except Exception as e:
    #        print(f"Error: {e}")
    #        return "error"

    # Prediction with binary dictionary output
    def predict_row(text):
        try:
            preds = classifier(str(text), truncation=True, max_length=max_length)[0]
            binary = {p['label']: 1 if p['score'] > threshold else 0 for p in preds}
            return binary
        except Exception as e:
            print(f"Error: {e}")
            return {}

    try:
        from tqdm import tqdm
        tqdm.pandas(desc=f"Predicting with {output_column}")
        df_result[output_column] = df_result[text_column].progress_apply(predict_row)
    except ImportError:
        df_result[output_column] = df_result[text_column].apply(predict_row)
    
    print(f" Predictions saved to column: '{output_column}'")
    return df_result
    
# Convert prediction dictionaries(binary) to lists(Strings) of predicted emotions
def prediction_to_labels(pred):
    if isinstance(pred, dict):
        return [emotion for emotion, value in pred.items() if value == 1]
    return []


def multilabel_metrics(true_df: pd.DataFrame, pred_series: pd.Series, 
                       emotion_cols: List[str]) -> Dict:
    """
    Calculate multilabel classification metrics.
    
    Parameters
    ----------
    true_df : pd.DataFrame
        DataFrame with true labels (0/1) for each emotion
    pred_series : pd.Series
        Series with predicted emotions as strings (e.g., "joy, sadness")
    emotion_cols : List[str]
        List of emotion column names
    
    Returns
    -------
    dict
        Dictionary containing all metrics
    """
    
    # Creates a binary dataframe for predictions with the same structure as true_df
    pred_binary = pd.DataFrame(0, index=pred_series.index, columns=emotion_cols)
    
    # Iterate through predictions and set emotions
    for idx, pred_value in enumerate(pred_series):
        if pd.isna(pred_value) or pred_value == "error":
            continue

        # Handle case where predictions are dictionaries of binary values
        if isinstance(pred_value, dict):
            predicted_emotions = [emotion for emotion, value in pred_value.items() if value == 1]
        # Handle case where predictions are already lists of emotions
        elif isinstance(pred_value, list):
            predicted_emotions = [emotion for emotion in pred_value if emotion in emotion_cols]
        # Handle case where predictions are strings of comma-separated emotions
        else:
            predicted_emotions = [e.strip() for e in str(pred_value).split(',') if e.strip()]

        for emotion in predicted_emotions:
            if emotion in emotion_cols:
                pred_binary.iloc[idx, pred_binary.columns.get_loc(emotion)] = 1
    
    # Get true labels as numpy array
    true_binary = true_df[emotion_cols].values
    pred_binary_values = pred_binary.values
    
    # Calculate metrics
    metrics = {
        'exact_match_ratio': accuracy_score(true_binary, pred_binary_values),
        'hamming_loss': hamming_loss(true_binary, pred_binary_values),
        'micro_f1': f1_score(true_binary, pred_binary_values, average='micro', zero_division=0),
        'macro_f1': f1_score(true_binary, pred_binary_values, average='macro', zero_division=0),
        'weighted_f1': f1_score(true_binary, pred_binary_values, average='weighted', zero_division=0),
        'samples_f1': f1_score(true_binary, pred_binary_values, average='samples', zero_division=0),
        'micro_precision': precision_score(true_binary, pred_binary_values, average='micro', zero_division=0),
        'micro_recall': recall_score(true_binary, pred_binary_values, average='micro', zero_division=0),
        'macro_precision': precision_score(true_binary, pred_binary_values, average='macro', zero_division=0),
        'macro_recall': recall_score(true_binary, pred_binary_values, average='macro', zero_division=0),
    }
    
    # Per-emotion metrics
    per_emotion = {}
    for i, emotion in enumerate(emotion_cols):
        per_emotion[emotion] = {
            'f1': f1_score(true_binary[:, i], pred_binary_values[:, i], zero_division=0),
            'precision': precision_score(true_binary[:, i], pred_binary_values[:, i], zero_division=0),
            'recall': recall_score(true_binary[:, i], pred_binary_values[:, i], zero_division=0),
            'support': true_binary[:, i].sum()
        }
    
    metrics['per_emotion'] = per_emotion
    
    return metrics

def create_model_comparison_df(df: pd.DataFrame, emotion_cols: List[str],
                                 prediction_cols: Dict[str, str]) -> pd.DataFrame:
    """
    Create a DataFrame where each row is a model/preprocessing version
    and columns are evaluation metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing true labels and predictions
    emotion_cols : List[str]
        List of emotion column names
    prediction_cols : Dict[str, str]
        Dictionary mapping model/preprocessing name to prediction column
        Example: {'Minimal Preprocessing': '01_baseline_predictions',
                  'Medium Preprocessing': '02_baseline_predictions',
                  'Harsh Preprocessing': '03_baseline_predictions'}
    
    Returns
    -------
    pd.DataFrame
        DataFrame with models as rows and metrics as columns
    """
    results = []
    
    for model_name, pred_col in prediction_cols.items():
        if pred_col not in df.columns:
            print(f"Warning: {pred_col} not found in DataFrame")
            continue
        
        # Remove rows with errors
        valid_rows = df[pred_col].apply(lambda x: x != "error" and x != {} and pd.notna(x))
        valid_df = df[valid_rows]
        
        if len(valid_df) == 0:
            print(f"No valid predictions for {model_name}")
            continue
        
        # Calculate metrics
        metrics = multilabel_metrics(
            true_df=valid_df,
            pred_series=valid_df[pred_col],
            emotion_cols=emotion_cols
        )
        
        # Store results as a row
        results.append({
            'model': model_name,
            'exact_match_ratio': metrics['exact_match_ratio'],
            'hamming_loss': metrics['hamming_loss'],
            'micro_f1': metrics['micro_f1'],
            'macro_f1': metrics['macro_f1'],
            'weighted_f1': metrics['weighted_f1'],
            'samples_f1': metrics['samples_f1'],
            'micro_precision': metrics['micro_precision'],
            'micro_recall': metrics['micro_recall'],
            'macro_precision': metrics['macro_precision'],
            'macro_recall': metrics['macro_recall'],
            'total_samples': len(valid_df),
            'error_count': len(df) - len(valid_df)
        })
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    column_order = ['model', 'micro_f1', 'macro_f1', 'samples_f1', 'weighted_f1',
                    'exact_match_ratio', 'hamming_loss', 'micro_precision', 
                    'micro_recall', 'macro_precision', 'macro_recall',
                    'total_samples', 'error_count']
    
    comparison_df = comparison_df[column_order]
    
    return comparison_df


# to trasform the data into the necessary format for the fine tunning of the model
# trasnforming the text into vectors and the labels into binary format for the multi-label classification task
class GoEmotionsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=500):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }
    


class SimplePredictionDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
    

#Extract sentiment scores for each text
def add_vader_features(df, text_column='01_minimal_preprocessing'):
    """
    Add VADER sentiment scores as features to the DataFrame
    
    Parameters:
    - df: DataFrame containing text data
    - text_column: Name of column containing text to analyze
    
    Returns:
    - df: DataFrame with added sentiment columns
    """
    # Apply VADER to each text
    vader_scores = df[text_column].apply(
        lambda x: analyzer.polarity_scores(str(x))
    )
    
    # Extract individual sentiment components
    df['vader_neg'] = vader_scores.apply(lambda x: x['neg'])
    df['vader_neu'] = vader_scores.apply(lambda x: x['neu'])
    df['vader_pos'] = vader_scores.apply(lambda x: x['pos'])
    df['vader_compound'] = vader_scores.apply(lambda x: x['compound'])
    
    # Create sentiment polarity label
    df['vader_sentiment'] = df['vader_compound'].apply(
        lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral')
    )
    
    return df

# ERROR ANALYSIS IMPORTS
from itertools import combinations
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
 
 
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