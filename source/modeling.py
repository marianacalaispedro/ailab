#ERASE?
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
        print("✅ Using authentication token")
        # Login first
        login(token=token)
    else:
        print("⚠️ No token found")
    
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
    
    #def predict_row(text):
    #    try:
    #        preds = classifier(str(text), truncation=True, max_length=max_length)[0]
    #        top_emotions = [p['label'] for p in preds if p['score'] > threshold]
    #        if not top_emotions:
    #            top_emotions = [preds[0]['label']]
    #        return ", ".join(top_emotions)
    #    except Exception as e:
    #        print(f"Error: {e}")
    #        return "error"

    #Change to binary output
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
    # Convert predictions to binary matrix
    pred_binary = pd.DataFrame(0, index=pred_series.index, columns=emotion_cols)
    
    for idx, pred_str in enumerate(pred_series):
        if pd.notna(pred_str) and pred_str != "error":
            predicted_emotions = [e.strip() for e in pred_str.split(',')]
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
        valid_rows = df[pred_col] != "error"
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