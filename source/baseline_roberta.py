''''"""
Baseline Model Module
---------------------
Contains functions to apply the pre-trained SamLowe/roberta-base-go_emotions 
model to the dataset so that future models can be compared against it.
"""

import pandas as pd
from transformers import pipeline

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

def apply_baseline_to_dataframe(df: pd.DataFrame, text_column: str = 'text', threshold: float = 0.5) -> pd.DataFrame:
    """
    Applies the baseline model to a text column in the DataFrame and returns the 
    DataFrame with a new 'baseline_predictions' column.
    
    Args:
        df: Pandas DataFrame containing the texts.
        text_column: Name of the column containing the text.
        threshold: Minimum probability (0.0 to 1.0) required for an emotion.
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
        df_result['baseline_predictions'] = df_result[text_column].progress_apply(predict_row)
    except ImportError:
        df_result['baseline_predictions'] = df_result[text_column].apply(predict_row)
        
    print("Predictions completed successfully!")
    return df_result'''

"""
Baseline Model Module
---------------------
Contains functions to apply the pre-trained SamLowe/roberta-base-go_emotions 
model to the dataset so that future models can be compared against it.

"""