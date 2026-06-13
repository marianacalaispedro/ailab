from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
import matplotlib.pyplot as plt



def add_vader_features(df, text_column='01_minimal_preprocessing'):
    """Add VADER sentiment scores as features"""
    analyzer = SentimentIntensityAnalyzer()
    vader_scores = df[text_column].apply(lambda x: analyzer.polarity_scores(str(x)))
    
    df['vader_neg'] = vader_scores.apply(lambda x: x['neg'])
    df['vader_neu'] = vader_scores.apply(lambda x: x['neu'])
    df['vader_pos'] = vader_scores.apply(lambda x: x['pos'])
    df['vader_compound'] = vader_scores.apply(lambda x: x['compound'])
    df['vader_sentiment'] = df['vader_compound'].apply(
        lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral')
    )
    return df


def evaluate_predictions(df, pred_col, emotion_cols, model_name="Model"):
    """Calculate multi-label classification metrics"""
    # Convert dict predictions to binary matrix
    pred_matrix = np.array([
        [row[pred_col].get(emo, 0) for emo in emotion_cols] 
        for _, row in df.iterrows()
    ])
    true_matrix = df[emotion_cols].values
    
    metrics = {
        'micro_f1': f1_score(true_matrix, pred_matrix, average='micro', zero_division=0),
        'macro_f1': f1_score(true_matrix, pred_matrix, average='macro', zero_division=0),
        'weighted_f1': f1_score(true_matrix, pred_matrix, average='weighted', zero_division=0),
        'samples_f1': f1_score(true_matrix, pred_matrix, average='samples', zero_division=0),
        'hamming_loss': hamming_loss(true_matrix, pred_matrix),
        'exact_match_ratio': np.all(true_matrix == pred_matrix, axis=1).mean()
    }
    return metrics

def apply_vader_correction(df, prediction_col, output_col='pred_vader_corrected', threshold=0.3):
    """
    Suppress emotion predictions that contradict VADER sentiment polarity
    """
    positive_emotions = ['admiration', 'approval', 'joy', 'gratitude', 'love', 
                        'optimism', 'relief', 'pride', 'amusement', 'excitement', 'caring']
    
    negative_emotions = ['anger', 'annoyance', 'disappointment', 'disapproval', 
                        'disgust', 'fear', 'grief', 'nervousness', 'remorse', 
                        'sadness', 'embarrassment']
    
    corrected_predictions = []
    corrections_applied = 0
    
    for idx, row in df.iterrows():
        pred_dict = row[prediction_col].copy()
        vader_compound = row['vader_compound']
        
        pos_count = sum(pred_dict.get(emo, 0) for emo in positive_emotions if emo in pred_dict)
        neg_count = sum(pred_dict.get(emo, 0) for emo in negative_emotions if emo in pred_dict)
        
        # Positive text: suppress negative emotions if they outnumber positives
        if vader_compound > threshold and neg_count > pos_count:
            for emo in negative_emotions:
                if emo in pred_dict and pred_dict[emo] == 1:
                    pred_dict[emo] = 0
                    corrections_applied += 1
                    if sum(pred_dict.get(e, 0) for e in negative_emotions if e in pred_dict) <= pos_count:
                        break
        
        # Negative text: suppress positive emotions if they outnumber negatives
        elif vader_compound < -threshold and pos_count > neg_count:
            for emo in positive_emotions:
                if emo in pred_dict and pred_dict[emo] == 1:
                    pred_dict[emo] = 0
                    corrections_applied += 1
                    if sum(pred_dict.get(e, 0) for e in positive_emotions if e in pred_dict) <= neg_count:
                        break
        
        corrected_predictions.append(pred_dict)
    
    df[output_col] = corrected_predictions
    print(f"Applied {corrections_applied} corrections")
    return df