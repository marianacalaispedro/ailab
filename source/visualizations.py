"""
Visualization Module
====================================================

Clean and consistent plotting helpers for the Text Mining project.
All charts follow a shared visual language based on the viridis palette
and minimal typography for visual harmony across the report.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter
import networkx as nx
from tqdm import tqdm
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
from typing import Union, Dict, Any
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import nltk


# =============================================================================
# VISUAL STYLE CONFIGURATION
# =============================================================================

class ColorPalette:
    """Centralized color definitions (Viridis-inspired)."""
    DARK_TEXT = "#2D3436"
    MEDIUM_TEXT = "#636E72"
    LIGHT_TEXT = "#A0A4A8"
    WHITE = "white"
    LIGHT_GRAY = "#F8F9FA"
    BORDER_GRAY = "#DDD"

    MAIN_PALLETE = ["#440154", "#3b528b", "#21908d", "#5dc962", "#fde725"]
    RATINGS_PALLETE = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#1a9850"]

class ChartConfig:
    """Default layout configuration for all charts."""
    DEFAULT_WIDTH = 1000
    DEFAULT_HEIGHT = 600
    LARGE_WIDTH = 1200
    LARGE_HEIGHT = 700
    
    TITLE_SIZE = 22
    AXIS_TITLE_SIZE = 16
    TICK_SIZE = 12
    TEXT_SIZE = 11
    LEGEND_SIZE = 12
    FONT_FAMILY = "Arial Black"

# =============================================================================
# SHARED LAYOUT FUNCTION
# =============================================================================

def _apply_base_layout(fig, title: str, x_label: str, y_label: str,
                       width: int, height: int) -> None:
    """Apply unified layout, colors, and typography to a Plotly figure."""
    fig.update_layout(
        title=dict(
            text=title, x=0.5, xanchor="center",
            font=dict(size=ChartConfig.TITLE_SIZE,
                      family=ChartConfig.FONT_FAMILY,
                      color=ColorPalette.DARK_TEXT)
        ),
        xaxis=dict(
            title=dict(text=x_label,
                       font=dict(size=ChartConfig.AXIS_TITLE_SIZE,
                                 color=ColorPalette.MEDIUM_TEXT)),
            tickangle=-45,
            tickfont=dict(size=ChartConfig.TICK_SIZE,
                          color=ColorPalette.LIGHT_TEXT),
            gridcolor=ColorPalette.LIGHT_GRAY,
            linecolor=ColorPalette.BORDER_GRAY
        ),
        yaxis=dict(
            title=dict(text=y_label,
                       font=dict(size=ChartConfig.AXIS_TITLE_SIZE,
                                 color=ColorPalette.MEDIUM_TEXT)),
            tickfont=dict(size=ChartConfig.TICK_SIZE,
                          color=ColorPalette.LIGHT_TEXT),
            gridcolor=ColorPalette.LIGHT_GRAY,
            linecolor=ColorPalette.BORDER_GRAY
        ),
        plot_bgcolor=ColorPalette.WHITE,
        paper_bgcolor=ColorPalette.WHITE,
        width=width,
        height=height,
        margin=dict(l=80, r=80, t=100, b=100),
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

# =============================================================================
# BASIC CHARTS
# =============================================================================

def bar_chart(data: pd.DataFrame, x: str, y: str, title: str,
              labels: Dict[str, str], top_n: int = 10) -> None:
    """Bar chart for the top-N records."""
    data = data.sort_values(by=y, ascending=False).head(top_n)

    # Prepare text labels: if the series is float, format to 2 decimal places
    if pd.api.types.is_float_dtype(data[y]):
        text_vals = data[y].round(2).map(lambda v: f"{v:.2f}")
    else:
        text_vals = data[y].astype(str)

    fig = px.bar(
        data, x=x, y=y, title=title, labels=labels,
        color=y, color_continuous_scale=ColorPalette.MAIN_PALLETE, text=text_vals
    )
    _apply_base_layout(fig, title, labels.get(x, x), labels.get(y, y),
                       ChartConfig.DEFAULT_WIDTH, ChartConfig.DEFAULT_HEIGHT)
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    fig.update_traces(
        texttemplate="%{text}", textposition="outside",
        marker=dict(line=dict(width=1.5, color=ColorPalette.MEDIUM_TEXT),
                    opacity=0.85)
    )
    fig.show()

def pie_chart(data: pd.DataFrame, names_col: str, values_col: str,
              title: str) -> None:
    """Pie chart using the Viridis palette."""
    fig = go.Figure(
        go.Pie(
            labels=data[names_col], values=data[values_col],
            textinfo="label+percent", hoverinfo="label+value+percent",
            marker=dict(colors=ColorPalette.MAIN_PALLETE,
                        line=dict(color=ColorPalette.WHITE, width=2))
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.5),
        width=ChartConfig.DEFAULT_WIDTH,
        height=ChartConfig.DEFAULT_HEIGHT,
        plot_bgcolor=ColorPalette.WHITE,
        paper_bgcolor=ColorPalette.WHITE,
    )
    fig.show()

def donut_chart(data: pd.DataFrame, names_col: str, values_col: str,
                title: str) -> None:
    """Donut chart using the Viridis palette."""

    fig = go.Figure(
        go.Pie(
            labels=data[names_col], values=data[values_col],
            textinfo="label+percent", hoverinfo="label+value+percent",
            marker=dict(colors=ColorPalette.RATINGS_PALLETE,
                        line=dict(color=ColorPalette.WHITE, width=2)),
            hole=0.4,
            sort=False,
            direction="counterclockwise" 
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.5),
        width=ChartConfig.DEFAULT_WIDTH,
        height=ChartConfig.DEFAULT_HEIGHT,
        plot_bgcolor=ColorPalette.WHITE,
        paper_bgcolor=ColorPalette.WHITE,
    )
    fig.show()

def heatmap_chart(data: pd.DataFrame, title: str) -> None:
    """Heatmap with annotations and custom project color palette."""
    
    fig = px.imshow(
        data,
        text_auto=".2f",                     # annot=True, fmt=".2f"
        color_continuous_scale=ColorPalette.MAIN_PALLETE,
        aspect="equal",                     # square=True
        labels={"x": "", "y": "", "color": "Similarity"},
        title=title
    )

    fig.update_layout(
        width=900,
        height=700,
        margin=dict(l=80, r=150, t=80, b=60),
        coloraxis_showscale=True,
    )

    fig.show()

def histogram_chart(data: pd.DataFrame, column: str, title: str,
                    x_label: str, bins: int = 30) -> None:
    """Histogram with unified style."""
    fig = px.histogram(
        data, x=column, nbins=bins, title=title,
        color_discrete_sequence=[ColorPalette.MAIN_PALLETE[0]]
    )
    _apply_base_layout(fig, title, x_label, "Frequency",
                       ChartConfig.DEFAULT_WIDTH, ChartConfig.DEFAULT_HEIGHT)
    fig.update_traces(marker=dict(line=dict(width=1.2, color=ColorPalette.BORDER_GRAY)))
    fig.show()

# =============================================================================
# COMPARATIVE CHARTS
# =============================================================================

def clustered_bar_chart(data: pd.DataFrame, x: str, y_columns: List[str],
                        title: str, labels: Dict[str, str]) -> None:
    """Grouped bar chart comparing multiple numeric series."""
    fig = go.Figure()
    for i, col in enumerate(y_columns):
        fig.add_trace(
            go.Bar(
                name=labels.get(col, col),
                x=data[x], y=data[col],
                text=data[col], textposition="outside",
                marker=dict(color=ColorPalette.MAIN_PALLETE[i % len(ColorPalette.MAIN_PALLETE)],
                            line=dict(width=1.2, color=ColorPalette.BORDER_GRAY))
            )
        )
    _apply_base_layout(fig, title, labels.get(x, x), "Count",
                       ChartConfig.DEFAULT_WIDTH, ChartConfig.DEFAULT_HEIGHT)
    fig.update_layout(
        barmode="group",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.1)
    )
    fig.show()

def clustered_bar_charts(data: pd.DataFrame, x: str, y_columns: List[str],
                         title: str, labels: Dict[str, str], top: int = 5) -> None:
    """Display two side-by-side clustered charts for quick comparison."""
    colors = ColorPalette.MAIN_PALLETE
    top_a = data.sort_values(by=y_columns[0], ascending=False).head(top)
    top_b = data.sort_values(by=y_columns[1], ascending=False).head(top)

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        subplot_titles=[f"Top {top} {labels.get(y_columns[0], y_columns[0])}",
                        f"Top {top} {labels.get(y_columns[1], y_columns[1])}"],
        horizontal_spacing=0.12
    )

    for i, col in enumerate(y_columns):
        for j, subset in enumerate([top_a, top_b]):
            fig.add_trace(
                go.Bar(
                    x=subset[x], y=subset[col],
                    name=labels.get(col, col),
                    marker_color=colors[i % len(colors)],
                    text=subset[col], textposition="outside",
                    showlegend=(j == 0)
                ),
                row=1, col=j + 1
            )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        barmode="group",
        width=ChartConfig.LARGE_WIDTH,
        height=ChartConfig.DEFAULT_HEIGHT,
        # place legend on the right side to avoid overlap with plots
        legend=dict(
            orientation="v",
            x=1.02,
            xanchor="left",
            y=0.5,
            yanchor="middle",
            font=dict(size=ChartConfig.LEGEND_SIZE)
        ),
        plot_bgcolor=ColorPalette.WHITE,
        paper_bgcolor=ColorPalette.WHITE,
        margin=dict(l=80, r=220, t=120, b=80)
    )

    fig.show()

# =============================================================================
# ADVANCED CHARTS
# =============================================================================

def scatter_plot(data: pd.DataFrame, x: str, y: str, title: str,
                 labels: Dict[str, str], size: Optional[str] = None) -> None:
    """Scatter plot using continuous Viridis scale."""
    fig = px.scatter(
        data, x=x, y=y, title=title, labels=labels, size=size,
        color_continuous_scale=ColorPalette.MAIN_PALLETE
    )
    _apply_base_layout(fig, title, labels.get(x, x), labels.get(y, y),
                       ChartConfig.LARGE_WIDTH, ChartConfig.LARGE_HEIGHT)
    fig.show()

def line_plot(data: pd.DataFrame, x: str, y: str, title: str,
              labels: Dict[str, str]) -> None:
    """Line plot for trend visualization."""
    fig = px.line(data, x=x, y=y, title=title, labels=labels,
                 color_discrete_sequence=ColorPalette.MAIN_PALLETE)
    _apply_base_layout(fig, title, labels.get(x, x), labels.get(y, y),
                       ChartConfig.LARGE_WIDTH, ChartConfig.LARGE_HEIGHT)
    fig.show()

def box_plot(data: pd.DataFrame, x: str, y: str, title: str,
             labels: Dict[str, str]) -> None:
    """Box plot for distribution comparison."""
    fig = px.box(data, x=x, y=y, title=title, labels=labels,
                 color_discrete_sequence=ColorPalette.MAIN_PALLETE)
    _apply_base_layout(fig, title, labels.get(x, x), labels.get(y, y),
                       ChartConfig.LARGE_WIDTH, ChartConfig.DEFAULT_HEIGHT)
    fig.show()

# =============================================================================
# Word Clouds
# =============================================================================

def word_cloud_generator(folder_path, df, wc, restaurant_name, vectorisation="bow"):

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    restaurant_df = df[df["title"] == restaurant_name]

    combined_text = " ".join(restaurant_df["text"])

    # Choose vectorizer
    if vectorisation == "bow":
        vectorizer = CountVectorizer(stop_words='english')
    else:
        vectorizer = TfidfVectorizer(stop_words='english')

    X = vectorizer.fit_transform([combined_text])
    words = vectorizer.get_feature_names_out()
    vector = X.toarray().flatten()

    # Create frequency dictionary
    freq_dict = dict(zip(words, vector))

    # Generate and save word cloud
    wc.generate_from_frequencies(freq_dict)

    filename = f"WC_{restaurant_name.replace(' ', '_')}_{vectorisation.upper()}.png"
    save_path = os.path.join(folder_path, filename)
    wc.to_file(save_path)

def wordcloud_from_vectorized(
    folder_path: str,
    filename: str,
    vectorized_df: Union[pd.DataFrame, np.ndarray],
    top_n: int = 20,
) -> Dict[str, Any]:
    """
    Generate and save a word cloud from a vectorized token-frequency matrix.

    Parameters
    ----------
    folder_path : str
        Directory to save the image (created if missing)
    filename : str
        Name of the saved word cloud image
    vectorized_df : DataFrame or ndarray
        Columns=tokens if DataFrame, or 2D array (samples x tokens)
    top_n : int
        Number of top tokens to include

    Returns
    -------
    dict
        Keys: 'path', 'freq_dict', 'wordcloud'
    """

    # Convert ndarray to DataFrame if needed
    if isinstance(vectorized_df, np.ndarray):
        vectorized_df = pd.DataFrame(vectorized_df, columns=[f"token_{i}" for i in range(vectorized_df.shape[1])])

    # Aggregate token counts/weights across rows
    scores = vectorized_df.mean(axis=0)

    # Take top N tokens
    top_scores = scores.sort_values(ascending=False).head(top_n)
    freq_dict = {str(tok): float(val) for tok, val in top_scores.items() if float(val) > 0}

    if not freq_dict:
        raise ValueError("No positive token frequencies found to build a word cloud.")

    # Generate WordCloud
    wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis")
    wc.generate_from_frequencies(freq_dict)

    # Ensure folder exists and save
    os.makedirs(folder_path, exist_ok=True)
    save_path = os.path.join(folder_path, filename)
    wc.to_file(save_path)

    display = True
    if display:
        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')

    return {"path": save_path, "wordcloud": wc}

def wordcloud_from_tokens(
    token_series: pd.Series,
    max_words: int = 100,
    title: Optional[str] = None,
) -> WordCloud:
    """
    Generate and display a word cloud directly from a Series of token lists (BoW).

    This matches the Week 4 'BoW wordcloud' style:
    - Counts raw token frequencies across all documents
    - Does NOT depend on TF-IDF or any vectorizer

    Parameters
    ----------
    token_series : pd.Series
        Each element should be a list (or tuple) of tokens.
    max_words : int, default=100
        Maximum number of words to show in the cloud.
    title : str, optional
        Optional title to show above the figure.

    Returns
    -------
    WordCloud
        The generated WordCloud object.
    """
    # Count tokens
    counter = Counter()
    for tokens in token_series:
        if isinstance(tokens, (list, tuple)):
            counter.update(tokens)

    # Take the most common tokens
    freq_dict = dict(counter.most_common(max_words))
    if not freq_dict:
        raise ValueError("No tokens found to build a word cloud.")

    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white"
    ).generate_from_frequencies(freq_dict)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()

    return wc

def wordcloud_by_rating(
    token_series: pd.Series,
    ratings_series: pd.Series,
    max_words: int = 100,
    title: Optional[str] = None,
) -> WordCloud:
    """
    Wordcloud where:
    - SIZE  = how often a word appears (BoW frequency)
    - COLOR = average star rating of reviews where the word appears,
              mapped to ColorPalette.RATINGS_PALLETE.

    Parameters
    ----------
    token_series : pd.Series
        Each element is a list/tuple of tokens for one review
        (e.g. dataset["03_normalized_text"]).
    ratings_series : pd.Series
        Numeric ratings aligned with token_series (same length/index).
    max_words : int
        Maximum number of words to draw.
    title : str, optional
        Title for the matplotlib figure.
    """
    # 1) Count frequencies and accumulate ratings per token
    freq_counter = Counter()
    rating_sum = defaultdict(float)
    rating_count = defaultdict(int)

    for tokens, rating in zip(token_series, ratings_series):
        if not isinstance(tokens, (list, tuple)):
            continue
        try:
            rating_val = float(rating)
        except (TypeError, ValueError):
            continue

        for tok in tokens:
            freq_counter[tok] += 1
            rating_sum[tok] += rating_val
            rating_count[tok] += 1

    # 2) Top words by frequency
    most_common = freq_counter.most_common(max_words)
    freq_dict = {tok: freq for tok, freq in most_common}

    if not freq_dict:
        raise ValueError("No tokens found to build rating-based word cloud.")

    # 3) Average rating per token
    avg_rating = {}
    for tok, _ in most_common:
        if rating_count[tok] > 0:
            avg_rating[tok] = rating_sum[tok] / rating_count[tok]
        else:
            avg_rating[tok] = np.nan

    global_mean = np.nanmean(list(avg_rating.values())) if avg_rating else 3.0

    # 4) Base wordcloud using frequencies (size)
    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white"
    ).generate_from_frequencies(freq_dict)

    # 5) Colour mapping using YOUR RATINGS_PALLETE
    palette = ColorPalette.RATINGS_PALLETE  # ["#d73027", ..., "#1a9850"]

    def rating_color_func(word, *args, **kwargs):
        r = avg_rating.get(word, global_mean)   # average rating for this word
        # normalise rating 1–5 → 0–1
        r_norm = (r - 1.0) / 4.0
        r_norm = min(max(r_norm, 0.0), 1.0)
        # map to palette index 0..4
        idx = int(round(r_norm * (len(palette) - 1)))
        idx = max(0, min(idx, len(palette) - 1))
        return palette[idx]

    # recolor using the rating-based colour function
    wc = wc.recolor(color_func=rating_color_func)

    # 6) Plot
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()

    return wc

def wordcloud_by_pos(
    token_series: pd.Series,
    max_words: int = 100,
    title: Optional[str] = None,
    pos_series: Optional[pd.Series] = None,
) -> WordCloud:
    """
    Wordcloud where:
    - SIZE  = frequency (BoW)
    - COLOR = dominant POS of each word (noun / verb / adjective / other).

    Parameters
    ----------
    token_series : pd.Series
        Each element is a list/tuple of tokens for one review.
    max_words : int
        Maximum number of words to include.
    title : str, optional
        Title for the plot.
    pos_series : pd.Series, optional
        If provided, each element is a list of POS tags aligned with token_series.
        If None, POS tags are inferred with nltk.pos_tag.
    """
    def coarse_pos(tag: str) -> str:
        """Map Penn Treebank POS tags to coarse categories."""
        if tag.startswith("N"):
            return "NOUN"
        if tag.startswith("V"):
            return "VERB"
        if tag.startswith("J"):
            return "ADJ"
        return "OTHER"

    freq_counter = Counter()
    pos_counts = defaultdict(lambda: Counter())

    
    if pos_series is not None:
        for tokens, tags in zip(token_series, pos_series):
            if not isinstance(tokens, (list, tuple)):
                continue
            if not isinstance(tags, (list, tuple)):
                continue
            for tok, tag in zip(tokens, tags):
                freq_counter[tok] += 1
                pos_counts[tok][coarse_pos(str(tag))] += 1
    else:
        for tokens in token_series:
            if not isinstance(tokens, (list, tuple)):
                continue
            tagged = nltk.pos_tag(tokens)
            for tok, tag in tagged:
                freq_counter[tok] += 1
                pos_counts[tok][coarse_pos(tag)] += 1

   
    most_common = freq_counter.most_common(max_words)
    freq_dict = {tok: freq for tok, freq in most_common}

    if not freq_dict:
        raise ValueError("No tokens found to build POS-based word cloud.")

    word_pos = {}
    for tok, _ in most_common:
        if pos_counts[tok]:
            word_pos[tok] = pos_counts[tok].most_common(1)[0][0]
        else:
            word_pos[tok] = "OTHER"

    pos_colors = {
        "NOUN": ColorPalette.MAIN_PALLETE[2],   # green-ish
        "VERB": ColorPalette.MAIN_PALLETE[1],   # blue-ish
        "ADJ":  ColorPalette.MAIN_PALLETE[0],   # purple
        "OTHER": "#95a5a6",                     # neutral gray
    }

    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white"
    ).generate_from_frequencies(freq_dict)

    def pos_color_func(word, *args, **kwargs):
        pos = word_pos.get(word, "OTHER")
        return pos_colors.get(pos, pos_colors["OTHER"])

    wc = wc.recolor(color_func=pos_color_func)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()

    return wc

# =============================================================================
# tree maps
# =============================================================================

def treemap_chart(data: pd.DataFrame, path_col: list, value_col: str, title: str) -> None:

    fig = px.treemap(
        data,
        path=path_col,
        values=value_col,
        color=value_col,
        color_continuous_scale="Viridis",
        title=title
    )

    fig.update_traces(
        texttemplate="%{label}",
        textfont=dict(size=16),
        hovertemplate="<b>%{label}</b><br>frequency=%{value}<extra></extra>"
    )

    fig.update_layout(
        width=1200,
        height=700,
        margin=dict(t=50, l=25, r=25, b=25),
        coloraxis_colorbar=dict(
            title=dict(text="Frequency", side="right")
        )
    )

    fig.show()

def build_pos_token_freq(token_series, pos_series):
    """
    Build a table: POS | token | frequency
    """
    freq = defaultdict(Counter)

    for tokens, tags in zip(token_series, pos_series):
        if not isinstance(tokens, (list, tuple)) or not isinstance(tags, (list, tuple)):
            continue

        for tok, tag in zip(tokens, tags):
            # coarse POS mapping (same as wordcloud_by_pos)
            if tag.startswith("N"):
                pos_group = "NOUN"
            elif tag.startswith("V"):
                pos_group = "VERB"
            elif tag.startswith("J"):
                pos_group = "ADJ"
            else:
                pos_group = "OTHER"

            freq[pos_group][tok] += 1

    # Convert to DataFrame
    rows = []
    for pos_group, token_counter in freq.items():
        for tok, count in token_counter.items():
            rows.append([pos_group, tok, count])

    df = pd.DataFrame(rows, columns=["pos", "token", "frequency"])
    df = df.sort_values(by="frequency", ascending=False)

    return df

# =============================================================================
# 
# =============================================================================

def most_common_words(df, text_col="text", category_col=None, top_n=20):
    """
    Find the most common words overall and optionally by category.
    """
    vectorizer = CountVectorizer(stop_words='english')
    
    # Overall frequencies
    X = vectorizer.fit_transform(df[text_col])
    word_counts = X.toarray().sum(axis=0)
    words = vectorizer.get_feature_names_out()
    overall_freq = pd.DataFrame({"word": words, "count": word_counts})
    overall_freq = overall_freq.sort_values(by="count", ascending=False).head(top_n)

    print("🔠 Most Common Words Overall:")
    print(overall_freq)

    # If category_col provided, compute by category
    if category_col:
        print("\n📊 Most Common Words by Category:")
        category_results = {}
        for cat, group in df.groupby(category_col):
            X_cat = vectorizer.fit_transform(group[text_col])
            word_counts_cat = X_cat.toarray().sum(axis=0)
            words_cat = vectorizer.get_feature_names_out()
            freq_df = pd.DataFrame({"word": words_cat, "count": word_counts_cat})
            freq_df = freq_df.sort_values(by="count", ascending=False).head(top_n).reset_index(drop=True)
            category_results[cat] = freq_df
            print(f"\n➡️ {cat}:")
            print(freq_df)
        #return overall_freq, category_results

    #return overall_freq
    
# =============================================================================
# GEO VISUALIZATION
# =============================================================================

def extract_coordinates(url: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract (lat, lon) coordinates from a Google Maps URL."""
    match = re.search(r"@([-+]?\d*\.\d+),([-+]?\d*\.\d+)", str(url))
    return (float(match.group(1)), float(match.group(2))) if match else (None, None)

def plot_restaurant_map(dataset: pd.DataFrame, color_by: str,
                        url_col: str = "url", title_col: str = "title",
                        category_col: str = "categoryName",
                        reviews_col: str = "reviewsCount",
                        zoom: int = 10, height: int = 700) -> None:
    
    """Map of restaurants colored by a chosen attribute."""

    dataset[["latitude", "longitude"]] = dataset[url_col].apply(
        lambda u: pd.Series(extract_coordinates(u))
    )
    dataset = dataset.dropna(subset=["latitude", "longitude"]).copy()

    color_args = (
        dict(color_continuous_scale=ColorPalette.RATINGS_PALLETE)
        if pd.api.types.is_numeric_dtype(dataset[color_by])
        else dict(color_discrete_sequence=px.colors.qualitative.Safe)
    )

    fig = px.scatter_mapbox(
        dataset, lat="latitude", lon="longitude", color=color_by,
        hover_name=title_col,
        hover_data={category_col: True, reviews_col: True, color_by: True},
        zoom=zoom, title=f"Restaurant Locations by {color_by}", **color_args
    )
    fig.update_layout(mapbox_style="carto-positron", height=height)
    fig.show()

#--------------------------------------------------------------------------------
# TERM FREQUENCY PLOTTING FUNCTION
#--------------------------------------------------------------------------------

def plot_term_frequency(df, nr_terms, df_name, show=True):
        
    # Create the Seaborn bar plot
    plt.figure(figsize=(10, 8))
    sns_plot = sns.barplot(x='frequency', y='words', data=df.head(nr_terms))  # Plotting top 20 terms for better visualization
    plt.title('Top 20 Term Frequencies of {}'.format(df_name))
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    if show==True:
        plt.show()

    fig = sns_plot.get_figure()
    plt.close()

    return fig

# =============================================================================
# CO-OCCURRENCE MATRIX FROM TOKENS
# =============================================================================

def build_cooccurrence_matrix_tokens(token_series, top_n=200):
    """
    Builds a term-term co-occurrence matrix from a Series of token lists.
    Matches the professor's Week 4/5 co-occurrence method:
    - uses frequent tokens only (top_n)
    - counts co-occurrence within each review (no TF-IDF)
    """

    from collections import Counter, defaultdict
    import numpy as np
    import pandas as pd

    # 1) Count frequencies
    freq_counter = Counter()
    for tokens in token_series:
        if isinstance(tokens, (list, tuple)):
            freq_counter.update(tokens)

    # 2) Select vocab of top-N frequent tokens
    vocab = [w for w, _ in freq_counter.most_common(top_n)]
    vocab_index = {w: i for i, w in enumerate(vocab)}

    # 3) Initialise matrix
    cooc_matrix = np.zeros((top_n, top_n), dtype=int)

    # 4) Build co-occurrence counts (per review)
    for tokens in token_series:
        if not isinstance(tokens, (list, tuple)):
            continue

        # Keep only tokens in vocab
        filtered = [w for w in tokens if w in vocab_index]
        unique = set(filtered)  # professor simplification: set() to avoid double counting

        for w1 in unique:
            i = vocab_index[w1]
            for w2 in unique:
                if w1 != w2:
                    j = vocab_index[w2]
                    cooc_matrix[i, j] += 1

    # 5) Build DataFrame
    cooc_df = pd.DataFrame(cooc_matrix, index=vocab, columns=vocab)
    return cooc_df

def plot_cooccurrence_heatmap(cooc_df, top_n=40):
    top_words = cooc_df.sum(axis=1).head(top_n).index
    filtered = cooc_df.loc[top_words, top_words]

    mask = np.triu(np.ones_like(filtered, dtype=bool))

    plt.figure(figsize=(18, 18))
    sns.heatmap(
        filtered,
        cmap="YlGnBu",
        mask=mask,
        square=True
    )
    plt.title(f"Co-occurrence Heatmap (Top {top_n} words)")
    plt.tight_layout()
    plt.show()

def plot_cooccurrence_network(cooc_df, top_n=25, min_weight=50):
    """
    Cleaner co-occurrence network:
    - Uses only top_n most frequent words
    - Removes weak co-occurrences (< min_weight)
    - Produces a graph similar to the professor’s
    """
    # Select words
    words = cooc_df.sum(axis=1).sort_values(ascending=False).head(top_n).index
    filtered = cooc_df.loc[words, words]

    G = nx.Graph()

    # Add nodes with size = frequency
    for w in words:
        G.add_node(w, size=filtered.loc[w].sum())

    # Add only strong edges
    for w1 in words:
        for w2 in words:
            if w1 >= w2:
                continue
            weight = filtered.loc[w1, w2]
            if weight >= min_weight:   # <<<<<<<<<< filter weak connections
                G.add_edge(w1, w2, weight=weight)

    # Layout
    plt.figure(figsize=(14, 12))
    pos = nx.spring_layout(G, k=1.8, seed=42)  # more spacing

    # Edge widths
    edge_widths = [0.02 * G[u][v]['weight'] for u, v in G.edges()]

    # Node sizes
    node_sizes = [data['size'] * 0.1 for _, data in G.nodes(data=True)]

    # Draw
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.25, edge_color="gray")
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="#7ec0ee")
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

    plt.title(f"Clean Co-occurrence Network (Top {top_n} words)")
    plt.axis("off")
    plt.show()
