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

def distribution_bar_chart(data: pd.DataFrame, x: str, y: str, title: str,
                           x_label: str, y_label: str, annotation_text: str = None,
                           color_scale: str = None) -> None:
    """
    Bar chart for distribution data with optional annotation.
    """
   
    if color_scale is None:
        color_scale = ColorPalette.MAIN_PALLETE
    
    fig = px.bar(
        data,
        x=x,
        y=y,
        title=title,
        labels={x: x_label, y: y_label},
        text=y,
        color=x,
        color_continuous_scale=color_scale
    )
    
    fig.update_traces(
        texttemplate='%{text:,}',
        textposition='outside',
        marker_line_width=1,
        marker_line_color='white'
    )
    
    fig.update_layout(
        width=ChartConfig.DEFAULT_WIDTH,
        height=ChartConfig.DEFAULT_HEIGHT,
        xaxis=dict(tickmode='linear', dtick=1),
        showlegend=False
    )
    
    if annotation_text:
        fig.add_annotation(
            x=0.95,
            y=0.95,
            xref='paper',
            yref='paper',
            text=annotation_text,
            showarrow=False,
            font=dict(size=14, color='white'),
            bgcolor='rgba(0,0,0,0.7)',
            bordercolor='white',
            borderwidth=1
        )
    
    fig.show()


def donut_chart(data: pd.DataFrame, names_col: str, values_col: str,
                            title: str, center_text: str = None,
                            colors: List[str] = None) -> None:
    """
    Donut chart with center text annotation.
    """
   
    if colors is None:
        colors = ['#21908d', '#440154']
    
    fig = go.Figure(data=[go.Pie(
        labels=data[names_col],
        values=data[values_col],
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent',
        textposition='auto',
        pull=[0, 0.05]  # Slightly pull the second slice
    )])
    
    annotations_list = []
    if center_text:
        annotations_list.append(dict(
            text=center_text,
            x=0.5, y=0.5,
            font_size=14,
            showarrow=False
        ))
    
    fig.update_layout(
        title=title,
        width=700,
        height=500,
        annotations=annotations_list
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
                    x_label: str, bins: int = 30, show_mean: bool = False, show_median: bool = False) -> None:
    """Histogram with unified style."""
    fig = px.histogram(
        data, x=column, nbins=bins, title=title,
        color_discrete_sequence=[ColorPalette.MAIN_PALLETE[0]]
    )

    if show_mean:
        mean_val = data[column].mean()
        fig.add_vline(
            x=mean_val, line_dash="dash", line_color="red",
            annotation_text=f"Mean: {mean_val:.1f}",
            annotation_position="top"
        )
    
    if show_median:
        median_val = data[column].median()
        fig.add_vline(
            x=median_val, line_dash="dot", line_color=ColorPalette.MAIN_PALLETE[3],
            annotation_text=f"Median: {median_val:.1f}",
            annotation_position="bottom"
        )

        
    _apply_base_layout(fig, title, x_label, "Frequency",
                       ChartConfig.DEFAULT_WIDTH, ChartConfig.DEFAULT_HEIGHT)
    fig.update_traces(marker=dict(line=dict(width=1.2, color=ColorPalette.BORDER_GRAY)))
    fig.show()

# =============================================================================
# COMPARATIVE CHARTS
# =============================================================================

#NOT BEIGN USED
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
def horizontal_box_plot(data: pd.DataFrame, metrics: List[str], title: str,
                        log_scale: bool = False) -> None:
    """
    Horizontal box plot comparing multiple metrics.
    """
    # Prepare data
    plot_data = pd.DataFrame()
    for metric in metrics:
        temp_df = pd.DataFrame({
            'Metric': [metric] * len(data),
            'Value': data[metric]
        })
        plot_data = pd.concat([plot_data, temp_df], ignore_index=True)
    
    # Create horizontal box plot
    fig = px.box(
        plot_data,
        x='Value',
        y='Metric',
        title=title,
        labels={'Value': 'Count', 'Metric': ''},
        color='Metric',
        color_discrete_sequence=ColorPalette.MAIN_PALLETE,
        orientation='h'
    )
    
    if log_scale:
        fig.update_xaxes(type="log")  # Fixed: changed from update_xaxis to update_xaxes
    
    fig.update_layout(
        width=ChartConfig.LARGE_WIDTH,
        height=ChartConfig.DEFAULT_HEIGHT,
        title=dict(
            text=title, x=0.5, xanchor="center",
            font=dict(size=ChartConfig.TITLE_SIZE,
                      family=ChartConfig.FONT_FAMILY,
                      color=ColorPalette.DARK_TEXT)
        ),
        xaxis=dict(
            title=dict(text='Count', font=dict(size=ChartConfig.AXIS_TITLE_SIZE,
                                               color=ColorPalette.MEDIUM_TEXT)),
            tickfont=dict(size=ChartConfig.TICK_SIZE,
                          color=ColorPalette.LIGHT_TEXT),
            gridcolor=ColorPalette.LIGHT_GRAY,
            linecolor=ColorPalette.BORDER_GRAY
        ),
        yaxis=dict(
            title=dict(text='', font=dict(size=ChartConfig.AXIS_TITLE_SIZE,
                                          color=ColorPalette.MEDIUM_TEXT)),
            tickfont=dict(size=ChartConfig.TICK_SIZE,
                          color=ColorPalette.DARK_TEXT),
            gridcolor=ColorPalette.LIGHT_GRAY,
            linecolor=ColorPalette.BORDER_GRAY
        ),
        plot_bgcolor=ColorPalette.WHITE,
        paper_bgcolor=ColorPalette.WHITE,
        showlegend=False
    )
    fig.show()

# =============================================================================
# Word Clouds
# =============================================================================
def wordcloud_from_text(text_series: pd.Series, title: str = None, 
                        max_words: int = 100, colormap: str = 'viridis',
                        save_path: str = None) -> WordCloud:
    """
    Generate and display a word cloud directly from a Series of text strings.
    
    Parameters
    ----------
    text_series : pd.Series
        Series containing text strings (e.g., dataset['01_minimal_preprocessing'])
    title : str, optional
        Title for the plot
    max_words : int, default=100
        Maximum number of words to show
    colormap : str, default='viridis'
        Color scheme for the word cloud
    save_path : str, optional
        Path to save the image
        
    Returns
    -------
    WordCloud
        The generated WordCloud object
    """
    # Combine all text
    combined_text = " ".join(text_series.astype(str))
    
    # Count word frequencies
    words = combined_text.split()
    word_freq = Counter(words)
    
    # Take top words
    top_words = dict(word_freq.most_common(max_words))
    
    if not top_words:
        raise ValueError("No words found to build a word cloud.")
    
    # Generate word cloud
    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap=colormap,
        max_words=max_words
    ).generate_from_frequencies(top_words)
    
    # Display
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    if title:
        plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Word cloud saved to {save_path}")
    
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

    print(" Most Common Words Overall:")
    print(overall_freq)

    # If category_col provided, compute by category
    if category_col:
        print("\n Most Common Words by Category:")
        category_results = {}
        for cat, group in df.groupby(category_col):
            X_cat = vectorizer.fit_transform(group[text_col])
            word_counts_cat = X_cat.toarray().sum(axis=0)
            words_cat = vectorizer.get_feature_names_out()
            freq_df = pd.DataFrame({"word": words_cat, "count": word_counts_cat})
            freq_df = freq_df.sort_values(by="count", ascending=False).head(top_n).reset_index(drop=True)
            category_results[cat] = freq_df
            print(f"\n {cat}:")
            print(freq_df)
        #return overall_freq, category_results

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
