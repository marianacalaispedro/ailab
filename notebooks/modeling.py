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