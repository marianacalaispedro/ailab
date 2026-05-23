"""
MainPipeline Class
=================
A comprehensive text preprocessing class for NLP tasks, including:
- Cleaning, tokenization, lemmatization
- TF-IDF, Bag-of-Words, Doc2Vec vectorization
- Co-occurrence matrix computation
- Language detection and translation
- Feature extraction for NER
"""

# --- Standard Libraries ---
import re
from collections import defaultdict, Counter

# --- Data Handling ---
import numpy as np
import pandas as pd

# --- Progress Bars ---
from tqdm import tqdm

# --- NLP / Text Processing ---
import nltk
import emoji
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import words
from unidecode import unidecode

# --- Machine Learning / Feature Extraction ---
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# --- Language Detection & Translation ---
try:
    from langdetect import detect as ld_detect
except Exception:
    ld_detect = None

try:
    import langid
except Exception:
    langid = None

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None


import pandas as pd
from collections import Counter
import Levenshtein
from jellyfish import jaro_winkler_similarity

class MainPipeline(BaseEstimator):
    """
    MainPipeline class for text preprocessing, vectorization, translation,
    and feature extraction.
    """
    
    def __init__(self, 
                 print_output=False, 
                 no_emojis=True, 
                 no_hashtags=True,
                 hashtag_retain_words=True,
                 no_newlines=True,
                 no_urls=True,
                 no_punctuation=True,
                 no_stopwords=True,
                 custom_stopwords=None, #DELETE?
                 stopwords_tokeep=None,  #DELETE?   
                 convert_diacritics=True, 
                 lowercase=True, 
                 lemmatized=True,
                 list_pos=["n","v","a","r","s"], #DELETE?
                 pos_tags_list="no_pos", #DELETE?
                 tokenized_output=False): #DELETE?
        self.print_output = print_output
        self.no_emojis = no_emojis
        self.no_hashtags = no_hashtags
        self.hashtag_retain_words = hashtag_retain_words
        self.no_newlines = no_newlines
        self.no_urls = no_urls
        self.no_punctuation = no_punctuation
        self.no_stopwords = no_stopwords
        self.custom_stopwords = custom_stopwords if custom_stopwords is not None else []
        self.stopwords_tokeep = set(stopwords_tokeep) if stopwords_tokeep else set() 
        self.convert_diacritics = convert_diacritics
        self.lowercase = lowercase
        self.lemmatized = lemmatized
        self.list_pos = list_pos
        self.pos_tags_list = pos_tags_list
        self.tokenized_output = tokenized_output


    # ---------------------------------------------
    # 1. Regex Cleaner
    # ---------------------------------------------
    def regex_cleaner(self, raw_text):
        """
        Cleans text with regex: emojis, hashtags, URLs, newlines, punctuation.

        Parameters
        ----------
        raw_text : str
            Raw input text.

        Returns
        -------
        str
            Cleaned text.
        """
        text = str(raw_text) if raw_text else ""

        if self.no_emojis:
            text = emoji.demojize(text, delimiters=("emoji_", ""))

        if self.no_hashtags:
            if self.hashtag_retain_words:
                text = re.sub(r"([#@])", "", text)
            else:
                text = re.sub(r"([#@]\w+)", "", text)

        if self.no_newlines:
            text = re.sub(r"\n", " ", text)

        if self.no_urls:
            text = re.sub(r"(?:\w+:/{2})?(?:www\.)?([a-z\d\-]+)\.(?:[a-z\d\.]{2,})(?:/[a-zA-Z/\d]*)?", r"\1", text)

        if self.no_punctuation:
            #text = re.sub(r"[\u0021-\u0026\u0028-\u002C\u002E-\u002F\u003A-\u003F\u005B-\u005F\u2010-\u2028\ufeff`]+", "", text)
            text = re.sub(r"[\u0021-\u0026\u0028-\u002C\u002E-\u002F\u003A-\u003F\u005B-\u005E\u0060\u2010-\u2028\ufeff`]+", "", text)
            text = re.sub(r"'(?=[A-Z\s])|(?<=[a-z\.\?\!\,\s])'", "", text)
            text = re.sub(r'\s*-\s*', " ", text)

        text = re.sub(r'\s+', " ", text).strip()
        return text

    # ---------------------------------------------
    # 2. Repeated Character Reduction
    # ---------------------------------------------
    @staticmethod
    def repeated_chars(token, max_repeat=2):
        """
        Reduces repeated characters to max_repeat.

        Example: 'soooo' -> 'soo'
        """
        return re.sub(r'(.)\1{%d,}' % max_repeat, r'\1' * max_repeat, token)

    # ---------------------------------------------
    # 3. Lemmatization
    # ---------------------------------------------
    def lemmatize_all(self, token):
        """
        Lemmatize token with multiple POS tags.
        """
        lemmatizer = nltk.stem.WordNetLemmatizer()
        for pos in self.list_pos:
            token = lemmatizer.lemmatize(token, pos)
        return token

    # ---------------------------------------------
    # 4. Main Text Pipeline
    # ---------------------------------------------
    def main_pipeline(self, raw_text, stemmed=False, treat_repeated_chars=False):
        """
        Full preprocessing pipeline: cleaning, tokenization, stopwords,
        lemmatization, diacritics, lowercasing, etc.

        Returns tokenized list or detokenized string based on settings.
        """
        #custom_stopwords = self.custom_stopwords if self.custom_stopwords is not None else []
        
        # stopwords_tokeep = self.stopwords_tokeep if self.stopwords_tokeep is not None else set()
        
        #if not isinstance(stopwords_tokeep, set):
        #    stopwords_tokeep = set(stopwords_tokeep)
            
        #list_pos = self.list_pos if self.list_pos is not None else ["n","v","a","r","s"]

        if self.print_output:
            print("Input:", raw_text)

        text = self.regex_cleaner(raw_text)
        tokens = nltk.word_tokenize(text)

        # Handle contractions
        contractions = {"'m": "am", "n't": "not", "'s": "is", "'re": "are", "'ve": "have", "'ll": "will", "'d": "would"}
        # Correct way
        tokens = [t for t in tokens]  # start with original tokens
        
        for pattern, repl in contractions.items():
            tokens = [re.sub(pattern, repl, t) for t in tokens]

        if treat_repeated_chars:
            tokens = [self.repeated_chars(t) for t in tokens]

        if self.no_stopwords:
            stopwords_set = set(nltk.corpus.stopwords.words("english"))
            stopwords_set.update(self.custom_stopwords)           # add custom stopwords
            stopwords_set -= self.stopwords_tokeep               # remove the words you want to keep
            tokens = [t for t in tokens if t.lower() not in stopwords_set]

        if self.convert_diacritics:
            tokens = [unidecode(t) for t in tokens]

        if self.lemmatized:
            tokens = [self.lemmatize_all(t) for t in tokens]

        if stemmed:
            stemmer = nltk.stem.PorterStemmer()
            tokens = [stemmer.stem(t) for t in tokens]

        if self.lowercase:
            tokens = [t.lower() for t in tokens]

        if self.pos_tags_list in {"pos_list", "pos_tuples"}:
            pos_tuples = nltk.pos_tag(tokens)
            if self.pos_tags_list == "pos_list":
                return [pos[1] for pos in pos_tuples]
            return pos_tuples

        if self.tokenized_output:
            return tokens

        return TreebankWordDetokenizer().detokenize(tokens)

    # ---------------------------------------------
    # 5. Vectorization
    # ---------------------------------------------
    @staticmethod
    def vectorize_texts(texts, vectorizer_type="tfidf", max_features=1000, ngram_range=(1, 1), vector_size=100):
        """
        Vectorizes text using TF-IDF, Count Vectorizer, or Doc2Vec.

        Returns document-term matrix or document vectors.
        """
        if vectorizer_type in ["tfidf", "count"]:
            processed_texts = [" ".join(x) if isinstance(x, (list, tuple)) else str(x) for x in texts]
            if vectorizer_type == "count":
                vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
            else:
                vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
            dtm = vectorizer.fit_transform(processed_texts)
            return dtm, vectorizer

        elif vectorizer_type == "doc2vec":
            tokenized_texts = [x if isinstance(x, (list, tuple)) else str(x).split() for x in texts]
            tagged_docs = [TaggedDocument(words=doc, tags=[i]) for i, doc in enumerate(tokenized_texts)]
            model = Doc2Vec(vector_size=vector_size, window=5, min_count=2, workers=4, epochs=40)
            model.build_vocab(tagged_docs)
            model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
            dtm = [model.infer_vector(doc.words) for doc in tagged_docs]
            return dtm, model
        else:
            raise ValueError("Choose 'tfidf', 'count', or 'doc2vec'")

    # ---------------------------------------------
    # 6. Co-occurrence Matrix
    # ---------------------------------------------
    @staticmethod
    def cooccurrence_matrix(vectorized_df):
        """
        Computes co-occurrence matrix from a DataFrame of vectorized features.
        """
        X_dense = vectorized_df.astype(float).values
        feature_names = vectorized_df.columns.tolist()
        n_words = len(feature_names)
        co_matrix = np.zeros((n_words, n_words), dtype=int)
        word_idx = {w: i for i, w in enumerate(feature_names)}

        for doc_vector in tqdm(X_dense, desc="Computing co-occurrences"):
            present_indices = np.where(doc_vector > 0)[0]
            for i in present_indices:
                for j in present_indices:
                    if i != j:
                        co_matrix[i, j] += 1

        cooc_df = pd.DataFrame(co_matrix, index=feature_names, columns=feature_names)
        cooc_df = cooc_df.reindex(cooc_df.sum().sort_values(ascending=False).index, axis=0)\
                         .reindex(cooc_df.sum().sort_values(ascending=False).index, axis=1)
        return cooc_df

    # ---------------------------------------------
    # 7. Translation & Language Detection
    # ---------------------------------------------
    @staticmethod
    def process_and_translate_dataset(dataset: pd.DataFrame, text_column: str = '00_before_translating_cleaning') -> pd.DataFrame:
        """
        Detect language and translate non-English text to English.
        Adds 'text_for_pipeline' for downstream processing.
        """
        def safe_detect_lang_langdetect(text):
            if ld_detect is None: return None
            try: return ld_detect(str(text))
            except: return None

        def safe_detect_lang_langid(text):
            if langid is None: return None
            try: return langid.classify(str(text))[0]
            except: return None

        dataset['lang_langdetect'] = dataset[text_column].apply(safe_detect_lang_langdetect)
        dataset['lang_langid'] = dataset[text_column].apply(safe_detect_lang_langid)

        def need_translation_row(row):
            ld = row['lang_langdetect']
            li = row['lang_langid']
            if ld is None and li is None:
                return True
            if ld is not None and ld != 'en':
                return True
            if li is not None and li != 'en':
                return True
            return False
        
        dataset['needs_translation'] = dataset.apply(need_translation_row, axis=1)

        translator = None
        if GoogleTranslator is not None:
            translator = GoogleTranslator(source='auto', target='en')

        def translate_safe(text):
            if translator is None: return str(text)
            try: return translator.translate(str(text))
            except: return str(text)

        dataset['text_translated'] = dataset.apply(
            lambda r: translate_safe(r[text_column]) if r['needs_translation'] else r[text_column],
            axis=1
        )

        dataset['text_for_pipeline'] = dataset['text_translated']
        return dataset

    # ---------------------------------------------
    # 8. NER Feature Extraction
    # ---------------------------------------------
    @staticmethod
    def word2features(token_list, POS_list, i):
        """
        Extracts features for a token at position i.
        """
        word = token_list[i]
        postag = POS_list[i]
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2]
        }
        if i > 0:
            word1 = token_list[i-1]; postag1 = POS_list[i-1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True

        if i < len(token_list)-1:
            word1 = token_list[i+1]; postag1 = POS_list[i+1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True

        return features

    @staticmethod
    def sent2features(token_list, POS_list):
        return [MainPipeline.word2features(token_list, POS_list, i) for i in range(len(token_list))]

    @staticmethod
    def align_bio(doc, tokens):
        """
        Aligns spaCy entities to tokens in BIO format.
        """
        bio_labels = ["O"] * len(tokens)
        for ent in doc.ents:
            ent_tokens = [t.text for t in ent]
            for i in range(len(tokens)):
                if tokens[i:i+len(ent_tokens)] == ent_tokens:
                    bio_labels[i] = f"B-{ent.label_}"
                    for j in range(i+1, i+len(ent_tokens)):
                        bio_labels[j] = f"I-{ent.label_}"
                    break
        return bio_labels

    @staticmethod
    def align_bio_to_custom_tokens(text, tokens, nlp, equivalence_table):
        """
        Aligns spaCy entities to custom token list with BIO format.
        """
        doc = nlp(text)
        bio = ["O"] * len(tokens)
        for ent in doc.ents:
            ent_tokens = [t.text for t in ent]
            for i in range(len(tokens)):
                if tokens[i:i+len(ent_tokens)] == ent_tokens:
                    bio[i] = "B" + equivalence_table[ent.label_]
                    for j in range(i+1, i+len(ent_tokens)):
                        bio[j] = "I" + equivalence_table[ent.label_]
                    break
        return bio


def main_pipeline(
    raw_text,
    stemmed=False,
    treat_repeated_chars=False,
    **pipeline_kwargs
):
    """
    Functional wrapper around MainPipeline.

    Allows calling the pipeline like a function while still
    using a class-based architecture.

    Parameters
    ----------
    raw_text : str
        Input text.
    stemmed : bool
        Whether to apply stemming.
    treat_repeated_chars : bool
        Whether to normalize repeated characters.
    **pipeline_kwargs :
        All preprocessing configuration options passed to MainPipeline.__init__.

    Returns
    -------
    str or list
        Processed text.
    """
    pipeline = MainPipeline(**pipeline_kwargs)
    return pipeline.main_pipeline(
        raw_text,
        stemmed=stemmed,
        treat_repeated_chars=treat_repeated_chars
    )

# --- Similarity metrics ---
def levenshtein_sim(w1, w2):
    # normalize distance into similarity [0,1]
    dist = Levenshtein.distance(w1, w2)
    return 1 - dist / max(len(w1), len(w2))

def jaro_sim(w1, w2):
    return jaro_winkler_similarity(w1, w2)

def jaccard_sim(w1, w2):
    set1, set2 = set(w1), set(w2)
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

def combined_similarity(w1, w2, weights=(0.4, 0.4, 0.2)):
    """Weighted average of Levenshtein, Jaro-Winkler, Jaccard"""
    lev = levenshtein_sim(w1, w2)
    jaro = jaro_sim(w1, w2)
    jacc = jaccard_sim(w1, w2)
    return weights[0]*lev + weights[1]*jaro + weights[2]*jacc

# --- Correction function ---
def correct_word(word, vocab, word_counts, threshold=0.85):
    """
    Correct a word by finding the most frequent similar candidate.
    threshold: minimum similarity to consider correction
    """
    if word in word_counts and word_counts[word] > 5:
        # frequent enough, assume correct
        return word
    
    best_word, best_score = word, 0
    for candidate in vocab:
        if candidate == word:
            continue
        score = combined_similarity(word, candidate)
        if score > threshold and score > best_score:
            # prefer more frequent candidate
            if word_counts[candidate] >= word_counts[word]:
                best_word, best_score = candidate, score
    return best_word


# --- Apply to dataset ---
def correct_tokens_column(dataset, token_col='normalized_tokens'):
    # Flatten all tokens to build frequency dictionary
    all_tokens = [w for tokens in dataset[token_col] for w in tokens]
    word_counts = Counter(all_tokens)
    vocab = list(word_counts.keys())
    
    dataset['words_corrected'] = dataset[token_col].apply(
        lambda tokens: [correct_word(w, vocab, word_counts) for w in tokens]
    )
    return dataset

from collections import Counter


def correct_tokens_column_string(
    dataset,
    text_col,
    output_col='words_corrected'
):
    # Step 1: tokenize using main_pipeline
    tokenized_col = dataset[text_col].apply(
        lambda x: main_pipeline(
            raw_text=x,
            no_emojis=True,             
            no_hashtags=True,           
            hashtag_retain_words=True,  
            no_newlines=True,           
            no_urls=True,
            no_punctuation=True,  
            no_stopwords=True,
            custom_stopwords=[],
            stopwords_tokeep=[],
            convert_diacritics=True,
            lowercase=True,
            lemmatized=True,
            list_pos=[],
            pos_tags_list='no_pos',
            tokenized_output=True,
            stemmed=False,
            treat_repeated_chars=True
        )
    )

    # Step 2: build frequency dictionary
    all_tokens = [w for tokens in tokenized_col for w in tokens]
    word_counts = Counter(all_tokens)
    vocab = list(word_counts.keys())

    # Step 3: correct tokens and convert back to string
    dataset[output_col] = tokenized_col.apply(
        lambda tokens: " ".join(
            correct_word(w, vocab, word_counts) for w in tokens
        )
    )

    return dataset