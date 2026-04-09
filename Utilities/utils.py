"""
Useful text processors 

Last Updated: 02.04.26

Status: Done
"""

import spacy
import pandas as pd
import numpy as np
import os
from pathlib import Path

from gensim.models import KeyedVectors
from gensim.models.doc2vec import Doc2Vec
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


_EMBEDDING_MODEL_CACHE = {}
_EMBEDDING_WORDS_CACHE = {}
_SPACY_NLP_CACHE = None
_SPACY_RULER_CONFIGURED = False


def _load_text_embeddings_robust(path: Path):
    """Load text-format embeddings while skipping malformed rows.

    Expected format: first line '<vocab_size> <vector_dim>', then one token per line
    with exactly vector_dim float values.
    """
    with path.open("r", encoding="utf-8", errors="ignore") as file_handle:
        header = file_handle.readline().strip().split()
        if len(header) != 2:
            raise ValueError(f"Invalid embedding header in {path}")

        try:
            vector_dim = int(header[1])
        except ValueError as exc:
            raise ValueError(f"Invalid vector dimension in header for {path}") from exc

        words = []
        vectors = []

        for line_number, raw_line in enumerate(file_handle, start=2):
            parts = raw_line.strip().split()
            if not parts:
                continue

            if len(parts) != vector_dim + 1:
                continue

            token = parts[0]
            try:
                values = np.asarray(parts[1:], dtype=np.float32)
            except ValueError:
                continue

            if values.shape[0] != vector_dim:
                continue

            words.append(token)
            vectors.append(values)

    if not vectors:
        raise ValueError(f"No valid vectors found in {path}")

    keyed_vectors = KeyedVectors(vector_size=vector_dim)
    keyed_vectors.add_vectors(words, np.vstack(vectors))
    return keyed_vectors


def _get_cached_embedding(embedding: str):
    """Load embedding model once per process and reuse it across instances."""
    if embedding in _EMBEDDING_MODEL_CACHE:
        return _EMBEDDING_MODEL_CACHE[embedding], _EMBEDDING_WORDS_CACHE.get(embedding)

    # Determine model directory (works from any working directory)
    model_dir = Path(__file__).parent.parent / "Models"

    def _load_kv_safe(path: Path):
        try:
            return KeyedVectors.load(str(path), mmap='r')
        except ValueError:
            return KeyedVectors.load(str(path), mmap=None)

    if embedding == 'word2vec':
        path = model_dir / "Word2Vec-google-300d"
        model = _load_kv_safe(path)
        words = set(model.index_to_key)
    elif embedding == 'law2vec':
        path = model_dir / "Law2Vec.200d.txt"
        model = _load_text_embeddings_robust(path)
        words = set(model.index_to_key)
    elif embedding == 'patent2vec':
        path = model_dir / "Patent2Vec_1.0"
        model = _load_kv_safe(path)
        words = set(model.wv.index_to_key)
    elif embedding == 'doc2vec':
        path = model_dir / "Doc2Vec_1.0"
        model = Doc2Vec.load(str(path))
        words = None
    else:
        raise ValueError(f"Unsupported embedding: {embedding}")

    _EMBEDDING_MODEL_CACHE[embedding] = model
    _EMBEDDING_WORDS_CACHE[embedding] = words
    return model, words


def _get_cached_spacy_nlp():
    """Load spaCy model once per process and reuse it across TextProcess instances."""
    global _SPACY_NLP_CACHE, _SPACY_RULER_CONFIGURED

    if _SPACY_NLP_CACHE is None:
        _SPACY_NLP_CACHE = spacy.load('en_core_web_sm', disable=["parser", "ner"])

    if not _SPACY_RULER_CONFIGURED:
        ruler = _SPACY_NLP_CACHE.get_pipe("attribute_ruler")
        patterns = [[{"TAG": {"IN": ["NNP", "NNPS"]}}]]
        attrs = {"POS": "NOUN"}
        ruler.add(patterns=patterns, attrs=attrs)
        _SPACY_RULER_CONFIGURED = True

    return _SPACY_NLP_CACHE

class PatentExtract:
    """
    A class to extract various attributes and text from XML nodes representing patent data.
    """

    def __init__(self, node):
        """
        Initialize the PatentExtract object with an XML node.

        Args:
            node: The XML node to extract data from.
        """
        self.node = node

    def getAttribute(self, attr):
        """
        Extract a specific attribute from the XML node.

        Args:
            attr (str): The name of the attribute to extract.

        Returns:
            list: A list containing the attribute value, or an empty list if the attribute is not found.
        """
        attribute = []
        try:
            attribute.append(self.node.getAttribute(attr))
        except AttributeError:
            # Log or handle the error if the node does not have the attribute
            pass
        return attribute

    def singleTagTextExtract(self, tag):
        """
        Extract text content from a single tag within the XML node.

        Args:
            tag (str): The name of the tag to extract text from.

        Returns:
            list: A list of text content from the specified tag.
        """
        text_NODE = self.node.getElementsByTagName(tag)
        text = []

        for i in text_NODE:
            try:
                text.append(i.firstChild.data)
            except AttributeError:
                # Handle cases where the tag has no child data
                pass
        return text

    def multiTagTextExtract(self, tag, inner_tag, attr=None):
        """
        Extract text content and optional attributes from nested tags within the XML node.

        Args:
            tag (str): The outer tag name.
            inner_tag (str): The inner tag name to extract text from.
            attr (str, optional): The attribute name to extract from the outer tag. Defaults to None.

        Returns:
            tuple or list: A tuple of (text, attributes) if attr is provided, otherwise a list of text content.
        """
        text_NODE = self.node.getElementsByTagName(tag)
        text = []
        attr_list = []

        attr_flag = attr is not None

        for i in text_NODE:
            if attr_flag:
                try:
                    attr_list.append(i.getAttribute(attr))
                except AttributeError:
                    # Handle cases where the attribute is not found
                    pass

            for j in i.getElementsByTagName(inner_tag):
                try:
                    text.append(j.firstChild.data)
                except AttributeError:
                    # Handle cases where the inner tag has no child data
                    pass

        return (text, attr_list) if attr_flag else text

    def classificationExtract(self):
        """
        Extract and format classification data from the XML node.

        Returns:
            list: A list of formatted classification strings.
        """
        classification_list = []

        # Extract classification components
        section = self.multiTagTextExtract(tag='classification-ipcr', inner_tag='section')
        class_ = self.multiTagTextExtract(tag='classification-ipcr', inner_tag='class')
        subclass = self.multiTagTextExtract(tag='classification-ipcr', inner_tag='subclass')
        main_group = self.multiTagTextExtract(tag='classification-ipcr', inner_tag='main-group')
        sub_group = self.multiTagTextExtract(tag='classification-ipcr', inner_tag='subgroup')

        # Combine components into formatted classification strings
        for i, sec in enumerate(section):
            try:
                sec_class = ''.join([sec, class_[i], subclass[i]])
            except IndexError:
                # Break if any component is missing
                break
            try:
                groups = '/'.join([main_group[i], sub_group[i]])
            except IndexError:
                groups = ''

            output = ' '.join([sec_class, groups])
            classification_list.append(output)

        return classification_list

    def cited_decisions(self):
        """
        Extract cited decisions from the XML node.

        Returns:
            list: A list of cited decisions, each formatted as 'code appeal_num year'.
        """
        text_NODE = self.node.getElementsByTagName("ep-cited-decisions")
        text = []

        for i in text_NODE:
            for j in i.getElementsByTagName('ep-cited-decision'):
                try:
                    code = j.getAttribute('code')
                    appeal_num = j.getElementsByTagName('ep-appeal-num')[0].firstChild.data
                    year = j.getElementsByTagName('ep-year')[0].firstChild.data
                    text.append(' '.join([code, appeal_num, year]))
                except (AttributeError, IndexError):
                    # Handle cases where attributes or child data are missing
                    pass

        return text 

class TableCreator:
    """
    A class to create and manage a table structure for storing patent-related data.
    """

    def __init__(self):
        """
        Initialize the TableCreator object with empty lists for various attributes.
        """
        self.procedure_lang = []  # Language used in the procedure
        self.court_type = []  # Type of court handling the case
        self.appeal_num = []  # Appeal number of the case
        self.fact_lang = []  # Language of the document facts
        self.summary_facts = []  # Summary of the facts
        self.legal_provisions = [] #Legal Provisions
        self.decision_reason = []  # Reasons for the decision
        self.order = []  # Order issued by the court
        self.date = []  # Date of the decision
        self.ecli = []  # European Case Law Identifier
        self.title = []  # Title of the invention
        self.board_code = []  # Code of the board handling the case
        self.keywords = []  # Keywords related to the case
        self.reference = []  # References cited in the case
        self.classification = []  # Classification of the case
        self.appno = [] # Application number
        self.cited = [] # Cited Cases


    def create_table(self):
        """
        Create a pandas DataFrame from the stored attributes.

        The DataFrame is created using a dictionary where the keys are column names
        and the values are the corresponding attribute lists.
        """
        dictionary = {
            'Reference': self.reference,
            'Procedure Language': self.procedure_lang,
            'Court Type': self.court_type,
            'Appeal Number': self.appeal_num,
            'Document Language': self.fact_lang,
            'Summary Facts': self.summary_facts,
            'Decision Reasons': self.decision_reason,
            'Order': self.order,
            'Legal Provisions': self.legal_provisions,
            'Date': self.date,
            'ECLI': self.ecli,
            'Invention Title': self.title,
            'Board Code': self.board_code,
            'Classification': self.classification,
            'Keywords': self.keywords,
            'App No': self.appno,
            'Cited':self.cited
        }

        # Create a pandas DataFrame from the dictionary
        self.df = pd.DataFrame(dictionary)

    def append(self, name, info):
        """
        Append information to a specific attribute list.

        Args:
            name (list): The attribute list to append to.
            info: The information to append to the list.
        """
        name.append(info)

class TextProcess(BaseEstimator, TransformerMixin):
    """
    Processes the text using SpaCy
    
    Parameters
    ----------
    stopwords : bool, optional
    numbers : bool, optional
    lemmatisation : bool, optional
    """  
    
    def __init__(self, stopwords=False, numbers=False, lemmatisation=False):
        self.nlp = _get_cached_spacy_nlp()
        
        self.stopwords = stopwords
        self.numbers = numbers
        self.lemmatisation = lemmatisation
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [
            [
                (t.lemma_.lower() if self.lemmatisation else t.lower_)
                for t in doc
                if (t.is_alpha or (self.numbers and t.is_digit))
                and not (self.stopwords and t.is_stop)
                and (len(t) > 1 or t.lower_ in {'a', 'i'})
            ]
            for doc in X
        ]

class Word2VecTransform(BaseEstimator, TransformerMixin):
    """Processes the Word2Vec/Doc2Vec embeddings and creates the final vectors

    Parameters
    ----------
    embedding : str, optional
    opposition : bool, optional
    
    """

    def __init__(self, embedding = 'word2vec',opposition=False):

        self.embedding = embedding
        self.opposition = opposition
        self.word2vec, self.words = _get_cached_embedding(self.embedding)

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):

        if self.embedding == 'patent2vec':
            vecs = [np.mean([self.word2vec.wv[i] for i in doc if i in self.words], axis=0).tolist() for doc in X]

        elif self.embedding == 'doc2vec':
            vecs = [self.word2vec.infer_vector(doc) for doc in X]

        else:
            vecs = [np.mean([self.word2vec[i] for i in doc if i in self.words], axis=0).tolist() for doc in X]
        
        if self.opposition is True:
            return pd.DataFrame(vecs)
        else:
            return vecs 


# ---------------------------------------------------------------------------
# Shared CLI helpers (used by run_experiment.py and run_deep_learning_experiment.py)
# ---------------------------------------------------------------------------

def parse_bool(value):
    """Parse a boolean from a string argument.

    Accepts: true, 1, yes, y, t  →  True
             anything else       →  False
    """
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def to_numpy_labels(y):
    """Convert a label array-like to a 1-D NumPy array."""
    if isinstance(y, pd.DataFrame):
        return y.iloc[:, 0].to_numpy()
    if isinstance(y, pd.Series):
        return y.to_numpy()
    return np.asarray(y).reshape(-1)
