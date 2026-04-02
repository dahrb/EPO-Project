"""
Useful text processors 

Last Updated: 02.04.26

Status: Done
"""

import spacy
import pandas as pd
import numpy as np

from gensim.models import KeyedVectors
from gensim.models.doc2vec import Doc2Vec
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
        
        self.nlp = spacy.load('en_core_web_sm',disable=["parser", "ner"])

        #lemma for proper nouns
        ruler = self.nlp.get_pipe("attribute_ruler")
        patterns = [[{"TAG": {"IN": ["NNP", "NNPS"]}}]]
        attrs = {"POS": "NOUN"}
        ruler.add(patterns=patterns, attrs=attrs)
        
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

        if self.embedding == 'word2vec':
            self.word2vec = KeyedVectors.load("Word2Vec-google-300d", mmap='r')
            self.words = set(self.word2vec.index_to_key)

        elif self.embedding == 'law2vec':
            self.word2vec = KeyedVectors.load_word2vec_format('Law2Vec.200d.txt', binary=False)
            self.words = set(self.word2vec.index_to_key)

        elif self.embedding == 'patent2vec':
            self.word2vec = KeyedVectors.load("Patent2Vec_1.0", mmap='r')
            self.words = set(self.word2vec.wv.index_to_key)

        elif self.embedding == 'doc2vec':
            self.word2vec = Doc2Vec.load('Doc2Vec_1.0')

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
