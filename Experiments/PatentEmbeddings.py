"""
Script to train Patent embeddings models i.e. Patent2Vec and PatentDoc2Vec

-- pre-trained Patent2Vec availiable in './Models/Patent2Vec_1.0' 
-- pre-trained PatentDoc2Vec available in './Models/Doc2Vec_1.0

Last Updated: 02.04.26

Status: Done
"""

import os
import re
import multiprocessing
import pandas as pd
import spacy
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument

class PatentCorpus:
    """Unified iterator for Word2Vec (sentences) and Doc2Vec (tagged documents)."""
    def __init__(self, dirname, nlp, mode='word2vec', stopwords=False, numbers=False, lemmatisation=False):
        self.dirname = dirname
        self.nlp = nlp
        self.mode = mode  # 'word2vec' or 'doc2vec'
        self.stopwords = stopwords
        self.numbers = numbers
        self.lemmatisation = lemmatisation
        self.total_tokens = 0
        self.doc_count = 0

    def _clean_tokens(self, tokens):
        """Pythonic filtering and transformation logic."""
        return [
            (t.lemma_.lower() if self.lemmatisation else t.lower_)
            for t in tokens
            if (t.is_alpha or (self.numbers and t.is_digit))
            and not (self.stopwords and t.is_stop)
            and (len(t) > 1 or t.lower_ in {'a', 'i'})
        ]

    def _process_text(self, text):
        """Removes HTML and converts to spaCy Doc."""
        clean_text = re.compile(r'<[^>]+>').sub(' ', str(text))
        return self.nlp(clean_text)

    def __iter__(self):
        self.total_tokens = 0
        self.doc_count = 0

        for fname in os.listdir(self.dirname):
            #raw Text Files
            if fname.endswith('.txt'):
                print(f"Processing: {fname}")
                data_chunks = pd.read_table(os.path.join(self.dirname, fname), header=None, chunksize=10000)

                for chunk in data_chunks:
                    #filter English and exclude PDF links
                    valid_rows = chunk[(chunk[4] == 'en') & (chunk[5] != 'PDFEP')][7]
                    
                    for text in valid_rows:
                        doc = self._process_text(text)
                        
                        if self.mode == 'doc2vec':
                            tokens = self._clean_tokens(doc)
                            yield TaggedDocument(tokens, [self.doc_count])
                            self.doc_count += 1
                        else:
                            for sent in doc.sents:
                                tokens = self._clean_tokens(sent)
                                if tokens:
                                    self.total_tokens += len(tokens)
                                    yield tokens

            #X_Train files
            elif fname == 'X_Train':
                print(f"Processing: {fname}")
                x_train_list = pd.read_pickle(fname).tolist()
                
                for i, doc in enumerate(self.nlp.pipe(x_train_list)):
                    tokens = self._clean_tokens(doc)
                    if self.mode == 'doc2vec':
                        yield TaggedDocument(tokens, [f"TRAIN_{i}"])
                    else:
                        self.total_tokens += len(tokens)
                        yield tokens

if __name__ == '__main__':
    #setup spaCy
    nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"])
    
    #if using Word2Vec, we need the sentencizer
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    
    #nlp.max_length = 5000000 #Doc2Vec
    nlp.max_length = 3000000 #Word2Vec
    
    #initialize Corpus (Change mode to 'doc2vec' for Doc2Vec training)
    current_mode = 'word2vec' 
    corpus = PatentCorpus('.', nlp, mode=current_mode, numbers=False, stopwords=False)

    #model Training
    cpus = multiprocessing.cpu_count() - 1

    if current_mode == 'word2vec':
        model = Word2Vec(vector_size=300, min_count=10, sg=1, window=5, workers=cpus, seed=42, epochs=3)
        model.build_vocab(corpus)
        model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
        model.save('Patent2Vec_1.0.model')
    else:
        model = Doc2Vec(documents=corpus, vector_size=300, min_count=10, dm=1, window=5, workers=cpus, seed=42, epochs=3)
        model.save('Doc2Vec_1.0.model')

    print(f'Model trained and saved in {current_mode} mode.')