#imports
import spacy
import os
import pandas as pd
from gensim.models import Word2Vec
import multiprocessing
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class MyDocs(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):

        self.tokens = 0

        for fname in os.listdir(self.dirname):

            if fname.endswith('.txt'):

                print(f'{fname}')

                data = pd.read_table(os.path.join(self.dirname,fname),header=None,chunksize=10000)

                for chunk in data:
                    df = chunk
                    df = PreProcessing(df,nlp)   
                    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df['nlp'])]

                    for doc in documents:                          
                        yield doc

            elif fname == 'X_Train':

                print(f'{fname}')

                X_train = pd.read_pickle('X_Train')
                X_train = [doc for doc in nlp.pipe(X_train.tolist())]
                X_train = [[tok.lower_ for tok in doc if (tok.is_alpha)] for doc in X_train]
                X_train = [TaggedDocument(doc, [i]) for i, doc in enumerate(X_train)]

               
                for doc in X_train:

                    yield doc 


def PreProcessing(df,nlp):
    
    #filter only english and exclude pdf links to orig docs
    df = df[(df[4]=='en') & (df[5]!='PDFEP')]
    df = df[[7]]
    #convert col to string
    df[7] = df[7].astype(str)
    #remove html tags
    df['nlp'] = df[7].apply(RemoveHTMLTags)
    #create spacy doc
    df["nlp"] = [doc for doc in nlp.pipe(df['nlp'].tolist())] #n_process = multiprocessing.cpu_count() - 1)]
    #lowercase and alpha/ punctuation/ removes individual letters such as 'f' present in original text due to abbrev
    #df['nlp'] = [[tok.lower_ for tok in doc if (tok.is_alpha)] for doc in df['nlp']]
    df['nlp'] = [[tok.lower_ for tok in doc if (tok.is_alpha)] for doc in df['nlp']]
    
    return df

#remove HTML tags
def RemoveHTMLTags(string):
     
    # Print string after removing tags
    x = re.compile(r'<[^>]+>').sub(' ', string)

    return x

if __name__ == '__main__':

    nlp = spacy.blank('en')
    nlp.max_length = 5000000
    docs = MyDocs('.')
    model = Doc2Vec(documents = docs, vector_size=300, min_count=10, dm=1, window=5, workers= multiprocessing.cpu_count() -1, seed=42, epochs=3)
    print('Model trained')
    model.save('Doc2Vec_1.0')
    print('Model Saved')
 

    