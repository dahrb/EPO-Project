import spacy
from spacy.tokens import Doc, Token
import os
import pandas as pd
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler 
import joblib 
import multiprocessing
import pickle
import time
import itertools
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import xgboost as xgb 
import spacy
from spacy.tokens import Doc, Token
import os
import pandas as pd
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import Binarizer
import joblib 
import multiprocessing
import pickle
import time
import itertools
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import numpy as np
import xgboost as xgb 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import re
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer

class PatentExtract():

    def __init__(self, node):
        self.node = node

    def getAttribute(self,attr):
        attribute = []
        attribute.append(self.node.getAttribute(f"{attr}"))
        
        return attribute

    def singleTagTextExtract(self,tag):
        text_NODE = self.node.getElementsByTagName(f"{tag}")
        text = []

        for i in text_NODE:
            text.append(i.firstChild.data)
        
        return text
        
    def multiTagTextExtract(self,tag,inner_tag,attr=None):
        text_NODE = self.node.getElementsByTagName(f"{tag}")
        text = []
        attr = []
        attr_flag = False

        for i in text_NODE:
            if attr != None:
                attr.append(i.getAttribute(f'{attr}'))
                attr_flag = True
                print(attr)

            for j in i.getElementsByTagName(f'{inner_tag}'):
                text.append(j.firstChild.data)

        if attr_flag == True:
            return text, attr
        else:
            return text     

class TableCreator():

    def __init__(self):
        self.procedure_lang = []
        self.court_type = []
        self.appeal_num = []
        self.fact_lang = []
        self.summary_facts = []
        self.decision_reason = []
        self.order = []
        self.date = []
        self.ecli = []
        self.title = []
        self.board_code = []
        self.keywords = []
        self.reference = []
        self.classification = []

    def create_table(self):
        
        dictionary = {'Reference':self.reference,
                      'Procedure Language':self.procedure_lang,
                      'Court Type':self.court_type,
                      'Appeal Number':self.appeal_num,
                      'Document Language':self.fact_lang,
                      'Summary Facts':self.summary_facts,
                      'Decision Reasons':self.decision_reason,
                      'Order':self.order,
                      'Date':self.date,
                      'ECLI':self.ecli,
                      'Invention Title':self.title,
                      'Board Code':self.board_code,
                      'Classification':self.classification,
                      'Keywords':self.keywords}
        
        self.df = pd.DataFrame(dictionary) 

    def append(self,name,info):
        name.append(info)

class TextProcess(BaseEstimator, TransformerMixin):
    
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
        X_ = X.copy()
        all_docs = []

        for doc in X_:

            individual_doc = []

            for token in doc:

                if self.numbers is True:

                   if token.is_alpha or token.is_digit:
                       
                       tok = token
                   else:
                       continue

                else: 
                    if token.is_alpha:

                        tok = token

                    else:
                        continue

                if self.stopwords is True:

                    if tok.is_stop:
                        continue
    
                if (len(tok)>1) or (tok.lower_ in ['a','i']):

                    #maybe change order so that it lowercases before finding lemmatisation
                    
                    if self.lemmatisation is True:
                        individual_doc.append(tok.lemma_.lower())

                    else:
                        individual_doc.append(tok.lower_)   

            all_docs.append(individual_doc)

        return all_docs

class Word2VecTransform(BaseEstimator, TransformerMixin):

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

class Experiments():

    def __init__(self,models=[], experiment = '1',opposition=False, cv_num = 3, num_iter = 100, train_both=False, boolean_edit = [], repeat = 10, no_grid = False):

        self.experiment = experiment
        self.opposition = opposition
        self.num_iter = num_iter
        self.train_both = train_both
        self.no_grid = no_grid
        self.repeat = repeat

        if not boolean_edit:
            self.boolean_edit = None 

        else:
            self.boolean_edit = boolean_edit

        if self.no_grid is False:
            self.set_params()

        self.params = []
        self.results = []

        if 'linear' in models:
            self.params.append(self.linear)
        
        if 'logistic' in models:
            self.params.append(self.logistic)
        
        if 'forest' in models:
            self.params.append(self.forest)

        if 'xgboost' in models:
            self.params.append(self.xgboost)

        if self.experiment == '2':
            tscv = TimeSeriesSplit(n_splits=cv_num)
            self.cv_num = tscv
        else:
            self.cv_num = cv_num

        self.scoring = {"Accuracy": "accuracy", "F1":"f1", 'Precision':'precision', 'Recall':'recall', "MCC": make_scorer(matthews_corrcoef), "AUC":"roc_auc"}
     
    def set_params(self):

        if self.opposition is True:
            self.num = 'num__'
        else:
            self.num = ''
        ###########################################
        self.linear = {
        'clf': [LinearSVC(random_state=42)],
        'vect': [TfidfVectorizer(tokenizer=lambda x:x,preprocessor=lambda x:x),Word2VecTransform],
        'scaler':[StandardScaler()],
        f'vect__{self.num}ngram_range': [(1,1),(1,2),(1,3),(2,2),(2,3),(3,3),(1,4),(2,4),(3,4),(4,4)],
        'clf__C': [0.1, 1, 10, 100],
        f'vect__{self.num}norm':[None,'l2'],
        f'vect__{self.num}min_df':[2,5,10],
        f'vect__{self.num}use_idf':[True,False],
        'word2vec':['word2vec','law2vec','patent2vec','doc2vec']
        }
        ############################################
        self.logistic = {
        'clf': [LogisticRegression()],
        'vect': [TfidfVectorizer(tokenizer=lambda x:x,preprocessor=lambda x:x),Word2VecTransform],
        'scaler':[StandardScaler()],
        f'vect__{self.num}ngram_range': [(1,1),(1,2),(1,3),(2,2),(2,3),(3,3),(1,4),(2,4),(3,4),(4,4)],
        'clf__C': [0.1, 1, 10, 100],
        'clf__solver':['lbfgs','sag'],
        'clf__penalty':[None,'l2'],
        'clf__max_iter':[100, 250, 500],
        f'vect__{self.num}norm':[None,'l2'],
        f'vect__{self.num}min_df':[2,5,10],
        f'vect__{self.num}use_idf':[True,False],
        'word2vec':['word2vec','law2vec','patent2vec','doc2vec']
        }
        self.forest = {
        'clf': [RandomForestClassifier(random_state=42)],
        'vect': [TfidfVectorizer(tokenizer=lambda x:x,preprocessor=lambda x:x),Word2VecTransform],
        'scaler':[StandardScaler()],
        f'vect__{self.num}ngram_range': [(1,1),(1,2),(1,3),(2,2),(2,3),(3,3),(1,4),(2,4),(3,4),(4,4)],
        'clf__n_estimators': [100, 200, 300],
        'clf__max_features':['sqrt','log2'],
        'clf__max_depth': [10, 50, 100, None],
        f'vect__{self.num}norm':[None,'l2'],
        f'vect__{self.num}min_df':[2,5,10],
        f'vect__{self.num}use_idf':[True,False],
        'word2vec':['word2vec','law2vec','patent2vec','doc2vec']
        }
        self.xgboost = {
        'clf': [xgb.XGBClassifier(random_state=42, objective='binary:logistic')],
        'vect': [TfidfVectorizer(tokenizer=lambda x:x,preprocessor=lambda x:x),Word2VecTransform],
        'scaler':[StandardScaler()],
        f'vect__{self.num}ngram_range': [(1,1),(1,2),(1,3),(2,2),(2,3),(3,3),(1,4),(2,4),(3,4),(4,4)],
        'clf__n_estimators': [100, 200, 300],
        'clf__num_boost_round':[100, 200, 300],
        'clf__learning_rate':[0.01, 0.02, 0.05],
        'clf__gamma':[ 0.0, 0.1, 0.2],
        f'vect__{self.num}norm':[None],
        f'vect__{self.num}min_df':[2,5,10],
        f'vect__{self.num}use_idf':[True,False],
        'word2vec':['word2vec','law2vec','patent2vec','doc2vec']
        }

    def training_helper_preprocess(self,X_train):
        
        tp = TextProcess()

        if self.opposition is False:
            self.X_train = [doc for doc in tp.nlp.pipe(X_train['New Summary Facts'].tolist())]
        else:
            self.X_train = X_train.copy()
            self.X_train['New Summary Facts'] = [doc for doc in tp.nlp.pipe(X_train['New Summary Facts'].tolist())]

    def define_own_params(self,results):

        param = {
        'clf': [self.model_selector(results['algo'])],
        'vect': [TfidfVectorizer(tokenizer=lambda x:x,preprocessor=lambda x:x),Word2VecTransform],
        'scaler':[StandardScaler()],
        'word2vec':[results['embedding']]
        }

        for i,j in results['params'].items():

            results['params'][i] = [j]

        param.update(results['params'])

        self.params.append(param)

    def model_selector(self, clf):
        if clf.startswith('Log'):
            return LogisticRegression(random_state=42) 
        elif clf.startswith('Lin'):
            return LinearSVC(random_state=42)
        elif clf.startswith('Ran'):
            return RandomForestClassifier(random_state=42)
        else:
            return xgb.XGBClassifier(random_state=42, objective='binary:logistic')

    def training_loop(self, X_train, y_train):

        self.training_helper_preprocess(X_train)

        for param in self.params:

            #classifier
            self.clf = param['clf'][0]
            self.vects_n = param['vect'][0]
            self.vects_we = param['vect'][1]
            self.scal = param['scaler'][0]
            self.word2vec = param['word2vec']

            #getting arguments by
            #popping out classifier
            param.pop('clf')
            param.pop('vect')
            param.pop('scaler')
            param.pop('word2vec')

            print(self.clf)

            print('N-grams')

            ##############################################################################
            if self.opposition is True:
                preprocessor = ColumnTransformer(transformers=[("num", self.vects_n,'New Summary Facts'),("Cats",Binarizer(),['1','2'])])
                steps = Pipeline([('vect', preprocessor), ('clf', self.clf)])

            else:
                steps = Pipeline([('vect', self.vects_n), ('clf', self.clf)])

            clf = self.name_process(self.clf)

            for i in itertools.product([True, False],[True, False],[True, False]): #stopwords, numbers, lemmatisation

                if self.boolean_edit is None:
                    self.training_ngram_core(i,param,steps,y_train,clf)
                else:

                    if len(self.boolean_edit) == 1:
                        if i[0] == self.boolean_edit[0]:
                            self.training_ngram_core(i,param,steps,y_train,clf)
                            print(i)
                    else:
                        if (i[0] == self.boolean_edit[0]) and (i[1] == self.boolean_edit[1]) and (i[2] == self.boolean_edit[2]):
                            self.training_ngram_core(i,param,steps,y_train,clf)
                            print(i)                  

            print('END!!!!!!!!!!!!!!!!')

            if self.train_both is True:
                self.training_loop_we(X_train,y_train)
                results_temp = pd.DataFrame(self.results)
                results_temp.to_pickle(f'results_{clf}_{self.experiment}_{self.opposition}.pkl')

        return self.results
    
    def training_ngram_core(self,i,param,steps,y_train,clf):
        
        tp = TextProcess(stopwords= i[0], numbers= i[1], lemmatisation= i[2])

        if self.opposition is True:
            X_train_ = self.X_train.copy()
            X_train_['New Summary Facts'] = tp.fit_transform(X_train_['New Summary Facts'])
                                
        else:
            X_train_ = self.X_train.copy()
            X_train_ = tp.fit_transform(X_train_)

        if self.no_grid is False:
            CV = RandomizedSearchCV(steps,param_distributions=param,n_iter=self.num_iter,cv=self.cv_num,n_jobs=-1,scoring=self.scoring,refit='F1')

        else:

            if self.experiment == '2':
                cv = self.cv_num 

            else:    
                cv = RepeatedStratifiedKFold(n_splits=self.cv_num, n_repeats=self.repeat, random_state=42)
            
            CV = GridSearchCV(steps,param_grid=param,cv=cv,n_jobs=-1,scoring=self.scoring,refit='F1',return_train_score=True)

        time_taken = time.time()
        CV.fit(X_train_, y_train)
        time_overall = time.time() - time_taken
        print("done in %0.3fs" % (time_overall))
        #storing result
        self.results.append({
                'algo': self.clf,
                'best score': CV.best_score_,
                'best params': CV.best_params_,
                'full results': CV.cv_results_,
                'stopwords': tp.stopwords,
                'lemma': tp.lemmatisation,
                'numbers': tp.numbers,
                'time taken': time_overall,
                'embedding': None })
        
        results_super_temp = pd.DataFrame(self.results)
        results_super_temp.to_pickle(f'results__{clf}_{i[0]}{i[1]}{i[2]}_{self.experiment}_{self.opposition}')

    def training_loop_we(self,X_train,y_train):

        self.training_helper_preprocess(X_train)

        for param in self.params:

            #classifier
            self.clf = param['clf'][0]
            self.vects_n = param['vect'][0]
            self.vects_we = param['vect'][1]
            self.scal = param['scaler'][0]
            self.word2vec = param['word2vec']

            clf = self.name_process(self.clf)
            
            #getting arguments by
            #popping out classifier
            param.pop('clf')
            param.pop('vect')
            param.pop('scaler')
            param.pop('word2vec')
        
            ######################## num ######################## 
            try:
                param.pop(f'vect__{self.num}min_df')
            except:
                pass
            try:
                param.pop(f'vect__{self.num}norm')
            except:
                pass
            try:
                param.pop(f'vect__{self.num}ngram_range')
            except:
                pass
            try:
                param.pop(f'vect__{self.num}use_idf')
            except:
                pass

            steps = Pipeline([('scal', self.scal), ('clf',self.clf)])
            
            for i in [True, False]: #stopwords

                if self.boolean_edit is None:
                    self.training_w2v_core(i,param,steps,y_train,clf)
                
                else:
                    if i == self.boolean_edit[0]:
                        self.training_w2v_core(i,param,steps,y_train,clf)
          

    def training_w2v_core(self, i, param, steps, y_train, clf):
        tp = TextProcess(stopwords= i)

        if self.opposition is True:
            X_train_ = self.X_train.copy()
            X_train_['New Summary Facts'] = tp.fit_transform(X_train_['New Summary Facts'])
                                    
        else:
            X_train_ = self.X_train.copy()
            X_train_ = tp.fit_transform(X_train_)

        for embedding in self.word2vec:
            print(embedding)

            w2v = self.vects_we(embedding)

            if self.opposition is True:
                preprocessor = ColumnTransformer(transformers=[("num", Word2VecTransform(embedding=embedding,opposition=True),'New Summary Facts'),("Cats",Binarizer(),['1','2'])])
                X_train_Embed = preprocessor.fit_transform(X_train_)

            else:
                X_train_Embed = w2v.fit_transform(X_train_)


            if self.no_grid is False:
                CV = GridSearchCV(steps,param_grid=param,cv=self.cv_num,n_jobs=-1,scoring=self.scoring,refit='F1')

            else:

                if self.experiment == '2':
                    cv = self.cv_num

                else:    
                    cv = RepeatedStratifiedKFold(n_splits=self.cv_num, n_repeats=self.repeat, random_state=42)
                
                CV = GridSearchCV(steps,param_grid=param,cv=cv,n_jobs=-1,scoring=self.scoring,refit='F1',return_train_score=True)

            time_taken = time.time()
            CV.fit(X_train_Embed, y_train)
            time_overall = time.time() - time_taken
            print("done in %0.3fs" % (time_overall))

            #storing result
            self.results.append({
                    'algo': self.clf,
                    'best score': CV.best_score_,
                    'best params': CV.best_params_,
                    'full results': CV.cv_results_,
                    'stopwords': tp.stopwords,
                    'lemma': tp.lemmatisation,
                    'numbers': tp.numbers,
                    'time taken': time_overall,
                    'embedding': embedding})
            
            results_super_temp = pd.DataFrame(self.results)
            results_super_temp.to_pickle(f'results__WE_{clf}_{embedding}_{i}_{self.experiment}_{self.opposition}')

    def name_process(self,clf):

        clf = str(clf)
        clf = re.split('\(',clf)
        
        return clf[0]

