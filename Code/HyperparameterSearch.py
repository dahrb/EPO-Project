from sklearnex import patch_sklearn
patch_sklearn()
import spacy
from spacy.tokens import Doc, Token
import os
import pandas as pd
from sklearn.model_selection import cross_validate
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
from sklearn.model_selection import StratifiedKFold
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
from Classes import TextProcess, Word2VecTransform

class PreProcessText(BaseEstimator, TransformerMixin):

    def __init__(self,stopwords=False,numbers=False,lemma = False, exp2 = False, opposition=False):

        self.stopwords = stopwords
        self.numbers = numbers
        self.lemma = lemma
        self.exp2 = exp2
        self.opposition = opposition

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):

        tp = TextProcess(stopwords= self.stopwords, numbers= self.numbers, lemmatisation= self.lemma)

        if self.opposition is True:
            X_train_ = X.copy()
            X_train_['New Summary Facts'] = tp.fit_transform(X_train_['New Summary Facts'])
                                
        else:
            X_train_ = X.copy()
            X_train_ = tp.fit_transform(X_train_)

        return X_train_


class HyperparameterExperiment():

    def __init__(self,models=[], experiment = '1',opposition=False, cv_num_in = 10, cv_num_out = 10, num_iter = 100, train_both=False, repeat = 1, no_grid = False, input_representation = 'N-Gram'):

        self.experiment = experiment
        self.opposition = opposition
        self.num_iter = num_iter
        self.train_both = train_both
        self.no_grid = no_grid
        self.repeat = repeat
        self.models = models

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

        self.cv_num_in = cv_num_in
        self.cv_num_out = cv_num_out

        self.input_representation = input_representation

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
        'word2vec':['word2vec','law2vec','patent2vec','doc2vec'],
        'preprocess__stopwords': [True,False],
        'preprocess__numbers': [True,False],
        'preprocess__lemma': [True,False],

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
        'word2vec':['word2vec','law2vec','patent2vec','doc2vec'],
        'preprocess__stopwords': [True,False],
        'preprocess__numbers': [True,False],
        'preprocess__lemma': [True,False],
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
        'word2vec':['word2vec','law2vec','patent2vec','doc2vec'],
        'preprocess__stopwords': [True,False],
        'preprocess__numbers': [True,False],
        'preprocess__lemma': [True,False],
        }
        self.xgboost = {
        'clf': [xgb.XGBClassifier(random_state=42, objective='binary:logistic', tree_method = 'hist', device='cuda')],
        'vect': [TfidfVectorizer(tokenizer=lambda x:x,preprocessor=lambda x:x),Word2VecTransform],
        'scaler':[StandardScaler()],
        f'vect__{self.num}ngram_range': [(1,1),(1,2),(1,3),(2,2),(2,3),(3,3),(1,4),(2,4),(3,4),(4,4)],
        'clf__n_estimators': [100, 200, 300],
        'clf__learning_rate':[0.01, 0.02, 0.05],
        'clf__gamma':[0.0, 0.1, 0.2],
        f'vect__{self.num}norm':[None, 'l2'],
        f'vect__{self.num}min_df':[2,5,10],
        f'vect__{self.num}use_idf':[True,False],
        'word2vec':['word2vec','law2vec','patent2vec','doc2vec'],
        'preprocess__stopwords': [True,False],
        'preprocess__numbers': [True,False],
        'preprocess__lemma': [True,False],
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

        #ONLY EXPERIMENT 1 SO FAR

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

            if self.input_representation == 'N-Gram':

                param[f'vect__{self.num}use_idf'] = [False]

                self.input_name = 'N-Grams'

                print('N-grams')

            elif self.input_representation == 'TF-IDF':
        
                param[f'vect__{self.num}norm'] = ['l2']
                param[f'vect__{self.num}use_idf'] = [True]

                self.input_name = 'TF-IDF'

                print('TF-IDF')


            else:
                pass
            
            print(param)

            ##############################################################################
            if self.opposition is True:
                preprocessor = ColumnTransformer(transformers=[("num", self.vects_n,'New Summary Facts'),("Cats",Binarizer(),['1','2'])])
                steps = Pipeline([('preprocess',PreProcessText(opposition=True)),('vect', preprocessor), ('clf', self.clf)])

            else:
                steps = Pipeline([('preprocess',PreProcessText()),('vect', self.vects_n), ('clf', self.clf)])

            clf = self.name_process(self.clf)

            self.training_ngram_core(param,steps,y_train)             

            print('END!!!!!!!!!!!!!!!!')

            if self.train_both is True:
                self.training_loop_we(X_train,y_train)
                results_temp = pd.DataFrame(self.results)
                results_temp.to_pickle(f'results_{clf}_{self.experiment}_{self.opposition}.pkl')

        return self.results
    
    def training_ngram_core(self,param,steps,y_train):

        # CHANGE FOR EXP 2 TIME SERIES SPLIT

        if self.experiment == '2':
            CV = RandomizedSearchCV(steps,param_distributions=param,n_iter=self.num_iter,cv=TimeSeriesSplit(n_splits=self.cv_num_out),scoring=self.scoring,refit='F1')

        else:
            CV = RandomizedSearchCV(steps,param_distributions=param,n_iter=self.num_iter,cv=RepeatedStratifiedKFold(n_splits=self.cv_num_out, n_repeats=self.repeat, random_state=42),scoring=self.scoring,refit='F1')

        time_taken = time.time()

        CV.fit(self.X_train, y_train)
      
        time_overall = time.time() - time_taken
        print("done in %0.3fs" % (time_overall))
        
        clf = self.name_process(self.clf)

        self.results.append({
                                'algo': clf,
                                'best score': CV.best_score_,
                                'best params': CV.best_params_,
                                'full results': CV.cv_results_,
                                'time taken': time_overall,
                                'embedding': self.input_name })

        results_super_temp = pd.DataFrame(self.results)

        results_super_temp.to_pickle(f'HYPERPARAMETERS__Experiment_{self.experiment}_Opposition_{self.opposition}_Algorithm_{self.models}_Input_{self.input_name}')

    def training_loop_we(self,X_train,y_train):

        #BROKEN FOR THIS

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

            print(self.clf)
        
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
            try:
                param.pop('preprocess__numbers')
            except:
                pass
            try:
                param.pop('preprocess__lemma')
            except:
                pass

            if self.input_representation == 'Word2Vec':

                self.input_name = 'word2vec'

            elif self.input_representation == 'Law2Vec':

                self.input_name = 'law2vec'

            elif self.input_representation == 'Patent2Vec':

                self.input_name = 'patent2vec'

            elif self.input_representation == 'Doc2Vec':

                self.input_name = 'doc2vec'

            else:
                raise TypeError
            
            print(self.input_name)

            self.embedding = self.input_name
            w2v = self.vects_we(self.embedding)

            if self.opposition is True:
                preprocessor = ColumnTransformer(transformers=[("num", Word2VecTransform(embedding=self.embedding,opposition=True),'New Summary Facts'),("Cats",Binarizer(),['1','2'])])
                steps = Pipeline([('preprocess',PreProcessText(opposition=True)),('vect', preprocessor),('scal', self.scal), ('clf',self.clf)])
            else:
                steps = Pipeline([('preprocess',PreProcessText()),('vect',Word2VecTransform(embedding=self.embedding,opposition=False)),('scal', self.scal), ('clf',self.clf)])
            
            self.training_w2v_core(param,steps,y_train)

            print('END!!!!!!!!!!!!!!!!')
      
    def training_w2v_core(self, param, steps, y_train):

        #BROKEN FOR THIS

        print(param)

        if self.experiment == '2':
            CV = RandomizedSearchCV(steps,param_distributions=param,n_iter=self.num_iter,cv=TimeSeriesSplit(n_splits=self.cv_num_in),scoring=self.scoring,refit='F1')

        else:
            CV = RandomizedSearchCV(steps,param_distributions=param,n_iter=self.num_iter,cv=self.cv_num_in,scoring=self.scoring,refit='F1')

        time_taken = time.time()
        CV.fit(self.X_train, y_train)

        time_overall = time.time() - time_taken
        print("done in %0.3fs" % (time_overall))

        results_super_temp = pd.DataFrame(self.results)
        results_super_temp.to_pickle(f'results__Experiment_{self.experiment}_Opposition_{self.opposition}_Algorithm_{self.models}_Input_{self.input_name}')

    def name_process(self,clf):

        clf = str(clf)
        clf = re.split('\(',clf)
        
        return clf[0]

