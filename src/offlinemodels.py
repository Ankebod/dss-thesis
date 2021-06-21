# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import data

# +
import pandas 
import spacy
import pickle
import numpy as np
from collections import Counter
import sklearn as sk
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_numeric, remove_stopwords, strip_short, stem_text, strip_multiple_whitespaces
from sklearn.preprocessing import LabelBinarizer

from gensim.models import KeyedVectors
import gensim.downloader

from river.base import Transformer
from river.utils import dict2numpy


# +
class Linear:
    '''
    Loading and training functions for linear models.
    
    Parameters
    ------------
        
    k : int
        Number of splits for cross validation
        
    n : int
        Number of examples to use
    
    features : str
        tifdf, spacy, glovemean (only mean embedding) or glovestd (mean and standard deviation embedding)
        
    stem : bool
        Word stemming or not
        
    extended : bool
        Add nwords and n recepients or not
        
    maxfeatures : int
        Max features for tfidf 
        
    mindf : int or float
        Minimum document frequency tfidf (ratio or count)
        
    maxdf : int or float
        Max document frequency tfifd (ratio or count)
        
        
    '''       
        
    def gloveoffline(self, X, Y, nwords, nrec, stem, std):
        load = data.DataLoader()
        glove_vectors = gensim.downloader.load('glove-wiki-gigaword-100')
        

        Xnew = []
        empty = []
        for i, x in enumerate(X):
            x = preprocess_string(x, filters = load.filters(stem))
            
            vectors = []
 
            for word in x:
                if word in glove_vectors.key_to_index.keys():
                    vector = glove_vectors[word]
                    vectors.append(vector)

            if len(vectors) > 0:
                vectors = np.dstack(vectors)   
                
                vectmean = np.mean(vectors, axis = 2).flatten()
                
                if std == False:
                    vectors = vectmean
                elif std == True:
                    vectstd = np.std(vectors, axis = 2).flatten()
                    vectors = np.hstack([vectmean, vectstd])

                Xnew.append(vectors) 
                
            else:
                empty.append(i)
        
        X = np.vstack(Xnew)
        Y = np.delete(Y.to_numpy(), empty, axis = 0)
        nwords = np.delete(nwords, empty, axis = 0)
        nrec = np.delete(nrec, empty, axis = 0)
        
        return X, Y, nwords, nrec
    
            
            
    def loadoffline(self, n, features, max_features, min_df, max_df, extended, stem):
        
        load = data.DataLoader()
        emails = load.fetch(online = False)

            
        emails['body'].replace('', np.nan, inplace=True)
        emails.dropna(subset=['body'], inplace=True)
        
        
        X = emails.loc[:n,'body']
        
        nwords = X.apply(len).to_numpy().reshape(len(X), 1)   
        recepients = emails.loc[:n,'recipients']
        nrec = X.apply(len).to_numpy().reshape(len(X), 1)
        
        Y = emails.loc[:n, 'lab_bin']
        
        if features == 'tfidf':
            X = X.apply(load.prep, stem = stem)
            vect = TfidfVectorizer(max_features = max_features , min_df = min_df, max_df = max_df)
            X = vect.fit_transform(X).toarray()
        
        elif features == 'spacy':
            embeddings = spacy.load('en_core_web_sm')
            X = X.apply(embeddings)
            X = X.apply(lambda x: x.vector)
            X = np.vstack(X)  
        
        elif features == 'glovemean':
            X, Y, nwords, nrec = self.gloveoffline(X, Y, nwords, nrec, stem, std = False)
            
        elif features == 'glovestd':
            X, Y, nwords, nrec = self.gloveoffline(X, Y, nwords, nrec, stem, std = True)
        
        if extended == True:
            X = np.hstack([X, nwords, nrec])

        return X, Y
    
    def plot(self, X):
        tsne = TSNE(n_components=2, random_state=0)
    
    
    def train(self, n, features = 'spacy', k = 5, max_features = 5000, min_df = 5, max_df = 0.7, extended = False, stem = False, test = False):
        
        X, Y = self.loadoffline(n, features, max_features = max_features, min_df = min_df, max_df = max_df, extended = extended, stem = stem)
        
        X_train, X_test = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
        y_train, y_test = Y[:int(len(Y)*0.8)], Y[int(len(Y)*0.8):]
        
        metrics = [sk.metrics.roc_auc_score, sk.metrics.accuracy_score, sk.metrics.f1_score, sk.metrics.precision_score, sk.metrics.recall_score]
        results = []
        
        cvscores = []
        if test == False:
            cv = model_selection.StratifiedKFold(n_splits= k, shuffle=True, random_state=42) 

            for metric in metrics:
                result = model_selection.cross_val_score(self.model, X_train, y_train, scoring=sk.metrics.make_scorer(metric), cv=cv)
                results.append(np.mean(result))
                cvscores.append(result)

        if test == True:
            self.model.fit(X_train, y_train)
        
            y_pred = self.model.predict(X_test)

            for metric in metrics:
                result = metric(y_test, y_pred)
                results.append(result)
        
        return results, cvscores
    
        
            
            
# -

class Offline(Linear):
    '''
    Child of Linear class, implements the offline version
    
    Parameters:
    ----------
    model_type : string
        'lr' for Logistic Regression or 'nb' for Naive Bayes
    
    '''
    def __init__(self, model_type = 'lr'): 
        
        if model_type == 'lr':
            classifier = sk.linear_model.LogisticRegression(max_iter=10000, tol=1e-3)
        elif model_type == 'nb':
            classifier = sk.naive_bayes.GaussianNB()
            
        
        self.model = make_pipeline(sk.preprocessing.StandardScaler(),
                                   classifier)


if __name__ == "__main__":
    N = 11049
    
    print("Offline:")
    
    print('\nLogistic Regression: \n-------------------')
    pipe = Offline(model_type = 'lr')
    results, cvscores = pipe.train(n = N, features = 'glovemean', extended = False, stem = False, test = False)
    print(f'AUC, ACC, F1: {results}')
    
    print('\nNaive Bayes: \n-------------------')
    pipe = Offline(model_type = 'nb')
    results, cvscores = pipe.train(n = N, features = 'glovemean', extended = False, stem = False, test = False)
    print(f'AUC, ACC, F1: {results}')
    print(cvscores)




