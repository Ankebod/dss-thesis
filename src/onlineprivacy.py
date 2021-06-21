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
import offlinemodels
import onlinemodels

# +
import pandas
import spacy
import pickle
import time
import numpy as np
np.seterr(divide = 'ignore') 
from collections import Counter
import sklearn as sk
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_numeric, remove_stopwords, strip_short, stem_text, strip_multiple_whitespaces
from sklearn.feature_extraction.text import TfidfTransformer
import diffprivlib as dp
from diffprivlib.models import GaussianNB, LogisticRegression, PCA, StandardScaler
from sklearn.linear_model import SGDClassifier

import river as rv
from river import feature_extraction
from river import metrics
from river.utils import dict2numpy
from river.base import Transformer


# -

class Online(object):
    '''
    Child of Linear class, implements the offline version
    
    Parameters:
    ----------
    priv : bool
        Private or not
        
    e : float
        Privacy budget
    
    norm : float
        Privacy budget
        
    features : str
        Spacy, glovestd, glovemean or tfidf
        
    imbal : bool
        Use balanced data 
    
    nbbounds : tuple
        Bounds of naive bayes
    
    ssbounds : tuple
        Bounds of standard scaler
    
    '''
    def __init__(self, model_type, priv = True, e = 4, norm = 50, features = 'glovemean', stem = False, nbbounds = (-16,16), ssbounds = (-3,3)): 
        
        if priv == False and model_type == 'nb':
            self.scaler = sk.preprocessing.StandardScaler()
            self.classifier = sk.naive_bayes.BernoulliNB()
            
        elif priv == True and model_type == 'nb':
            self.scaler = dp.models.StandardScaler(epsilon = e, bounds = ssbounds)
            self.classifier = GaussianNB(epsilon = e, bounds = nbbounds)
            
        if priv == False and model_type == 'lr':
            self.scaler = sk.preprocessing.StandardScaler()
            self.classifier = SGDClassifier(loss = 'log', max_iter=10000, tol=1e-3)
            
        if priv == True and model_type == 'lr':
            raise Exception("Sorry, not yet possible")
            
        self.features = features
        self.online = onlinemodels.LinearOnline()
        linear = offlinemodels.Linear()
        self.filters = linear.filters(stem)
        self.priv = priv
        
    def loadonline(self, i):

            load = data.DataLoader()
            email = load.fetch(index = i) 
            
            x, y = email['body'], int(email['lab_bin'])
            
            if self.features == 'tfidf':
                x = ' '.join(preprocess_string(x, filters = self.filters))
                vect = feature_extraction.TFIDF()
                vect = vect.learn_one(x)
                x = vect.transform_one(x) 
                x = dict2numpy(x)
                
                if x.shape[0] == 0:
                    x = None
            
                else:
                    x  = x.reshape(-1, x.shape[0])
                    padding = int(100 - x.shape[0])
                    x = np.pad(x, ((0,0), (0, padding)))
                    x = x[:,:100]
            
            elif self.features == 'spacy':    
                x = self.online.spacyonline(x)
                if x is not None:
                    x = x.reshape(1, x.shape[0])
        
            elif self.features == 'glovemean':    
                x = self.online.gloveonline(x, std = False)
                if x is not None:
                    x = x.reshape(1, x.shape[0])
            
            elif self.features == 'glovestd':    
                x = self.online.gloveonline(x, std = True)
                if x is not None:
                    x = x.reshape(1, x.shape[0])
            
            if x is not None:
                x = self.scaler.fit_transform(x)
            
            return x , y
        
        
    def train(self, n, nbatch = 5, imbal = False, window_size = 500):
        auc, acc, f1 = metrics.ROCAUC(), metrics.Accuracy(), metrics.F1()  
        auchist, acchist, f1hist = [], [], []   
        
        size = int(n/nbatch)
        start = 0
        for batch in range(nbatch):
            
            end = start + size
            
            x, y = self.loadonline(1)
            
            self.classifier.partial_fit(x, np.array(y).reshape(1), classes =[0, 1]) # initial fit

            y_prob = self.classifier.predict_proba(x)[0]
            
            count0, count1 = 0, 0
            aucs, accs, f1s = [], [], [] # these are for the history within batches
           
            for i in range(1, n):
                x, y = self.loadonline(i)
                if x is None:
                    continue

                else:
                    x = x.reshape(1, -1)

                    if imbal == True:
                        if y == 0:
                            count0 += 1
                            ratio = count0/(count0+count1)
                            if i > 20 and ratio > 0.75 and y_prob[0] > 0.5: # continue if overrepresented group with high confidence
                                count0 -= 1
                                continue

                        if y == 1:
                            count1 += 1
                            ratio = count1/(count0+count1)

                            if i > 20 and ratio > 0.75 and y_prob[1] > 0.5:
                                count1 -= 1
                                continue
                        
                    if window_size != None:
                        if i % (window_size/2+200) == 0 and i % window_size != 0:
                            auc2, acc2, f12 = metrics.ROCAUC(), metrics.Accuracy(), metrics.F1()

                        if i % window_size == 0 and i != 0:
                            auc, acc, f1 = auc2, acc2, f12

                        if i > (window_size/2+200) and i % window_size != 0: 
                            for metric in [auc2, acc2, f12]:
                                metric.update(y_pred=y_pred, y_true=y)

                    y_pred = self.classifier.predict(x).item()

                    if y_pred is not None and i > 200:
                        for metric in [auc, acc, f1]:
                            metric.update(y_pred=y_pred, y_true=y)

                    self.classifier.partial_fit(x, np.array(y).reshape(1))
                    
                    aucs.append(auc.get()), accs.append(acc.get()), f1s.append(f1.get())
                    
            auchist.append(np.mean(aucs[200:])), acchist.append(np.mean(accs[200:])), f1hist.append(np.mean(f1s[200:]))

            start = end

        return [np.mean(auchist), np.mean(acchist), np.mean(f1hist)], [aucs, accs, f1s]

# +
if __name__ == "__main__":
    N = 1000
    print("\n\nOnline:")
    
#     print('\nNon-Private Logistic Regression (SGD): \n-------------------')
#     pipe = Online(priv = False, model_type = 'lr', features = 'glovestd')
#     results, hist = pipe.train(n = N, imbal = False, nbatch = 1)
#     print(results)
    
#     print('\nPrivate Naive Bayes: \n-------------------')
#     pipe = Online(priv = True, e = float('inf'), model_type = 'nb', features = 'glovemean')
#     results, hist = pipe.train(n = N, imbal = True, nbatch = 1)
#     print(results)
    
    print('\nNon-Private Naive Bayes: \n-------------------')
    pipe = Online(priv = False, e = float('inf'), model_type = 'nb', features = 'glovemean')
    results, hist = pipe.train(n = N, imbal = True, nbatch = 1)
    print(results)
# -




