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

import spacy
import time
import matplotlib.pyplot as plt
import river as rv
import numpy as np
from river import linear_model
from river import optim
from river import feature_extraction
from river import naive_bayes
from river import metrics
from river import preprocessing
from river import dummy
from river import imblearn
from river import drift
from river import ensemble
from river import compat
from river import compose
from river.base import Transformer
from river.utils import numpy2dict
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_numeric, remove_stopwords, strip_short, stem_text, strip_multiple_whitespaces
from gensim.models import KeyedVectors
import gensim.downloader


class LinearOnline(object):
    '''
    TFIDF vectorization, training and testing online No Change, Logistic Regression and Naive bayes
    
    Parameters
    -------------
    model_type : string
        lr for Logistic Regression or nb for Naive Bayes
        
    emails : object
        Input pandas dataframe with emails
        
    imbal : bool 
        Account for imbalance if True
        
    drift : bool
        Drift detection if True
        
    N : int
        Number of items to train on
        
    window_size : int
        Number of observations to calculate the metric on. None if no sliding window evaluation.
        
    
    '''
    def __init__(self, model_type = 'lr', features = 'embeddings', imbal = False, drift = False, stem = False):
        
        self.embeddings = spacy.load('en_core_web_sm')
        
        # Choose a model type
        if model_type == 'nc':
            classifier = dummy.NoChangeClassifier()
        elif model_type == 'lr':
            classifier = linear_model.LogisticRegression()            
        elif model_type == 'nb':
            classifier = naive_bayes.GaussianNB()
               
        
        # Account for imbalanced classes
        if imbal == True:
            classifier = imblearn.RandomUnderSampler(
                    classifier=classifier,
                    desired_dist={0: .5, 1: .5},
                    seed=42 )
        
        if features != 'tfidf':
            model = (
                rv.preprocessing.StandardScaler() | 
                classifier )   
                
        elif features == 'tfidf':
            method = feature_extraction.TFIDF()
            model = (
                method |
                rv.preprocessing.StandardScaler() | 
                classifier ) 
        
        # Account for concept drift
        if drift == True:
            self.model = ensemble.ADWINBaggingClassifier(
                            model = model,
                            n_models=3,
                            seed=42 )
        
        elif drift == False:
            self.model = model
            
        self.features = features
        self.load = data.DataLoader()
        self.glove_vectors = gensim.downloader.load('glove-wiki-gigaword-100')
        linear = offlinemodels.Linear()
        self.filters = self.load.filters(stem)
    
    def loadonline(self, i):
        email = self.load.fetch(index = i)
        x, y = email['body'], email['lab_bin']

        return x, y
    
    def spacyonline(self, x):
        return self.embeddings(x).vector
    
    def gloveonline(self, x, std):
        
        x = preprocess_string(x, filters = self.filters)

        vectors = []

        for word in x:
            if word in self.glove_vectors.key_to_index.keys():
                vector = self.glove_vectors[word]
                vectors.append(vector)

        if len(vectors) > 0:
            vectors = np.dstack(vectors)
            
            vectmean = np.mean(vectors, axis = 2).flatten()
            
            if std == False:
                vectors = vectmean
            elif std == True:
                vectstd = np.std(vectors, axis = 2).flatten()
                vectors = np.hstack([vectmean, vectstd])
        
        else:
            vectors = None
         
        return vectors
        
    def train(self, n, window_size = 1000, nbatch = 5, test = False):
        
        auc, acc, f1, prec, rec = metrics.ROCAUC(), metrics.Accuracy(), metrics.F1(), metrics.Precision(), metrics.Recall()
        if test == False:
            end = int(n*0.8)
            
        if test == True:
            teststart = int(n*0.8)
            end = n
            nbatch = 1
        
        aucs, accs, f1s, precs, recs = [], [], [], [], []
        
        batchsize = int(end / nbatch)
        stop = 0
        
        cvscores = []
        for batch in range(nbatch):
            stop += batchsize
            
            for i in range(batchsize):
                x, y = self.loadonline(i)

                if self.features == 'glovemean':
                    x = self.gloveonline(x, std = False)
                    if x is None:
                        continue 
                    x = numpy2dict(x)

                if self.features == 'glovestd':
                    x = self.gloveonline(x, std = True)
                    if x is None:
                        continue 
                    x = numpy2dict(x)

                if self.features == 'spacy':
                    x = self.spacyonline(x)
                    x = numpy2dict(x)

                y_pred = self.model.predict_one(x)

                if window_size != None:
                    if i % (window_size/2) == 0 and i % (window_size) != 0:
                        auc2, acc2, f12, prec2, rec2 = metrics.ROCAUC(), metrics.Accuracy(), metrics.F1(), metrics.Precision(), metrics.Recall()

                    if i % window_size == 0 and i != 0:
                        auc, acc, f1, prec, rec = auc2, acc2, f12, prec2, rec2

                    if i > (window_size/2) and i % window_size != 0: 
                        for metric in [auc2, acc2, f12, prec2, rec2]:
                            if y_pred is not None:
                                metric.update(y_pred=y_pred, y_true=y)

                if y_pred is not None and i > 200:
                    for metric in [auc, acc, f1, prec, rec]:
                        metric.update(y_pred=y_pred, y_true=y)

                aucs.append(auc.get()), accs.append(acc.get()), f1s.append(f1.get()), precs.append(prec.get()), recs.append(rec.get())

                self.model.learn_one(x, y)
                
            
            if test == False:
                result = [np.mean(aucs[200:]), np.mean(accs[200:]), np.mean(f1s[200:]),np.mean(precs[200:]),np.mean(recs[200:])]
                cvscores.append(result)

        if test == False:
            return np.mean(cvscores), cvscores
        
        if test == True:
            return [np.mean(aucs[teststart:]), np.mean(accs[teststart:]),np.mean(f1s[teststart:]),np.mean(precs[teststart:]), np.mean(recs[teststart:])],  [aucs[teststart:], accs[teststart:], f1s[teststart:]], [aucs, accs, f1s]


if __name__ == "__main__":
    N = 11050
    stem = False
    drift = False
    imbal = True
    feature = 'tfidf'
    
    print("\n\nOnline:")
    
    print('\nNo Change: \n-------------------')
    pipe = LinearOnline(model_type = 'nc', stem = stem, features = feature, imbal = imbal, drift = drift)    
    results, testhist, hist = pipe.train(n = N, test = True)
    print(results)
    
    print('\nLogistic Regression: \n-------------------')
    pipe = LinearOnline(model_type = 'lr', stem = stem, features = feature, imbal = imbal, drift = drift)
    lrresults, lrtesthist, lrhist = pipe.train(n = N, test = True)
    print(lrresults)
    
    print('\nNaive Bayes: \n-------------------')
    pipe = LinearOnline(model_type = 'nb', stem = stem, features = feature, imbal = imbal, drift = drift)
    nbresults, nbtesthist, nbhist = pipe.train(n = N, test = True)
    print(nbresults)
    
    feature = 'glovemean'
    
    print('\nNo Change: \n-------------------')
    pipe = LinearOnline(model_type = 'nc', stem = stem, features = feature, imbal = imbal, drift = drift)    
    results, testhist, glhist = pipe.train(n = N, test = True)
    print(results)
    
    print('\nLogistic Regression: \n-------------------')
    pipe = LinearOnline(model_type = 'lr', stem = stem, features = feature, imbal = imbal, drift = drift)
    lrresults, lrtesthist, gllrhist = pipe.train(n = N, test = True)
    print(lrresults)
    
    print('\nNaive Bayes: \n-------------------')
    pipe = LinearOnline(model_type = 'nb', stem = stem, features = feature, imbal = imbal, drift = drift)
    nbresults, nbtesthist, glnbhist = pipe.train(n = N, test = True)
    print(nbresults)
    
    teststart = int(N*0.8)
    
    plt.figure(1, figsize=(7, 6), dpi=80)
    plt.subplot(211)
    plt.plot(hist[0], 'lightgreen', label="Tfidf NC")
    plt.plot(lrhist[0], 'tomato', label="Tfidf LR")
    plt.plot(nbhist[0], 'royalblue', label="Tfidf NB")
    plt.axvline(x=teststart, linestyle = '--', color='black', label='Start Testset')

    plt.ylabel("AUC")
    plt.ylim(0, 1)
    plt.xlim(202, len(hist[0]))
    plt.legend(loc=0)
    
    plt.subplot(212)
    plt.plot(glhist[0], 'lightgreen', label="Glove NC")
    plt.plot(gllrhist[0], 'tomato', label="Glove LR")
    plt.plot(glnbhist[0], 'royalblue', label="Glove NB")
    plt.axvline(x=teststart, linestyle = '--', color='black', label='Start Testset')

    plt.xlabel("Email")
    plt.ylabel("AUC")
    plt.ylim(0, 1)
    plt.xlim(202, len(hist[0]))
    plt.legend(loc=3)
    
    plt.savefig('../output/online1000.pdf', format = 'pdf')




