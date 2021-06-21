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

import pandas
import pickle
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_numeric, remove_stopwords, strip_short, stem_text, strip_multiple_whitespaces


from diffprivlib.models import GaussianNB, LogisticRegression, PCA, StandardScaler
from diffprivlib import BudgetAccountant


class Privacy(offlinemodels.Linear):
    '''
    Child of Linear class, implements the private version
    
    Parameters
    ------------
    
    model_type : string
        lr for Logistic Regression or nb for Naive Bayes
        
    e : float
        Privacy budget
        
    norm : float
        Data norm (range of privacy) in data
        
    nbbounds : tuple
        Bounds of naive bayes
    
    ssbounds : tuple
        Bounds of standard scaler
        
    '''
    def __init__(self, model_type = 'lr', lre = 8, sse = 14, norm = 50, nbbounds = (-16, 16), ssbounds = (-3, 3), scale = False):    
        
        self.account = BudgetAccountant()
        self.account.set_default()
        
        # Choose a model type
        if model_type == 'lr':
            classifier = LogisticRegression(epsilon = lre, C = 1, data_norm = norm, max_iter = 10000, tol=1e-3)
        elif model_type == 'nb':
            classifier = GaussianNB(epsilon = lre, bounds = nbbounds)
        
        if scale == True:
            self.model = pipeline.Pipeline([  
                        ('scale', StandardScaler(epsilon = sse, bounds = ssbounds)),
                        ('class', classifier) 
                        ])
        elif scale == False:
            self.model = pipeline.Pipeline([  
                        ('class', classifier) 
                        ])


if __name__ == "__main__":
    N = 11049
    epsilons = np.logspace(-1, 4, 100)
    lrres, nbres = [], []
    
    for epsilon in epsilons:
        pipe = Privacy(model_type = 'lr', lre = epsilon/2, sse = epsilon/2)
        lrresults = pipe.train(n = N, features = 'glovemean', k = 5, extended = False, stem = False)
        print(lrresults)
        lrres.append(lrresults)
        
        pipe = Privacy(model_type = 'nb', lre = epsilon/2, sse = epsilon/2)
        nbresults = pipe.train(n = N, features = 'glovemean', k = 5, extended = False, stem = False)
        print(nbresults)
        nbres.append(nbresults)
        
    lrarray = np.array(lrres)
    nbarray = np.array(nbres)
    
    lrbaseline = 0.751
    nbbaseline = 0.818

    plt.figure(figsize=(7, 4))
    plt.semilogx(epsilons, lrarray[:,0], color = 'tomato',label="Private LR", zorder=3)
    plt.semilogx(epsilons, lrbaseline * np.ones_like(epsilons), dashes=[2,2], color = 'tomato', label="Non-private LR", zorder=3)

    plt.semilogx(epsilons, nbarray[:,0],  color = 'royalblue', label="Private NB", zorder=10)
    plt.semilogx(epsilons, nbbaseline * np.ones_like(epsilons), dashes=[2,2], color = 'royalblue', label="Non-private NB", zorder=10)
    plt.xlabel("epsilon (log scale)")
    plt.ylabel("AUC (linear scale)")
    plt.xlim(epsilons[0], epsilons[-1])
    plt.legend(loc=0)
    plt.savefig('../output/epsilonplotvalidate.eps', format = 'eps')


