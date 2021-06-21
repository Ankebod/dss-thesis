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
import privacymodels
import onlineprivacy

import pickle
from pprint import pprint


class Experiments(object):
    def __init__(self, test = False):
        self.Nls = [11049]
        self.stemls = [False] # [True, False]
        self.featuresls = ['tfidf', 'glovemean'] #['tfidf', 'spacy', 'glovemean', 'glovestd']
        self.extendedls = [False] #[True, False]
        self.modells = ['nb', 'lr']
        self.test = test

    def offline(self):
        allresults = []
        for N in self.Nls:
            for stem in self.stemls:
                for feature in self.featuresls:
                    for extended in self.extendedls:
                        for model in self.modells:
                            settings = {'N': N, 'stem': stem, 'feature': feature, 'extended': extended, 'model' : model}
                            offline = offlinemodels.Offline(model_type = model)
                            results = offline.train(n = N, features = feature, extended = extended, stem = stem, test = self.test)
                            allresults.append((settings, results))
        return allresults

    def offlineprivacy(self):
        lrels = [4, float('inf'), 15]
        ssels = [4, float('inf'), 15]
        scalels = [True]
        featuresls = ["glovemean"]

        allresults = []
        count = 0
        for N in self.Nls:
            for stem in self.stemls:
                for feature in featuresls:
                    for extended in self.extendedls:
                        for model in self.modells:                       
                            for lre in lrels:
                                for sse in ssels:
                                    for scale in scalels:
                                        if lre != sse:
                                            continue
                                        settings = {'N': N, 'stem': stem, 'feature': feature, 'extended': extended, 'model' : model, 'lre': lre, 'sse': sse, 'scale' : scale}
                                        privacy = privacymodels.Privacy(model_type = model, scale = scale, sse = sse, lre = lre)
                                        results = privacy.train(n = N, features = feature, extended = extended, stem = stem)
                                        allresults.append((settings, results))
                                        count += 1
                                        if count % 10 == 0:
                                            print(f"{count, settings} done")
        return allresults

    def online(self, window_size = 1000):
        imballs = [True]
        driftls = [True, False]
        self.modells = ['nc','nb','lr']

        allresults = []
        for N in self.Nls:
            for stem in self.stemls:
                for feature in self.featuresls:
                    for imbal in imballs:
                        for drift in driftls:
                            for model in self.modells:
                                settings = {'N': N, 'stem': stem, 'drift' : drift, 'imbal' : imbal, 'feature': feature, 'model' : model}
                                online = onlinemodels.LinearOnline(model_type = model, features = feature, drift = drift, imbal = imbal, stem = stem)
                                results, hist, longhist = online.train(n = N,  window_size = window_size, nbatch = 1,  test = self.test)
                                allresults.append((settings, results))
        return allresults
    
    
    def onlineprivacy(self):
        imballs = [True, False]
        driftls = [True, False]
        model = 'nb'
        allresults = []
        els = [float('inf'), 4]
        for N in self.Nls:
            for stem in self.stemls:
                for feature in self.featuresls:
                    for imbal in imballs:
                        for drift in driftls:
                            for e in els:
                                settings = {'N': N, 'stem': stem, 'imbal' : imbal, 'feature': feature, 'model' : model, 'e': e}
                                onlinepriv = onlineprivacy.Online(priv = True, e = e, model_type = model, features = feature, stem = stem)
                                results, hist = onlinepriv.train(n = N, nbatch = 1, imbal = imbal, window_size = 500)
                                allresults.append((settings, results))
        return allresults


if __name__ == "__main__":
    print("Validation")
    tuner = Experiments(test = False)
    
    offlineres = tuner.offline()
    pprint(offlineres)
    with open('../output/offlinevalidation.pkl', 'wb') as output:
        pickle.dump(offlineres, output)
        
    privacyres = tuner.offlineprivacy()
    pprint(privacyres)
    with open('../output/privacyvalidation.pkl', 'wb') as output:
        pickle.dump(privacyres, output)
        
    onlineres = tuner.online()
    pprint(onlineres)
    with open('../output/online.pkl', 'wb') as output:
        pickle.dump(onlineres, output)