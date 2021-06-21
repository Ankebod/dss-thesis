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

import pickle
from tabulate import tabulate
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

import offlinemodels
import onlinemodels
import privacymodels
import experiments
import data


def offlineselect(results, testresults):
    output = []
    
    for res in [results, testresults]:    
    
        offlinesort = sorted(res, key=lambda x: x[1][0], reverse = True)

        best_nb_tfidf, best_lr_tfidf, best_nb_emb, best_lr_emb = None, None, None, None
        for i, item in enumerate(offlinesort):
            if 'e' not in item[0].keys() and item[0]['stem'] == False and item[0]['extended'] == False:
                if item[0]['feature'] == 'tfidf' and item[0]['model'] == 'nb' and best_nb_tfidf == None:
                    best_nb_tfidf = item
                if item[0]['feature'] == 'tfidf' and item[0]['model'] == 'lr' and best_lr_tfidf == None:
                    best_lr_tfidf = item
                if item[0]['feature'] == 'glovemean' and item[0]['model'] == 'nb' and best_nb_emb == None:
                    best_nb_emb = item
                if item[0]['feature'] == 'glovemean' and item[0]['model'] == 'lr' and best_lr_emb == None:
                    best_lr_emb = item
                    
        for model in [best_nb_tfidf, best_lr_tfidf, best_nb_emb, best_lr_emb]:
            output.append(model)
    
    return output 


def privselect(results, testresults):
    output = []
    
    for res in [results, testresults]:       
        offlinesort = sorted(res, key=lambda x: x[1][0], reverse = True)

        best_lr_emb_15, best_lr_emb_4, best_nb_emb_15, best_nb_emb_4 = None, None, None, None
        for i, item in enumerate(offlinesort):
            if item[0]['stem'] == False and item[0]['extended'] == False:
                if item[0]['feature'] == 'glovemean' and item[0]['model'] == 'lr':
                        
                    if  item[0]['lre'] == 15 and item[0]['sse'] == 15 and best_lr_emb_15 == None:
                        best_lr_emb_15 = item

                    elif  item[0]['lre'] == 4 and item[0]['sse'] == 4 and best_lr_emb_4 == None:
                        best_lr_emb_4 = item

                if item[0]['feature'] == 'glovemean' and item[0]['model'] == 'nb':
                        
                    if item[0]['lre'] == 15 and item[0]['sse'] == 15 and best_nb_emb_15 == None:
                        best_nb_emb_15 = item

                    elif item[0]['lre'] == 4 and item[0]['sse'] == 4 and best_nb_emb_4 == None:
                        best_nb_emb_4 = item
        for model in [best_nb_emb_15, best_nb_emb_4, best_lr_emb_15, best_lr_emb_4]:
            output.append(model)

    return output


def baseonlselect(results, testresults):
    output = []
    for res in [results, testresults]:    
        onlinesort = sorted(res, key=lambda x: x[1][0], reverse = True)
        best_nc_tfidf, best_nc_emb = None, None

        for i, item in enumerate(onlinesort):
            if 'e' not in item[0].keys() and item[0]['stem'] == False:
                if item[0]['feature'] == 'tfidf' and item[0]['model'] == 'nc' and item[0]['drift'] == False and best_nc_tfidf == None:
                    best_nc_tfidf = item
                if item[0]['feature'] == 'glovemean' and item[0]['model'] == 'nc' and item[0]['drift'] == False and best_nc_emb == None:
                    best_nc_emb = item
        for model in [best_nc_tfidf, best_nc_emb]:
            output.append(model)
                  
    return output


def onlineselect(results, testresults, drift = False):
    output = []
    for res in [results, testresults]:    
        onlinesort = sorted(res, key=lambda x: x[1][0], reverse = True)
        best_nb_tfidf, best_lr_tfidf, best_nb_emb, best_lr_emb, best_nc_tfidf = None, None, None, None, None

        for i, item in enumerate(onlinesort):
            if 'e' not in item[0].keys() and item[0]['stem'] == False:
                if item[0]['feature'] == 'tfidf' and item[0]['model'] == 'nb' and item[0]['drift'] == drift and best_nb_tfidf == None:
                    best_nb_tfidf = item
                if item[0]['feature'] == 'tfidf' and item[0]['model'] == 'lr' and item[0]['drift'] == drift and best_lr_tfidf == None:
                    best_lr_tfidf = item
                if item[0]['feature'] == 'glovemean' and item[0]['model'] == 'nb' and item[0]['drift'] == drift and  best_nb_emb == None:
                    best_nb_emb = item
                if item[0]['feature'] == 'glovemean' and item[0]['model'] == 'lr' and item[0]['drift'] == drift and best_lr_emb == None:
                    best_lr_emb = item
        for model in [best_nb_tfidf, best_lr_tfidf, best_nb_emb, best_lr_emb]:
            output.append(model)
                  
    return output


def tablebuild(results, headers):
    headers = headers
    rows = []
    for m, metric in enumerate(["bftab AUC", "bftab Acc.", "bftab F1", "bftab Prec.", "bftab Rec."]):
        row = [metric]
        for i in range(len(results)):
            row.append('{:.3f}'.format(results[i][1][m]))
        
        border = int((len(row)-1)/2)+1

        maxval, maxtest = max(row[1:border]), max(row[border:]) 
        maxvali, maxtesti = row.index(maxval), row.index(maxtest)

        row[maxvali] = "bftab {0}".format(maxval)
        row[maxtesti] = "bftab {0}".format(maxtest)

        rows.append(row)
        
    tab = tabulate(rows, headers, tablefmt = 'latex_raw')
    
    return tab


# +
from scipy.stats import ttest_rel
    
def test(cvoff, cvpriv, cvon):
    onres = []
    privres = []

    headers = ["Stat Onl", "P Onl", "Stat Priv", "P Priv"]
    
    for m, metric in enumerate(["AUC", "Acc", "F1"]):
        on = ttest_rel(cvon[m], cvoff[m])
        priv = ttest_rel(cvpriv[m], cvoff[m])
        onres.append([metric, np.round(on[0], decimals = 3), np.round(on[1], decimals = 3)])
        privres.append(np.round([priv[0], priv[1]], decimals = 3))
    
    a = np.vstack(onres)
    b = np.vstack(privres)
    c = np.hstack([a, b])
    
    tab = tabulate(c, headers, tablefmt = 'latex_raw')

    return tab


# +
if __name__ == "__main__":
    # Process and write data to disk
    PATH = '../data/'
    EMAILNAME = 'emails.json'
    LABNAME = 'enron_emails.csv'
    ANNOTATORS = 8

#     load = data.DataLoader(annotators=ANNOTATORS)
#     emails = load.run(path=PATH, emailname=EMAILNAME, labname=LABNAME)
    
    # Validation and testing
    print("Validation")
    tuner = experiments.Experiments(test = False)
    
    offlineres = tuner.offline()
    pprint(offlineres)
    with open('../output/offlinevalidation.pkl', 'wb') as output:
        pickle.dump(offlineres, output)
        
    privacyres = tuner.offlineprivacy()
    pprint(privacyres)
    with open('../output/privacyvalidation.pkl', 'wb') as output:
        pickle.dump(privacyres, output)
        
    onlineres = tuner.online(window_size = 1000)
    pprint(onlineres)
    with open('../output/onlinevalidation.pkl', 'wb') as output:
        pickle.dump(onlineres, output)
    
    print("Testing")
    tester = experiments.Experiments(test = True)
    
    offlinetest = tester.offline()
    pprint(offlinetest)
    with open('../output/offlinetest.pkl', 'wb') as output:
        pickle.dump(offlinetest, output)
        
    privacytest = tester.offlineprivacy()
    pprint(privacytest)
    with open('../output/privacytest.pkl', 'wb') as output:
        pickle.dump(privacytest, output)
        
    onlinetest = tester.online(window_size = 1000)
    pprint(onlineres)
    with open('../output/onlinetest.pkl', 'wb') as output:
        pickle.dump(onlinetest, output)
    
    # Make tables
    offline = offlineselect(offlineres, offlinetest)
    privacy = privselect(privacyres, privacytest)   
    online = onlineselect(onlineres, onlinetest, drift = False)
    onlinedrift = onlineselect(onlineres, onlinetest, drift = True)
    baseonline = baseonlselect(onlineres, onlinetest)

    offlinetable = tablebuild(offline, ["NB Tf", "LR Tf","NB Gl","LR Gl", "NB Tf", "LR Tf","NB Gl","LR Gl"])
    baseonltable = tablebuild(baseonline, ["NC Tf","NC Emb","NC Tf","NC Emb"])
    onlinetable = tablebuild(online, ["NB Tf","LR Tf","NB Gl","LR Gl", "NB Tf","LR Tf","NB Gl","LR Gl"]) 
    onlinedrifttable = tablebuild(onlinedrift, ["NC Tf","NB Tf","LR Tf","NB Gl","LR Gl", "NC Tf","NB Tf","LR Tf","NB Gl","LR Gl"])
    privtable = tablebuild(privacy, ["NB Gl10","NB Gl8","LR Gl10","LR Gl8", "NB Gl10","NB Gl8","LR Gl10","LR Gl8"])
    
    exports = zip(["offlinetab", "privtab", "onlinetab", "baseonltable", "onlinedrifttable"], [offlinetable, privtable, onlinetable, baseonltable, onlinedrifttable])
    for name, table in exports:
        print(name, table)
        file = open("../output/" + name + ".txt","w")
        file.write(table)
        file.close()
    
    # Make tables
    offline = offlineselect(offlineres, offlinetest)
    privacy = privselect(privacyres, privacytest)   
    online = onlineselect(onlineres, onlinetest, drift = False)
    onlinedrift = onlineselect(onlineres, onlinetest, drift = True)
    baseonline = baseonlselect(onlineres, onlinetest)

    offlinetable = tablebuild(offline, ["NB Tf", "LR Tf","NB Gl","LR Gl", "NB Tf", "LR Tf","NB Gl","LR Gl"])
    baseonltable = tablebuild(baseonline, ["NC Tf","NC Emb","NC Tf","NC Emb"])
    onlinetable = tablebuild(online, ["NB Tf","LR Tf","NB Gl","LR Gl", "NB Tf","LR Tf","NB Gl","LR Gl"]) 
    onlinedrifttable = tablebuild(onlinedrift, ["NC Tf","NB Tf","LR Tf","NB Gl","LR Gl", "NC Tf","NB Tf","LR Tf","NB Gl","LR Gl"])
    privtable = tablebuild(privacy, ["NB Gl10","NB Gl8","LR Gl10","LR Gl8", "NB Gl10","NB Gl8","LR Gl10","LR Gl8"])
    
    exports = zip(["offlinetab", "privtab", "onlinetab", "baseonltable", "onlinedrifttable"], [offlinetable, privtable, onlinetable, baseonltable, onlinedrifttable])
    for name, table in exports:
        print(name, table)
        file = open("../output/" + name + ".txt","w")
        file.write(table)
        file.close()
# -
if __name__ == "__main__":    
    N = 11049
    stem = False
    drift = False
    imbal = True
    extended = False
    feature = "glovemean"
    epsilon = 30
    k = 5
    
    print("\n\nLogistic Regression:")
    
    pipe = offlinemodels.Offline(model_type = 'lr')
    results, offcv = pipe.train(n = N, features = feature, k = k, extended = extended, stem = stem, test = False)
    
    pipe = privacymodels.Privacy(model_type = 'lr', lre = epsilon/2, sse = epsilon/2)
    results, privcv = pipe.train(n = N, features = 'glovemean', k = k, extended = extended, stem = stem)
    
    pipe = onlinemodels.LinearOnline(model_type = 'lr', stem = stem, features = feature, imbal = imbal, drift = drift)    
    results, oncv = pipe.train(n = N, test = False, nbatch = k)
    
    print("Test")
    print(test(offcv, privcv, oncv))
    
    print("\n\nNaive Bayes:")
    
    pipe = offlinemodels.Offline(model_type = 'nb')
    results, offcv = pipe.train(n = N, features = feature, k = k, extended = extended, stem = stem, test = False)
    
    pipe = privacymodels.Privacy(model_type = 'nb', lre = epsilon/2, sse = epsilon/2)
    results, privcv = pipe.train(n = N, features = 'glovemean', k = k, extended = extended, stem = stem)

    pipe = onlinemodels.LinearOnline(model_type = 'nb', stem = stem, features = feature, imbal = imbal, drift = drift)    
    results, oncv = pipe.train(n = N, test = False, nbatch = k)
    
    print("Test")
    print(test(offcv, privcv, oncv))


