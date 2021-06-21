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

import json
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import datetime
import argparse
import pickle
import scipy, pylab
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_numeric, remove_stopwords, strip_short, stem_text, strip_multiple_whitespaces
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from tabulate import tabulate
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


class DataLoader(object):
    '''
    Loads the data, merges emails with the labels and orders emails

    Parameters
    ----------
    path : str
        Path to data

    emailname : str
        Name of email file, including .json

    labname1 : str
        Name of label file, including .csv

    annotators : int
        Number of annotators
        
    online : bool
        True to fetch one email at the time (online learning), false to fetch the full batch for offline learning.
        
    emailslab : object
        Full preprocessed email dataset for reporting, returned by the fetch offline function
        
    '''

    def __init__(self, annotators=8):
        self.annotators = annotators
        
    def filters(self, stem):
        '''
        Define preprocessing filters
        '''
        if stem == False:
            filters = [
                    lambda x: x.lower(), strip_tags, strip_punctuation,
                    strip_multiple_whitespaces, strip_numeric,
                    remove_stopwords
                    ] 
        if stem == True:
            filters = [
                    lambda x: x.lower(), strip_tags, strip_punctuation,
                    strip_multiple_whitespaces, strip_numeric,
                    remove_stopwords, stem_text
                    ]             
        return filters        

    def prep_data(self, emails, labels):
        '''
        Add the labels to the data
        '''
        emails = emails.rename(columns={'uid': 'eid'})
        
        cnames = []        
        for i in range(self.annotators):
            name = 'lab' + str(i)
            cnames.append(name)
            
        emailslab = emails.merge(labels, on=['eid'], how='inner')
                # Split the labels for different annotators

        emailslab[cnames] = emailslab['labels'].str.split(';', expand=True)
        emailslab = emailslab.drop(columns='labels')
            
        del emails

        # Compute average label and make binary label
        emailslab[cnames] = emailslab[cnames].apply(pd.to_numeric)
        emailslab[cnames] = emailslab[cnames].replace(6, None)  # 6 means cannot determine
        emailslab['lab_avg'] = np.floor(emailslab[cnames].mean(axis=1))

        emailslab.loc[emailslab['lab_avg'] <= 2, 'lab_bin'] = 0
        emailslab.loc[emailslab['lab_avg'] > 2, 'lab_bin'] = 1

        return emailslab

    
    def order_data(self, emailslab):
        '''
        Takes emails with labels and sorts them based on time-date variable
        '''
        # Convert date/time field
        emailslab = emailslab.dropna(subset = ['date_time'])
        emailslab['date_time'] = [list(d.values())[0] for d in emailslab['date_time']]
        emailslab['date_time'] = pd.to_datetime(emailslab['date_time'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')

        # Sort from oldest to newest
        emailslab.sort_values(by=['date_time'], ascending=True, inplace = True)

        return emailslab
    
    
    def prep(self, x, stem = False):
        '''
        Preprocessing function
        '''
        return ' '.join(preprocess_string(x, filters = self.filters(stem)))
    
    
    def top_tfidf_feats(self, row, features, top_n=20):
        '''
        Function for tfidf top table
        '''
        topn_ids = np.argsort(row)[::-1][:top_n]
        top_feats = [(features[i], row[i]) for i in topn_ids]
        df = pd.DataFrame(top_feats, columns=['features', 'score'])
        return df

    
    def top_mean_feats(self, D, features, top_n=25):
        '''
        Second function for tifdf top table
        '''
        tfidf_means = np.round(np.mean(D, axis=0), decimals = 3)
        
        return self.top_tfidf_feats(tfidf_means, features, top_n)
    
    def maketables(self, emailslab, vect):
        '''
        Makes tables for data reporting
        '''
        frame = np.array(["Mails","Oldest", "Newest", "Mean Length", "Std. Length", "Min. Length", "Max. Length"]).reshape((-1,1))
        headers = ["Train bus", "Train pers", "Test bus", "Test pers"]
        
        X_train, X_test = emailslab.loc[:int(len(emailslab)*0.8),:], emailslab.loc[int(len(emailslab)*0.8):,:]
        
        # Make statistics table
        reports = []
        for dataset in [X_train, X_test]:
            
            bus_train, pers_train = dataset.loc[dataset.lab_bin == 0], dataset.loc[dataset.lab_bin == 1] 
            for data in [bus_train, pers_train]:
                bodies = data.loc[:,'body']
                data['emaillen'] = bodies.apply(len)
                data = data[data['emaillen']!=0]
                
                number = len(data)
                oldest= min(data['date_time']).strftime("%b %d, %Y")
                newest = max(data['date_time']).strftime("%b %d, %Y")
                
                results = []
                for stat in [np.mean, np.std, np.min, np.max]:
                    res = stat(data['emaillen'])
                    res = np.round(res, decimals = 2)
                    results.append(res)

                
                report = np.array([number, oldest, newest, results[0], results[1], results[2], results[3]])
                reports.append(report)
           
        reports = np.transpose(np.vstack(reports)) 
        frame = np.hstack([frame, reports])
        statstab = tabulate(frame, headers, tablefmt = 'latex')
        
       # Make top word lists
        n = 35
        wordlists = np.arange(1,n+1).reshape(-1,1)
        
        for dclass in [bus_train, pers_train]:
            X = vect.transform(dclass.loc[:,'body']).toarray()

            features = vect.get_feature_names()
            top = self.top_mean_feats(X, features, top_n = n)
            wordlists = np.hstack((wordlists, top))
        
        headers = ["Business Feature", "Business Score", "Personal Feature", "Personal Score"]
        wordtab = tabulate(wordlists, headers, tablefmt = 'latex')
        
        return statstab, wordtab
        
    def makegraphs(self, emailslab, X):
        '''
        Makes graphs for data reporting
        '''
        # Make vector graph
        
        y = emailslab.loc[:,'lab_bin']
        y[y==0.0] = "Business"
        y[y==1.0] = "Personal"
        
        p = 40
        Xt = TSNE(n_components=2, perplexity = p, random_state=0).fit_transform(X)
       
        fig = plt.figure(figsize=(9, 7))
        
        nbatch = 4
        
        step = int(len(emailslab) / nbatch)
        index = int(str(int(nbatch/2)) + str(2) + str(1))
        
        for i in np.arange(step, len(emailslab), step = step):
            ax = fig.add_subplot(index)
            
            start = i - step
            end = i
            X0, X1 = Xt[start:end, 0], Xt[start:end, 1]
            y_part = y[start:end]
            cdict = {"Business": 'skyblue', "Personal": 'coral'}
            
            for g in np.unique(y_part):
                ix = np.where(y_part == g)
                scatter = ax.scatter(X0[ix], X1[ix], c = cdict[g], label = g, s = 5, alpha = 0.7)
        
            plt.legend(["Business", "Personal"], loc=2)
            plt.title("Email " + str(start) + " to " + str(end) )
             
            index += 1
            
        fig.tight_layout()
        plt.show()
        fig.savefig('../output/VectorGraphPer' + str(p) + '.pdf', format = 'pdf')
        
        # Make histogram
        teststart = int(11050*0.8)
        df = emailslab
        df['dates'] = [datetime.strptime(str(d)[:-15], '%Y-%m-%d') for d in df['date_time']]

        plt.figure(figsize=(4,3), dpi= 80)
        fig, ax = plt.subplots()

        plt.ylabel = "Emails"

        x1 = df.loc[df.lab_bin==0, 'dates']
        x2 = df.loc[df.lab_bin==1, 'dates']

        formatter = mdates.DateFormatter("%b %Y")
        ax.xaxis.set_major_formatter(formatter)
        year = mdates.YearLocator()
        ax.xaxis.set_major_locator(year)

        ax.axvline(df.loc[teststart, 'dates'], linestyle = '--', color='black', label='Start Testset')
        plt.hist([x1, x2], bins = 40, stacked = True, color = ('skyblue', 'coral'), label = ["Business", "Personal"])

        ax.set_xlabel("Time")
        ax.set_ylabel("Number of emails")
        ax.xaxis.set_visible(True)
        plt.legend(loc = 2)
        plt.show()
        plt.savefig('../output/timeplot.eps', format = 'eps')
        
        
    def report(self, emailslab):
        '''
        Makes tables and graphs for data reporting
        '''
        vect = TfidfVectorizer(max_features = 5000, min_df = 5, max_df = 0.7)
        X = vect.fit_transform(emailslab.loc[:,'body']).toarray()
        
        statstab, wordtab = self.maketables(emailslab, vect)
        self.makegraphs(emailslab, X)
        
        return statstab, wordtab
         

    def run(self, path, emailname, labname):
        '''
        Stores all emails in online and offline setting
        '''
        emailpath = path + emailname
        emails = pd.read_json(emailpath, encoding='UTF-8', lines=True)
        
        labpath = path + labname
        labels = pd.read_csv(labpath)
        
        emailslab = self.prep_data(emails, labels)
        emailslab = self.order_data(emailslab)
        
        self.report(emailslab)
        
        with open(path + '/enronlab.pickle', 'wb') as f: # offline
             pickle.dump(emailslab, f)

        for index, row in emailslab.iterrows(): # online
            try:
                os.mkdir(path + "/emails")
            except:
                pass
            
            filename = path + "/emails/email" + str(index) + ".pickle"
            with open(filename, 'wb') as f:
                 pickle.dump(row, f)
        return emailslab
    
            
    def fetch(self, online = True, index = 0):
        '''
        Fetches emails in online or offline setting
        '''
        if online == False:
            email = pickle.load( open( "../data/enronlab.pickle", "rb" ) )
        else:
            filename = "../data/emails/email" + str(index) + ".pickle"
            email = pickle.load( open( filename, "rb" ) )            
        return email

if __name__ == "__main__":

    PATH = '../data/'
    EMAILNAME = 'emails.json'
    LABNAME = 'enron_emails.csv'
    ANNOTATORS = 8

    load = DataLoader(annotators=ANNOTATORS)
#     emails = load.run(path=PATH, emailname=EMAILNAME, labname=LABNAME)
    
    # Make table
    emails = load.fetch(online = False)
    report, wordtab = load.report(emails)
    print(report)
    print(wordtab)
  

