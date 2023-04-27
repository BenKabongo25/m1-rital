# Bow Project - RITAL
# Ben Kabongo & Sofia Borchani
# Master DAC
# 
# Prédictions

import codecs
import json
import numpy as np
import pandas as pd
import nltk
import re
from itertools import combinations, product
from imblearn.under_sampling import RandomUnderSampler
from nltk.stem.snowball import FrenchStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split


# Préprocessing

def delete_punctuation(text):
    punctuation = r"[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~\n\t]"
    text = re.sub(punctuation, " ", text)
    text = re.sub('( )+', ' ', text)
    return text

def replace_maj_word(text):
    token = '<MAJ>' # mot pour remplacer les mots en majuscules
    return ' '.join([w if not w.isupper() else token for w in delete_punctuation(text).split()])

def delete_digit(text):
    return re.sub('[0-9]+', '', text)

def first_line(text):
    return re.split(r'[.!?]', text)[0]

def last_line(text):
    if text.endswith('\n'): text = text[:-2]
    return re.split(r'[.!?]', text)[-1]

def delete_balise(text):
    return re.sub("<.*?>", "", text)

def stem(text):
    stemmer = FrenchStemmer()
    tokens = nltk.word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    stemmed_text = " ".join(stemmed_tokens)
    return stemmed_text

# Chargement des données

def load_pres(fname="./datasets/AFDpresidentutf8/corpus.tache1.learn.utf8"):
    alltxts = []
    alllabs = []
    s = codecs.open(fname, 'r','utf-8') # pour régler le codage
    while True:
        txt = s.readline()
        if(len(txt))<5:
            break
        #
        lab = re.sub(r"<[0-9]*:[0-9]*:(.)>.*","\\1",txt)
        txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)","\\1",txt)
        if lab.count('M') >0:
            alllabs.append(-1)
        else: 
            alllabs.append(1)
        alltxts.append(txt)
    all_locuteur_df = pd.DataFrame()
    all_locuteur_df['text'] = alltxts
    all_locuteur_df['label'] = alllabs
    return all_locuteur_df

def load_pres_test(fname="./datasets/AFDpresidentutf8/corpus.tache1.test.utf8"):
    alltxts = []
    s = codecs.open(fname, 'r','utf-8') # pour régler le codage
    while True:
        txt = s.readline()
        if(len(txt))<5:
            break
        txt = re.sub(r"<[0-9]*:[0-9]*>(.*)","\\1",txt).strip()
        alltxts.append(txt)
    all_locuteur_test_df = pd.DataFrame()
    all_locuteur_test_df['text'] = alltxts
    return all_locuteur_test_df


# main

if __name__ == '__main__':
    fname = "corpus.tache1.learn.utf8"
    all_locuteur_df = load_pres(fname)

    fname = "corpus.tache1.test.utf8"
    all_locuteur_test_df = load_pres_test(fname)

    preprocessor = lambda x: delete_digit(last_line(delete_balise(x)))
    tfidf_params = {'max_df': 0.9, 'min_df': 2, 'ngram_range': (1, 3), 'binary': True, 'lowercase': False}
    lr_params = {'C': 0.1, 'penalty': 'none'}

    tfidf = TfidfVectorizer(preprocessor=preprocessor, **tfidf_params)
    X_train = tfidf.fit_transform(all_locuteur_df.text)
    y_train = all_locuteur_df.label
    X_test = tfidf.transform(all_locuteur_test_df.text)
    
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    lr = LogisticRegression(class_weight='balanced', **lr_params)
    lr.fit(X_resampled, y_resampled)

    # evaluation ??
    y_pred = lr.predict(X_train)
    f1 = f1_score(y_train, y_pred)
    auc = roc_auc_score(y_train, y_pred)

    print('F1-score:', f1)
    print('AUC:', auc)

    # prédiction ??
    y_test = lr.predict(X_test)
    arr = np.where(y_test == -1, 'M', 'C')
    np.savetxt('output.txt', arr.reshape(-1, 1), fmt='%s')
