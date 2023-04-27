# Bow Project - RITAL
# Ben Kabongo & Sofia Borchani
# Master DAC
# 
# Grid search

import codecs
import json
import nltk
import os
import pandas as pd
import re
import sys
from itertools import combinations, product
from imblearn.under_sampling import RandomUnderSampler
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, 
                            precision_recall_fscore_support, 
                            f1_score, roc_auc_score)
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

# Chargement des données: train

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

    


# Grid search

def grid_search(n_preprocessors=0):

    fname="./datasets/AFDpresidentutf8/corpus.tache1.learn.utf8"
    all_locuteur_df = load_pres(fname)

    LOG = open(f'logs/grid_{n_preprocessors}.log', 'w')

    preprocessors = [
        delete_punctuation, 
        delete_digit, 
        replace_maj_word, 
        stem,
        delete_balise,
    ]
    tfidf_params = {
        'stop_words': [stopwords.words('french')],
        'max_df': [.5, .8, .9, 1.], 
        'min_df': [2, 3, 5], 
        'ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)], 
        'binary': [False, True],
        'lowercase': [False, True],
        'use_idf': [False, True],
        'sublinear_tf': [True, False],
        'max_features': [None, 2_000, 5_000, 10_000, 20_000, 50_000],
    }

    best_params = None
    best_scores = {'f1_min': 0, 'f1': 0, 'auc': 0, 'acc': 0, 'bacc': 0}
    
    for preprocessors_combination in combinations(preprocessors, n_preprocessors):
            
        # préprocessing
        preprocessor_func = None
        if n_preprocessors == 1:
            preprocessor_func = preprocessors_combination[0]
        elif n_preprocessors == 2:
            f, g = preprocessors_combination
            preprocessor_func = lambda x: f(g(x))
        elif n_preprocessors == 3:
            f, g, h = preprocessors_combination
            preprocessor_func = lambda x: f(g(h(x)))
        elif n_preprocessors == 4:
            f, g, h, i = preprocessors_combination
            preprocessor_func = lambda x: f(g(h(i(x))))
        elif n_preprocessors == 5:
            f, g, h, i, j = preprocessors_combination
            preprocessor_func = lambda x: f(g(h(i(j(x)))))
           
        for tfidf_param_combination in product(*tfidf_params.values()):
            # paramètres de tf-idf
            tfidf_param_dict = dict(zip(tfidf_params.keys(), tfidf_param_combination))
            tfidf = TfidfVectorizer(preprocessor=preprocessor_func, **tfidf_param_dict)

            X_text_train, X_text_test, y_train, y_test = train_test_split(all_locuteur_df['text'], 
                all_locuteur_df['label'], test_size=0.2, random_state=42)

            X_train = tfidf.fit_transform(X_text_train)
            X_test = tfidf.transform(X_text_test)
            
            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

            clf = LogisticRegression()
            clf.fit(X_resampled, y_resampled)

            y_pred = clf.predict(X_test)
            
            f1 = f1_score(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            bacc = balanced_accuracy_score(y_test, y_pred)
            precision, recall, f1_s, support = precision_recall_fscore_support(y_test, y_pred)
            f1_min = f1_s[0]
            auc = roc_auc_score(y_test, y_pred)

            #report = classification_report(y_test, y_pred)
            #print(report)


            del tfidf_param_dict['stop_words']
            print('---------------------------------------------------')
            print('Params:',
                '\n\tpreprocessors:', preprocessors_combination, 
                '\n\ttfidf_params:', tfidf_param_dict
            )
            print("F1-score -1\t:", f1_min)
            print('Accuracy \t:', acc)
            print('Bal. acc \t:', bacc)
            print('F1 score \t:', f1)
            print('AUC \t\t:', auc)

            LOG.write('\n\n')
            LOG.write('Params:' +
                '\n\tpreprocessors : ' + str(preprocessors_combination) +
                '\n\ttfidf_params : ' + str(tfidf_param_dict)
            )
            LOG.write('\nF1-score -1: ' + str(f1_min))
            LOG.write('\nF1-score: ' + str(f1))
            LOG.write('\nAUC : '+ str(auc))
            LOG.write('\nAcc: ' + str(acc))
            LOG.write('\nB. Acc: ' + str(bacc))
            if f1_min > .60:
                LOG.write('*******************************************')

            if f1_min > best_scores['f1_min']:
                best_scores = {
                    'f1_min': f1_min,
                    'f1': f1,
                    'auc': auc,
                    'acc': acc,
                    'bacc': bacc
                }
                best_params = {
                    'preprocessors': preprocessors_combination, 
                    'tfidf_params': tfidf_param_dict
                }

    print('---------------------------------------------------')
    print('Best params:', best_params)
    print('Best scores:', best_scores)
    return best_params


# main

if __name__ == '__main__':
    n_preprocessors = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    best_params = grid_search(n_preprocessors=n_preprocessors)
    json.dump(best_params, open(f'best_params_{n_preprocessors}_save.json', 'wb'))