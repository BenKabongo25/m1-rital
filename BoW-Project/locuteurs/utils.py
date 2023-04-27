# Bow Project - RITAL
# Ben Kabongo & Sofia Borchani
# Master DAC
# 
# Fonctions utiles et de préprocessing

import codecs
import pandas as pd
import re
import nltk
from nltk.stem.snowball import FrenchStemmer

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

