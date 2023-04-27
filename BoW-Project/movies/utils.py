# Bow Project - RITAL
# Ben Kabongo & Sofia Borchani
# Master DAC
# 
# Fonctions utiles et de préprocessing

import codecs
import pandas as pd
import re
import nltk
from nltk.stem.snowball import EnglishStemmer
from nltk.stem import WordNetLemmatizer
import os

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
    stemmer = EnglishStemmer()
    tokens = nltk.word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    stemmed_text = " ".join(stemmed_tokens)
    return stemmed_text

def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    lemmatized_text = " ".join(lemmatized_tokens)
    return lemmatized_text

# Chargement des données: train

def load_movies(fname="./datasets/movies/movies1000/"):
    alltxts = [] # init vide
    labs = []
    cpt = 0
    #for cl in os.listdir(path2data): # parcours des fichiers d'un répertoire
    for cl in ['neg', 'pos']: # parcours des fichiers d'un répertoire
        for f in os.listdir(fname+cl):
            txt = open(fname+cl+'/'+f).read()
            alltxts.append(txt)
            labs.append(cpt)
        cpt+=1 # chg répertoire = cht classe
        
    all_movies_df = pd.DataFrame()
    all_movies_df['text'] = alltxts
    all_movies_df['label'] = labs
        
    return all_movies_df
    
def load_movies_T(fname="./datasets/movies/testSentiment.txt"):
    return open(fname).readlines()

