from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score

import numpy as np
import os
import time

# paramètres 
preprocessor = None
tfidf_params = {
    'stop_words': 'english', 
    'max_df': 1.0, 
    'min_df': 3, 
    'ngram_range': (1, 2), 
    'binary': False, 
    'lowercase': False, 
    'use_idf': False, 
    'smooth_idf': False, 
    'sublinear_tf': True, 
    'max_features': 10000
}
lr_params = {
    'C': 10, 
    'penalty': 'none'
}

''''
preprocessors= lambda x: delete_punctuation(replace_maj_word(x))
tfidf_params={'stop_words': 'english', 'max_df': 0.5, 'min_df': 2, 'ngram_range': (1, 2), 'binary': True, 'lowercase': False, 'use_idf': True, 'max_features': 10000} 
lr_params={'C': 10, 'penalty': 'l2'}

preprocessors= lambda x: delete_punctuation(delete_digit(replace_maj_word(x)))
tfidf_params= {'stop_words': 'english', 'max_df': 0.5, 'min_df': 3, 'ngram_range': (1, 3), 'binary': True, 'lowercase': True, 'use_idf': True, 'max_features': None} 
lr_params= {'C': 10, 'penalty': 'l2'}

preprocessors= lambda x: delete_punctuation(delete_digit((x)))
tfidf_params={'stop_words': 'english', 'max_df': 0.5, 'min_df': 3, 'ngram_range': (1, 3), 'binary': False, 'lowercase': False, 'use_idf': False, 'sublinear_tf': True, 'max_features': None}
lr_params= {'C': 1, 'penalty': 'none'}
'''

# train
fname="./datasets/movies/movies1000/"
all_movies_df = load_movies(fname)

# test
fname_test="./datasets/movies/testSentiment.txt"
all_movies_test = load_movies_T(fname_test)

# TF-IDF
tfidf = TfidfVectorizer(preprocessor=preprocessor, **tfidf_params)
tfidf = TfidfVectorizer(preprocessor=preprocessor, lowercase=True, binary=True, ngram_range=(1, 3), max_features=40_000)
X_train = tfidf.fit_transform(all_movies_df.text)
y_train = all_movies_df.label
X_test = tfidf.transform(all_movies_test)

# LR
lr = LogisticRegression(**lr_params)
lr.fit(X_train, y_train)


# evaluation
y_pred = lr.predict(X_train)
f1 = f1_score(y_train, y_pred)
auc = roc_auc_score(y_train, y_pred)

print('F1-score:', f1)
print('AUC:', auc)


# prédictions
y_pred = lr.predict(X_test)
arr = np.where(y_pred == 0, "N", "P")

# Sauvegarder le résultat dans un fichier
v = len(os.listdir('preds'))
np.savetxt(f"preds/v_00{v}.txt", arr, fmt="%s")