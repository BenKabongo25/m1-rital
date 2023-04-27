from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.svm import SVC

import numpy as np
import os

# paramètres 
preprocessor = delete_punctuation
tfidf_params = {
    'max_df': 0.5, 
    'min_df': 2, 
    'ngram_range': (1, 2), 
    'binary': True, 
    'lowercase': False, 
    'use_idf': True, 
    'sublinear_tf': True, 
    'max_features': 50000
}
lr_params = {
    'C': 1, 
    'penalty': 'none'
}

preprocessors= lambda x: delete_punctuation(delete_digit(x)) 
tfidf_params={'max_df': 0.5, 'min_df': 2, 'ngram_range': (1, 3), 'binary': True, 'lowercase': False, 'use_idf': True, 'sublinear_tf': False, 'max_features': 20000}

preprocessors= delete_punctuation
tfidf_params={'max_df': 0.5, 'min_df': 3, 'ngram_range': (1, 3), 'binary': True, 'lowercase': True, 'use_idf': True, 'sublinear_tf': True, 'max_features': 20000}

preprocessors=delete_punctuation
tfidf_params={'max_df': 0.5, 'min_df': 2, 'ngram_range': (1, 2), 'binary': True, 'lowercase': False, 'use_idf': True, 'sublinear_tf': False, 'max_features': None}

tfidf_params={'max_df': 0.5, 'min_df': 2, 'ngram_range': (1, 3), 'binary': False, 'lowercase': True, 'use_idf': True, 'sublinear_tf': True, 'max_features': 50000}

tfidf_params={'max_df': 0.8, 'min_df': 3, 'ngram_range': (1, 3), 'binary': True, 'lowercase': False, 'use_idf': True, 'smooth_idf': True, 'sublinear_tf': True, 'max_features': None}

tfidf_params={'max_df': 0.2, 'min_df': 50, 'ngram_range': (3, 3), 'binary': False, 'lowercase': True, 'use_idf': False, 'smooth_idf': False, 'sublinear_tf': False, 'max_features': 1000}
lr_params ={'C': 0.1, 'penalty': 'none'}

preprocessors= lambda x: delete_punctuation(delete_digit(replace_maj_word(x))) 
tfidf_params={'max_df': 0.5, 'min_df': 5, 'ngram_range': (1, 3), 'binary': True, 'lowercase': False, 'use_idf': True, 'sublinear_tf': False, 'max_features': 50000}

preprocessors=lambda x: delete_punctuation(delete_digit(x)) 
tfidf_params={'max_df': 0.8, 'min_df': 5, 'ngram_range': (1, 3), 'binary': True, 'lowercase': True, 'use_idf': True, 'sublinear_tf': True, 'max_features': 50000}
tfidf_params={'max_df': 0.5, 'min_df': 5, 'ngram_range': (1, 4), 'binary': True, 'lowercase': False, 'use_idf': True, 'sublinear_tf': True, 'max_features': None}

preprocessors=lambda x: delete_punctuation(replace_maj_word(x))
tfidf_params={'max_df': 1.0, 'min_df': 2, 'ngram_range': (1, 2), 'binary': True, 'lowercase': False, 'use_idf': True, 'sublinear_tf': True, 'max_features': 50000}

preprocessors= lambda x: delete_punctuation(delete_digit(replace_maj_word(x))) 
tfidf_params={'max_df': 0.8, 'min_df': 5, 'ngram_range': (1, 3), 'binary': True, 'lowercase': True, 'use_idf': True, 'sublinear_tf': True, 'max_features': 50000}
tfidf_params={'max_df': 0.9, 'min_df': 5, 'ngram_range': (1, 3), 'binary': True, 'lowercase': True, 'use_idf': True, 'sublinear_tf': True, 'max_features': None}
# train
fname="./datasets/AFDpresidentutf8/corpus.tache1.learn.utf8"
all_locuteur_df = load_pres(fname)

# test
fname_test="./datasets/AFDpresidentutf8/corpus.tache1.test.utf8"
all_locuteur_test_df = load_pres_test(fname_test)

# TF-IDF
tfidf = TfidfVectorizer(preprocessor=preprocessor, **tfidf_params)
X_train = tfidf.fit_transform(all_locuteur_df.text)
y_train = all_locuteur_df.label
X_test = tfidf.transform(all_locuteur_test_df.text)

# LR
lr = LogisticRegression(**lr_params)
#lr = SVC()
lr.fit(X_train, y_train)

# evaluation
y_pred = lr.predict(X_train)
f1 = f1_score(y_train, y_pred)
auc = roc_auc_score(y_train, y_pred)

print('F1-score:', f1)
print('AUC:', auc)


# prédictions
y_pred = lr.predict(X_test)
arr = np.where(y_pred == -1, 'M', 'C')

# Sauvegarder le résultat dans un fichier
v = len(os.listdir('preds'))
np.savetxt(f"preds/v_00{v}.txt", arr, fmt="%s")