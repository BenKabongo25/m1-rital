# RITAL - Master DAC
# Sorbonne Université
# Ben Kabongo
#
# TP 01 - Recherche d'information

import porter
import math
from collections import Counter

#- Question 1.1
 
# retourne la liste des mots d'un document et leurs occurencrs
def words_count(doc):
    words = doc.split(' ')
    words_stem = []
    for w in words:
        if w == '': continue
        w_stem = porter.stem(w)
        words_stem.append(w_stem)
    return dict(Counter(words_stem))

#- Question 1.2

# crée un index pour un corpus de document
def create_index(docs):
    index = {}
    for doc_id, doc in docs.items():
        wc = words_count(doc)
        index[doc_id] = wc
    return index

# crée un index inversé
def reverse_index(index):
    reversed_index = {}
    for doc_id, wc_doc in index.items():
        for w, c in wc_doc.items():
            if w not in reversed_index:
                reversed_index[w] = {}
            w_doc = reversed_index[w]
            w_doc[doc_id] = c
            reversed_index[w] = w_doc
    return reversed_index

#- Question 1.3

# calcule les pondérations tfidf de chaque terme de chaque document
def tfidf(docs):
    index = create_index(docs)
    reversed_index = reverse_index(index)
    tfidf_index = {}
    N = len(docs)
    for doc_id, wc in index.items():
        tfidf_doc = {}
        n = sum(list(wc.values()))
        for w, c in wc.items():
            tf = c/n
            idf = math.log((1 + N) / (1 + len(reversed_index[w])))
            tfidf_doc[w] = tf * idf
        tfidf_index[doc_id] = tfidf_doc
    return tfidf_index

#- Question 2.1

def TAAT(query, docs, K):
    tfidf_scores = tfidf(docs)
    reversed_index = reverse_index(tfidf_scores)

    terms = list(words_count(query).keys())
    # TODO : tri par importance

    all_docs_scores = {}

    for t in terms:
        docs_socres = reversed_index.get(t, {})
        for doc, score in docs_socres.items():
            if doc not in all_docs_scores:
                all_docs_scores[doc] = 0
            all_docs_scores[doc] += score

    # récupération des K premiers documents
    docs_sorted = sorted(all_docs_scores, key=all_docs_scores.get, reverse=True)
    res = {}
    for i, doc in enumerate(docs_sorted):
        if i == K: break
        res[doc] = all_docs_scores[doc]
    return res

#- Question 2.2

def DAAT(query, docs, K):
    # TODO
    pass

#- Question 2.3

def WAND(query, docs, K):
    # TODO
    pass
