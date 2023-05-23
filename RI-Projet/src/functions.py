import itertools
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import kendalltau, pearsonr
from metriques import *


def recommander_evaluation(algo, 
                           seuil, 
                           queries, 
                           docs,
                           ratings_df,
                           query_col_name='query_id',
                           doc_col_name='doc_id',
                           rating_col_name='rating'
                          ):
    predictions = dict()
    for query in queries:
        preds_df = pd.DataFrame()
        preds_df[doc_col_name] = docs
        preds_df['score'] = [algo.predict(uid=query, iid=doc).est for doc in docs]
        preds_df = preds_df.sort_values(by='score', ascending=False)
        query_ratings_df = ratings_df[ratings_df[query_col_name] == query][[doc_col_name, rating_col_name]]
        preds_df = preds_df.merge(query_ratings_df, on=doc_col_name, how='left')
        preds_df['rel'] = preds_df[rating_col_name].fillna(0)
        preds_df['rank'] = preds_df['score'].rank(ascending=False).apply(int)
        preds_df['bin'] = preds_df['rel'].apply(lambda x: 1 if x >= seuil else 0)
        predictions[query] = preds_df[[doc_col_name, 'score', 'rel', 'rank', 'bin']]
    return predictions


def compare(predictions1, predictions2, queries):
    rpp_all, dcgpp_all, invpp_all, ndcg_all, dcg_all, ap_all, rr_all, asl_all, oi_all = [], [], [], [], [], [], [], [], []
    
    for query_id in queries:
        rankings1 = np.array(predictions1[query_id][predictions1[query_id]["bin"] == 1]['rank'])
        rankings2 = np.array(predictions2[query_id][predictions2[query_id]["bin"] == 1]['rank'])
        if len(rankings1) == 0:
            rankings1 = np.append(rankings1, len(predictions1[query_id]))
        if len(rankings2) == 0:
            rankings2 = np.append(rankings2, len(predictions2[query_id]))
        
        relevant1 = np.array(predictions1[query_id]['rel'])
        relevant2 = np.array(predictions2[query_id]['rel'])
        
        rpp_all.append( RPP(rankings1, rankings2) )
        dcgpp_all.append( DCGPP(rankings1, rankings2) )
        invpp_all.append( INVPP(rankings1, rankings2) )
        ndcg_all.append( delta_NDCG(relevant1, relevant2) )
        dcg_all.append( delta_DCG(relevant1, relevant2) )
        ap_all.append( delta_AP(rankings1, rankings2) )
        rr_all.append( delta_RR(rankings1, rankings2) )
        asl_all.append(delta_ASL(rankings1, rankings2) )
        oi_all.append(OI(rankings1, rankings2) )
        
    return  rpp_all, dcgpp_all, invpp_all, ndcg_all, dcg_all ,ap_all, rr_all, asl_all, oi_all


def robustess(predictions1,
            predictions2,
            queries,
            docs,
            rpp_all,
            dcgpp_all,
            invpp_all,
            ndcg_all,
            ap_all,
            rr_all,
            doc_col_name='doc_id'):
    
    rpp_tau, dcgpp_tau, invpp_tau, ndcg_tau, ap_tau, rr_tau = [], [], [], [], [], []
    
    for missing_rate in range(90, 0, -10):
        rpp, dcgpp, invpp, ndcg, ap, rr = [], [], [], [], [], []
    
        for query_id in queries:
            preds1_df = predictions1[query_id]
            preds2_df = predictions2[query_id]
        
            n_docs = missing_rate * len(docs) // 100
            missing_docs = np.random.choice(docs, n_docs)
            
            missing_preds1_df = preds1_df.drop(preds1_df[preds1_df[doc_col_name].isin(missing_docs)].index)
            missing_preds1_df["rank"] = missing_preds1_df["score"].rank(ascending=False).apply(int)
            rankings1 = np.array(missing_preds1_df[missing_preds1_df["bin"] == 1]['rank'])
            relevant1 = np.array(missing_preds1_df['rel'])
            
            missing_preds2_df = preds2_df.drop(preds2_df[preds2_df[doc_col_name].isin(missing_docs)].index)
            missing_preds2_df["rank"] = missing_preds2_df["score"].rank(ascending=False).apply(int)
            rankings2 = np.array(missing_preds2_df[missing_preds2_df["bin"] == 1]['rank'])
            relevant2 = np.array(missing_preds2_df['rel'])
            
            if len(rankings1) == 0:
                rankings1 = np.append(rankings1, len(missing_preds1_df))
            if len(rankings2) == 0:
                rankings2 = np.append(rankings2, len(missing_preds2_df))

            rpp.append( RPP(rankings1, rankings2) )
            dcgpp.append( DCGPP(rankings1, rankings2) )
            invpp.append( INVPP(rankings1, rankings2) )
            ndcg.append( delta_NDCG(relevant1, relevant2) )
            ap.append( delta_AP(rankings1, rankings2) )
            rr.append( delta_RR(rankings1, rankings2) )
            
        rpp_tau.append( kendalltau(rpp, rpp_all)[0] )
        dcgpp_tau.append( kendalltau(dcgpp, dcgpp_all)[0] )
        invpp_tau.append( kendalltau(invpp, invpp_all)[0] )
        ndcg_tau.append( kendalltau(ndcg, ndcg_all)[0] )
        ap_tau.append( kendalltau(ap, ap_all)[0] )
        rr_tau.append( kendalltau(rr, rr_all)[0] )
        
    return rpp_tau, dcgpp_tau, invpp_tau, ndcg_tau, ap_tau, rr_tau


def t_test_with_bonferroni_correction(data):
    # Effectue un test T de Student avec correction de Bonferroni
    # sur la liste de données d'entrée
    _, p_val = stats.ttest_ind(data, [0]*len(data))
    p_val *= len(data) # applique la correction de Bonferroni
    return p_val


def power(all_runs, queries, test_fn=t_test_with_bonferroni_correction):
    N = len(all_runs)
    rpp_pvalue, dcgpp_pvalue, invpp_pvalue, ndcg_pvalue, ap_pvalue, rr_pvalue = [], [], [], [], [], []
    for (i, j) in list(itertools.combinations(range(N), 2)):
        rpp_all, dcgpp_all, invpp_all, ndcg_all, _ ,ap_all, rr_all, _, _ = compare(
            all_runs[i], all_runs[j], queries
        )
        rpp_pvalue.append( test_fn(rpp_all) )
        dcgpp_pvalue.append( test_fn(dcgpp_all) )
        invpp_pvalue.append( test_fn(invpp_all) )
        ndcg_pvalue.append( test_fn(ndcg_all) )
        ap_pvalue.append( test_fn(ap_all) )
        rr_pvalue.append( test_fn(rr_all) )
        
    pvalue = .05
    rpp_ratio = np.where(np.array(rpp_pvalue) < pvalue, 1, 0).mean() * 100
    dcgpp_ratio = np.where(np.array(dcgpp_pvalue) < pvalue, 1, 0).mean() * 100
    invpp_ratio = np.where(np.array(invpp_pvalue) < pvalue, 1, 0).mean() * 100
    ndcg_ratio = np.where(np.array(ndcg_pvalue) < pvalue, 1, 0).mean() * 100
    ap_ratio = np.where(np.array(ap_pvalue) < pvalue, 1, 0).mean() * 100
    rr_ratio = np.where(np.array(rr_pvalue) < pvalue, 1, 0).mean() * 100
    
    return rpp_ratio, dcgpp_ratio, invpp_ratio, ndcg_ratio, ap_ratio, rr_ratio
