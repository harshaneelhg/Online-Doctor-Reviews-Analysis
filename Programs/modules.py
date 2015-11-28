#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import scipy.sparse
from sklearn.preprocessing import normalize
import scipy.io
import json
from scipy.stats import *
import pandas as pd
import textblob as tb
import sys
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import math
import pdb

__all__ = ['get_entropy','get_ranks_rwr', 'compute_rank_correlation']

def get_entropy(arr):
    """
        This function finds out the entropy of an array.

        Input:
        arr (type:List): Array containing data.

        Output:
        entropy: Entropy of data stored in array.
    """
    if type(arr) != list:
        try:
            arr = arr.flatten(-1).tolist()
        except:
            raise TypeError('Incorrect input type: List or ndarray required; found ' + str(type(arr)) + '.')
    unique_elements = list(set(arr))
    count = []
    for ue in unique_elements:
    	count.append(arr.count(ue))

    s = sum(count)
    for i in range(len(count)):
    	count[i] = count[i]*1.0/s

    entropy = 0
    for c in count:
    	entropy += c * math.log(c,2)
    return -1*entropy

def get_ranks_rwr(q, c, W):
	"""
        This function implements Random Walk with Restarts using iterative
        definition. It takes adjecency matrix of the graph, query vector and
        restart probability of the algorithm and computes the ranked order of
        all the nodes.

		Input:
		q: Sparse query vector.
		c: Restart probabilities.
		W: Sparse adjecency matrix.

		Output:
		r: Sparse relevancy vector with every other node.
	"""
	#Column Normalize Adjecency Matrix.
	W = normalize(W, norm='l1', axis=0)
	#Basic Random Walk with Restarts Algorithm.
	r = q
	r1 = (1-c)*(r*W) + c*q
	i=0
	# Iterations of Random Walk Algorithm.
	while (r1-r).dot((r1-r).T) > 1e-5:
		r = r1
		r1 = (1-c)*(r*W) + c*q
		i= i+1
	return r1

def compute_rank_correlation(c,key_ord,mat,n,x,tup_list,groups):
    """
        This function computes Spearman's ranking correlation coefficient
        and the Significance value of ranking. This functions first computes
        probable ranking using RWR algorithm. To compute the Significance
        value of ranking results, it first sorts the data using key ordering
        specified and then finds out the Spearman's ranking correlation
        coefficient. Significance value(two-sided) is computed using Fischer
        projection and Z-score.

        Input:
        c: Restart probability for RWR algorithm.
        key_ord: Key ordering for raw sorting of original data.
        mat: Adjecency matrix of the data.
        n: Size of matrix.
        x: Number of reviews.
        tup_list: List of tuples having all the rows as tuples.
        groups: Dictionary of groups containing rows for individual doctors.

        Output:
        corr: Spearman's ranking correlation coefficient.
        p_val: P-value (Significance test) of ranking prediction.
    """
    q = np.zeros(n,dtype=float)
    q[n-1] = 1.0
    q = scipy.sparse.csc_matrix(q)
    ranks = get_ranks_rwr(q, c, mat).todense().reshape(-1).tolist()[0][:x]

    keys = groups.keys()
    doc_ranks = []
    doc_ranks_crude = []
    for k in keys:
    	values = groups[k]
    	count = 0
    	s = 0.0
    	for v in values:
    		s += ranks[v]
    		count += 1
    	doc_ranks.append((k,s*1.0/count))

    for k in keys:
    	values = groups[k]
    	count = 0
    	s = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    	for v in values:
    		for i in range(5):
    			s[i]+=tup_list[v][i+1]
    		count += 1
    	doc_ranks_crude.append((k, s*1.0/count))

    sorted_ranks = sorted(doc_ranks, key= lambda x: x[1])[::-1]
    sorted_truth = sorted(
    	doc_ranks_crude,
    	key= lambda x: (
    		x[1][key_ord[0]],
    		x[1][key_ord[1]],
    		x[1][key_ord[2]],
    		x[1][key_ord[3]],
    		x[1][key_ord[4]])
    	)[::-1]

    r1 = [x[0] for x in sorted_ranks]
    r2 = [x[0] for x in sorted_truth]

    ranks1 = {}
    for i in range(len(r1)):
    	ranks1[r1[i]] = i+1

    ranks2 = {}
    for i in range(len(r2)):
    	ranks2[r2[i]] = i+1

    n_ranks = len(r1)
    d_2 = 0.0
    for k in r1[:n_ranks]:
    	d_2 += (ranks1[k] - ranks2[k])**2

    corr = 1 - ((6.0*d_2)/(n_ranks*(n_ranks**2-1)))
    F = np.arctan(corr)
    z = (((len(r1)-3)*1.0/1.06)**0.5)*F
    p_val = scipy.stats.norm.sf(abs(z))*2
    return corr, p_val
