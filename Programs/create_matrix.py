#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import scipy.sparse
import scipy.io
import json
import sys
from modules import get_entropy

if __name__ == '__main__':
	data = pd.read_csv('../Data/data_sentiment.csv')
	x = data.shape[0]
	y = data.shape[1]
	n = x+y-1

	mat = scipy.sparse.lil_matrix((n,n), dtype=float)
	ids = data.doc_id.tolist()
	unique_ids = list(set(data.doc_id.tolist()))
	groups={}

	for u in unique_ids:
		groups[str(u)] = []

	help = data.help.tolist()
	knowledge = data.knowledge.tolist()
	punctuality = data.punctuality.tolist()
	staff = data.staff.tolist()
	rating = data.rating.tolist()
	sentiment_score = data.sentiment_score.tolist()

	for i in range(len(ids)):
		mat[x+0,i] = help[i]
		mat[i,x+0] = help[i]
		mat[x+1,i] = knowledge[i]
		mat[i,x+1] = knowledge[i]
		mat[x+2,i] = punctuality[i]
		mat[i,x+2] = punctuality[i]
		mat[x+3,i] = staff[i]
		mat[i,x+3] = staff[i]
		mat[x+4,i] = rating[i]
		mat[i,x+4] = rating[i]
		mat[x+5,i] = sentiment_score[i]
		mat[i,x+5] = sentiment_score[i]
		mat[x+6,i] = 0.1
		mat[i,x+6] = 0.1
		groups[str(ids[i])].append(i)

	f_mat = mat[n-y+1:n,0:x]
	vals = []
	for i in range(7):
		arr = np.array(f_mat[i].todense()).flatten(-1).tolist()
		vals.append(get_entropy(arr))
	#print "Entropies:", vals
	sv = sum(vals)
	for i in range(7):
		mat[n-1,x+i] = vals[i]
		mat[x+i,n-1] = vals[i]

	scipy.io.savemat('../Data/graph.mat', {'mat': mat, 'n': n, 'n_reviews':x})
	with open('../Data/groups.json', 'wb') as outfile:
		json.dump(groups, outfile)
