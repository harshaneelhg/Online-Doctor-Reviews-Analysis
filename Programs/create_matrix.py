#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import scipy.sparse
import scipy.io
import json
import sys
import pdb
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
		mat[x+0,i] = 1.0/help[i] if help[i] != 0.0 else sys.maxint
		mat[i,x+0] = 1.0/help[i] if help[i] != 0.0 else sys.maxint
		mat[x+1,i] = 1.0/knowledge[i] if knowledge[i] != 0.0 else sys.maxint
		mat[i,x+1] = 1.0/knowledge[i] if knowledge[i] != 0.0 else sys.maxint
		mat[x+2,i] = 1.0/punctuality[i] if punctuality[i] != 0.0 else sys.maxint
		mat[i,x+2] = 1.0/punctuality[i] if punctuality[i] != 0.0 else sys.maxint
		mat[x+3,i] = 1.0/staff[i] if staff[i] != 0.0 else sys.maxint
		mat[i,x+3] = 1.0/staff[i] if staff[i] != 0.0 else sys.maxint
		mat[x+4,i] = 1.0/rating[i] if rating[i] != 0.0 else sys.maxint
		mat[i,x+4] = 1.0/rating[i] if rating[i] != 0.0 else sys.maxint
		mat[x+5,i] = sentiment_score[i]
		mat[i,x+5] = sentiment_score[i]
		mat[x+6,i] = 0.02
		mat[i,x+6] = 0.02
		groups[str(ids[i])].append(i)

	f_mat = mat[n-y+1:n,0:x]
	pdb.set_trace()
	vals = []

	for i in range(7):
		arr = np.array(f_mat[i].todense()).flatten(-1).tolist()
		vals.append(get_entropy(arr))
	print "Entropies:", vals
	sv = sum(vals)
	for i in range(7):
		mat[n-1,x+i] = sv*1.0/vals[i] if vals[i] != 0.0 else sys.maxint
		mat[x+i,n-1] = sv*1.0/vals[i] if vals[i] != 0.0 else sys.maxint

	scipy.io.savemat('../Data/graph.mat', {'mat': mat, 'n': n, 'n_reviews':x})
	with open('../Data/groups.json', 'wb') as outfile:
		json.dump(groups, outfile)
