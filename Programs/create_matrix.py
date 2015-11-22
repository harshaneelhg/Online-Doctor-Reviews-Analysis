import pandas as pd
import numpy as np
import scipy.sparse
import scipy.io
import json
import sys
import pdb

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
max_score = data.max_score.tolist()

for i in range(len(ids)):
	mat[x+0,i] = 1.0/help[i] if help[i] != 0.0 else 100.0
	mat[i,x+0] = 1.0/help[i] if help[i] != 0.0 else 100.0
	mat[x+1,i] = 1.0/knowledge[i] if knowledge[i] != 0.0 else 100.0
	mat[i,x+1] = 1.0/knowledge[i] if knowledge[i] != 0.0 else 100.0
	mat[x+2,i] = 1.0/punctuality[i] if punctuality[i] != 0.0 else 100.0
	mat[i,x+2] = 1.0/punctuality[i] if punctuality[i] != 0.0 else 100.0
	mat[x+3,i] = 1.0/staff[i] if staff[i] != 0.0 else 100.0
	mat[i,x+3] = 1.0/staff[i] if staff[i] != 0.0 else 100.0
	mat[x+4,i] = 1.0/rating[i] if rating[i] != 0.0 else 100.0
	mat[i,x+4] = 1.0/rating[i] if rating[i] != 0.0 else 100.0
	mat[x+5,i] = sentiment_score[i]
	mat[i,x+5] = sentiment_score[i]
	mat[x+6,i] = 0.0 #max_score[i]
	mat[i,x+6] = 0.0 #max_score[i]
	mat[x+7,i] = 0.02
	mat[i,x+7] = 0.02
	groups[str(ids[i])].append(i)

#vals  = [3.51051355e-20,  -1.97357609e-10,  -6.11146680e-01,
#         2.74153508e-01,   8.52938809e-01,   4.84054364e-01]
#vals = [2.50437477e+10,   2.99813869e+01,   7.83729703e+00,
#   	    6.19982401e+00,   3.80212724e+00,   2.08235716e+00]
vals = [1.0,1.0,1.0,1.0,1.0,1.0,1.0]
#vals =  [2.50437477e+10,   2.99813869e+01,   7.83729703e+00,
#         6.19982401e+00,   3.80212724e+00,   2.08235716e+00,
#         0.00000000e+00]
sv = sum(vals)
for i in range(0,7):
	mat[n-1,x+i] = vals[i]/sv
	mat[x+i,n-1] = vals[i]/sv

scipy.io.savemat('../Data/graph.mat', {'mat': mat, 'n': n, 'n_reviews':x})
with open('../Data/groups.json', 'wb') as outfile:
	json.dump(groups, outfile)

f_mat = mat[n-y+1:n-1,0:x]

pdb.set_trace()
