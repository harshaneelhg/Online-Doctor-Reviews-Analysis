#!/usr/bin/env python

import numpy as np
import scipy.sparse
from sklearn.preprocessing import normalize
import scipy.io
import json
import scipy.stats
import matplotlib.pyplot as plt
import pdb

#========================================== FUNCTION DEFINITIONS ==========================================
#==========================================================================================================

def get_ranks_rwr(q, c, W):
	"""
		Input:
		q: Sparse query vector.
		c: Restart probabilities.
		W: Sparse adjecency matrix.

		Output:
		r: Sparse relevancy vector with every other node.

		This function implements basic version of RWR to output
		Relevancy of query node with other nodes in a graph represented
		by adjecency matrix W.
	"""

	#Column Normalize Adjecency Matrix.

	W = normalize(W, norm='l1', axis=0)

	#Basic Random Walk with Restarts Algorithm.
	r = q
	r1 = (1-c)*(r*W) + c*q
	i=0

	while (r1-r).dot((r1-r).T) > 1e-5:
		r = r1
		r1 = (1-c)*(r*W) + c*q
		i= i+1

	#print 'Reached to the convergence in %s iterations.'%i
	return r1

def main_func(c,key_ord):
	data = scipy.io.loadmat('../Data/graph.mat')
	mat = data['mat']
	n = data['n']
	x = data['n_reviews']

	tup_list = scipy.io.loadmat('../Data/rating_list.mat')['rating_list'].tolist()
	tup_list = [[float(t[i]) if i>0 else t[i] for i in range(len(t))] for t in tup_list]
	#pdb.set_trace()

	with open('../Data/groups.json','rb') as infile:
		groups = json.load(infile)

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

	ret = 1 - ((6.0*d_2)/(n_ranks*(n_ranks**2-1)))


	#pdb.set_trace()
	#ret = scipy.stats.spearmanr(r1,r2)
	#print ret
	return ret
	#pdb.set_trace()
	#print "Top 20 doctors:"
	#print sorted_ranks[:20]
	#print ""
	#print "Bottom 20 doctors:"
	#print sorted_ranks[150:]

#main_func(0.19)

"""
for a1 in range(0,5):
	for a2 in range(0,5):
		if a2!=a1:
			for a3 in range(0,5):
				if a3!= a2 and a3!=a1:
					for a4 in range(0,5):
						if a4!=a3 and a4!=a2 and a4!=a1:
							for a5 in range(0,5):
								if a5!=a4 and a5!=a3 and a5!=a2 and a5!= a1:
									x = []
									y = []
									p = []
									key_ord = (a1,a2,a3,a4,a5)
									for c in range(0,101):
										i = c*1.0/100
										cor = main_func(i,key_ord)
										x.append(i)
										y.append(cor)
										#p.append(cor[1])

									#plt.plot(x,y,'b-')
									#plt.grid()
									#plt.show()
									print max(y), (a1,a2,a3,a4,a5)
print y
"""
x = []
y = []
p = []
key_ord = (3, 1, 0, 2, 4)
for c in range(0,101):
	i = c*1.0/100
	cor = main_func(i,key_ord)
	x.append(i)
	y.append(cor)
	#p.append(cor[1])
plt.plot(x,y,'b-')
plt.grid()
plt.show()
print max(y)
#fig, ax1 = plt.subplots()
#ax1.plot(x,y,'b-')
#ax1.set_xlabel('Restart probabilities')
#ax1.set_ylabel('Rank correlation coefficient', color='b')

#ax2 = ax1.twinx()
#ax2.plot(x,p,'r-')
#ax2.set_ylabel('p-value', color = 'r')
