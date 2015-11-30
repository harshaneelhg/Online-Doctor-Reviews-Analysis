#!/usr/bin/python
# -*- coding: UTF-8 -*-

from modules import compute_rank_correlation
import matplotlib.pyplot as plt
import scipy.io
import json
import numpy as np

if __name__ == '__main__':
	# Read adjecency matrix and matrix parameters.
	data = scipy.io.loadmat('../Data/graph.mat')
	mat = data['mat']
	n = data['n']
	x = data['n_reviews']
	tup_list = scipy.io.loadmat('../Data/rating_list.mat')['rating_list'].tolist()
	tup_list = [[float(t[i]) if i>0 else t[i] for i in range(len(t))]for t in tup_list]
	# Read groups that identify different reviews for the same doctor.
	with open('../Data/groups.json','rb') as infile:
		groups = json.load(infile)
	# Initialize some variables to hold plotting data
	# and specify key order used for sorting.
	a = []
	b = []
	p = []
	key_ord = (4,3,2,1,0)
	# Iterate over restart probabilities of random walk algorithm.
	for c in range(0,101):
		i = c*1.0/100
		cor, pval = compute_rank_correlation(i,key_ord,mat,n,x,tup_list,groups)
		a.append(i)
		b.append(cor)
		p.append(pval)

	# Plot results.
	fig, ax1 = plt.subplots()
	plt.title('Performance of RWR and the effect of restart probability')
	ax1.plot(a,b,'b-',label='Performence of RWR')
	ax1.set_xlabel('Restart probability')
	ax1.set_ylabel("Spearman's ranking correlation coefficient")

	ax2 = ax1.twinx()
	ax2.plot(a,p,'r-',label='Significance value')
	ax2.set_ylabel('Significance value')
	ax2.set_yscale('log')
	h1, l1 = ax1.get_legend_handles_labels()
	h2, l2 = ax2.get_legend_handles_labels()
	ax1.legend(h1+h2, l1+l2)
	plt.grid()
	plt.show()

	best_corr = max(abs(np.array(b)))
	best_c = np.argmax(b)
	print best_corr, best_c, p[best_c]
