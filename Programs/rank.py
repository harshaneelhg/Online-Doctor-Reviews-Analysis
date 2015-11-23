#!/usr/bin/env python
from modules import compute_rank_correlation
import matplotlib.pyplot as plt
import scipy.io
import json
import numpy as np

if __name__ == '__main__':
	data = scipy.io.loadmat('../Data/graph.mat')
	mat = data['mat']
	n = data['n']
	x = data['n_reviews']

	tup_list = scipy.io.loadmat('../Data/rating_list.mat')['rating_list'].tolist()
	tup_list = [
				[float(t[i]) if i>0 else t[i] for i in range(len(t))]
				for t in tup_list
			   ]

	with open('../Data/groups.json','rb') as infile:
		groups = json.load(infile)
	a = []
	b = []
	key_ord = (4,3,2,1,0)
	for c in range(0,101):
		i = c*1.0/100
		cor = compute_rank_correlation(i,key_ord,mat,n,x,tup_list,groups)
		a.append(i)
		b.append(cor)
	plt.plot(a,b,'b-')
	plt.xlabel('Restart probability')
	plt.ylabel("Spearman's ranking correlation coefficient")
	plt.title('Performence of Random Walk with Restarts algorithm')
	plt.grid()
	plt.show()
	print max(b),np.argmax(b)
