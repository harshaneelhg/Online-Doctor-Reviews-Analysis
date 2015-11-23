#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import textblob as tb
import scipy.io
import sys
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import pdb

def get_pos_neg_score(s1):
	blob = tb.TextBlob(s1).correct()
	sid = SIA()
	p = sid.polarity_scores(str(s1))
	if  p['compound'] == 0:
		senti_score = 1 - 0.02
	else:
		senti_score = 1 - (0.5 + p['compound']/2)
	return [senti_score]

reload(sys)
sys.setdefaultencoding('utf')
f = open('../Data/data.txt','rb')
doc_id, rating, staff = [], [], []
punctuality, help, knowledge = [], [], []
review, score= [], []
tup_list = []
line = f.readline()
count = 0
while line != '':
	parts = line.split('|')
	if len(parts[7]) > 1:
		try:
			x = get_pos_neg_score(parts[7])
		except UnicodeDecodeError:
			print "Error occured at line :",line
			line = f.readline()
			continue
		doc_id.append(parts[0])
		rating.append(float(parts[1]))
		staff.append(float(parts[2]))
		punctuality.append(float(parts[3]))
		help.append(float(parts[4]))
		knowledge.append(float(parts[5]))
		review.append(parts[7])
		score.append(x[0])
		tup_list.append((str(parts[0]), float(parts[1]), float(parts[2]),
	 		float(parts[3]), float(parts[4]), float(parts[5]), x[0]))
	line = f.readline()
	count+=1
	if count%1 == 0:
		x = '-'*int(count*50.0/1000) + '>' + '_'*int((count-1000)*50.0/1000) + '| ' + str(int(count*100.0/1000)) + '%'
		sys.stdout.flush()
		sys.stdout.write('%s\r' % x)

df = pd.DataFrame({
					'doc_id': doc_id,
					'rating': rating,
					'staff' : staff,
					'punctuality' : punctuality,
					'help' : help,
					'knowledge' : knowledge,
					'review' : review,
					'sentiment_score' : score,
				 })
df.to_csv('../Data/data_sentiment.csv', index = False)
scipy.io.savemat('../Data/rating_list.mat', {'rating_list': tup_list})
