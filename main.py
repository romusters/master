#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import gensimW2v
import scipy

import utils

import os
import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def main():
	fname = '/home/cluster/Dropbox/Master/resources/20151201:00.out'

	#Load Twitter data (only load geolocation)
	data, geolocations = utils.loadData(fname, True)
	logger.info("Number of tweets: %i", len(data))

	bow = list(utils.getBagOfwords(data))

	import numpy as np
	features = np.zeros((len(data), len(bow)))
	#for every tweet
	for i in range(0, len(data)):
		#for every word in a tweet
		for j in range(0, len(data[i])):
		#Stem data, normalize, remove duplicates and only use geo-enabled data
			word = utils.filter(data[i][j])
			data[i][j] = word
			if word is not True:
				#for every word in the bagOfWords
				for k in range(0, len(bow)):
					#if the word is the same
					if word == bow[k]:
						#update the features vector
						logger.info("Update vector")
						features[i][k] = features[i][k] + 1
		#what happens if word is false?


	#find the distribution of the BOW
	f = open('/home/cluster/Dropbox/Master/resources/bow.txt', 'w')
	distr = features.sum(axis=0, dtype='int')
	logger.info("The distribution of the Bag of words vector is:%s", distr)
	for d in distr:
		try:
			f.write(str(d))
		except TypeError:
			import traceback
			traceback.print_exc()
			print d
	f.close()

	logger.info("The length of the BOW vector is: %s", len(bow))
	logger.info("The length of the data is: %s", len(data))
	
	#Topic detection using LDA
	import lda
	lda.LDA(data, scipy.sparse.coo_matrix(features), len(data), len(bow), bow)

	#Topic detection using Word2vec (for disambiguation)
	#w2v = word2vecReader.Word2Vec()
	#wname = '/home/cluster/Dropbox/Master/Data/word2vec_twitter.bin'
	#model = w2v.load_word2vec_format(wname, binary=True)

	#model = gensim.models.Word2Vec(data, size=100, window=5, min_count=5, workers=4)

	#tweet = 'hoverboards verboden in nederland'
	#tweet = utils.checkWords(tweet, model)

	#w2vWords = model.most_similar(positive=tweet)
	#logger.info("The most similar words for the tweet are: %s", w2vWords)
	#tags = utils.getTags(w2vWords)
	#logger.info("tags: %s", tags)
	#logger.info("The most similar hashtags for the tweet are: %s", model.most_similar(positive=tags))

	#w2vHashtags = ['#']
	#w2vHashtags = utils.checkWords("".join(w2vHashtags), model)
	#logger.info("The most similar hashtags for the tweet are: %s", model.most_similar(positive=w2vHashtags))

	logger.info("The coordinates for the geolocations are: %s", geolocations)


if __name__ == "__main__":
	main()
