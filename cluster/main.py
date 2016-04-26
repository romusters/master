#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import print_function
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
import logging
from optparse import OptionParser
import sys
from time import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import utils
import math

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def get_data(dataset):
	tweets = []
	length = len(dataset)
	i = 0
	for d in dataset:
		tweet = utils.filterTweet(d)
		print(str(length) + " : " + str(i))
		i = i + 1
		tweets.append(" ".join(tweet))
	return tweets


def cluster(dataset):
	tweets = get_data(dataset)

	print("Extracting features from the training dataset using a sparse vectorizer")
	t0 = time()
	vectorizer = CountVectorizer(min_df=1, lowercase=False)
	X = vectorizer.fit_transform(tweets)
	print("done in %fs" % (time() - t0))
	print("n_samples: %d, n_features: %d" % X.shape)

	###############################################################################
	# Do the actual clustering
	gap_statistic(X)

	sys.exit(0)
	k = 5
	km = KMeans(n_clusters=5, init='k-means++', max_iter=100, n_init=1,
					verbose=opts.verbose)

	print("Clustering sparse data with %s" % km)
	t0 = time()
	km.fit(X)
	print("done in %0.3fs" % (time() - t0))
	print()

	print("Top terms per cluster:")

	order_centroids = km.cluster_centers_.argsort()[:, ::-1]
	print(len(km.cluster_centers_[0]))
	terms = vectorizer.get_feature_names()
	print(terms)

	terms = vectorizer.get_feature_names()

	for i in range(k):
		print("Cluster %d:" % i, end='')
		for ind in order_centroids[i, :4]:
			print(' %s' % terms[ind], end='')
		print()


def iter_kmeans(X, n_clusters, num_iters=5):
	rng =  range(1, num_iters + 1)
	vals = [0]*num_iters
	print(vals)
	for i in rng:
		print(i)
		k = KMeans(n_clusters=n_clusters, n_init=3)
		k.fit(X)
		print("Ref k: %s" % k.get_params()['n_clusters'])
		vals[i-1] = k.inertia_
	print(vals)
	return vals

def gap_statistic(X, max_k=10):
	gaps = range(1, max_k + 1)
	for k in gaps:
		km_act = KMeans(n_clusters=k)#, n_init=3)
		km_act.fit(X)

		# get ref dataset
		#ref = df.apply(get_rand_data)
		refs = iter_kmeans(X, n_clusters=k)
		ref_inertia = np.mean(refs)
		print(ref_inertia)
		print(km_act.inertia_)
		try:
			gap = math.log(ref_inertia - km_act.inertia_)
		except ValueError:
			gap = 0
		print("Ref: %s   Act: %s  Gap: %s" % ( ref_inertia, km_act.inertia_, gap))
		gaps[k] = gap

	return gaps

if __name__ == '__main__':
	fname = '/home/robert/master/cluster/data/test.txt'
	sentences = open(fname, 'r').readlines()
	cluster(sentences)
