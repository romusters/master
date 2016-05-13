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
import matplotlib.pyplot as plt

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
	n_clusters = 90
	n_clusters_step = 5
	n_clusters_mean = 10
	ks = range(1, n_clusters, n_clusters_step)
	indices = range(1, n_clusters_mean, 1)
	errors = []
	ref_errors = []
	gaps = []
	for k in ks:
		km = KMeans(n_clusters=k, n_jobs=4)
		km.fit(X)
		km_inert = km.inertia_
		print(km_inert)
		errors.append(km.inertia_)

		ref = []
		for i in indices:
			km = KMeans(n_clusters=k, n_jobs=4)
			km.fit(X)
			ref.append(km.inertia_)
		ref_mean = np.mean(ref)
		print(ref_mean)
		ref_errors.append(ref_mean)
		try:
			gap = math.log(ref_mean - km_inert)
			gaps.append(gap)
			print("Gap: ", gap)

		except:
			gaps.append(0)
			continue
	print(gaps)
	plt.plot(gaps)
	plt.plot(ks, errors)
	plt.plot(ks, ref_errors)
	plt.show()
	sys.exit(0)

	gap_statistic(X)


	k = 5
	km = KMeans(n_clusters=5, init='k-means++', max_iter=100, n_init=1, verbose=opts.verbose, n_jobs=4)

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





if __name__ == '__main__':
	fname = '/home/robert/master/cluster/data/20151201:00.out.gz.txt'
	sentences = open(fname, 'r').readlines()
	cluster(sentences)
