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

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
					format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
			  dest="n_components", type="int",
			  help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
			  action="store_false", dest="minibatch", default=True,
			  help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
			  action="store_false", dest="use_idf", default=True,
			  help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
			  action="store_true", default=False,
			  help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
			  help="Maximum number of features (dimensions)"
				   " to extract from text.")
op.add_option("--verbose",
			  action="store_true", dest="verbose", default=False,
			  help="Print progress reports inside k-means algorithm.")

print(__doc__)
op.print_help()

(opts, args) = op.parse_args()
if len(args) > 0:
	op.error("this script takes no arguments.")
	sys.exit(1)


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
	vals = []
	for i in rng:
		k = KMeans(n_clusters=n_clusters, n_init=3)
		k.fit(X)
		print("Ref k: %s" % k.get_params()['n_clusters'])
		vals[i] = k.inertia_
	return vals

def gap_statistic(X, max_k=10):
	gaps = range(1, max_k + 1)
	for k in gaps:
		km_act = KMeans(n_clusters=k, n_init=3)
		km_act.fit(X)

		# get ref dataset
		#ref = df.apply(get_rand_data)
		ref_inertia = iter_kmeans(X, n_clusters=k).mean()

		gap = log(ref_inertia - km_act.inertia_)

		print("Ref: %s   Act: %s  Gap: %s" % ( ref_inertia, km_act.inertia_, gap))
		gaps[k] = gap

	return gaps

if __name__ == '__main__':
	fname = '/home/robert/master/cluster/data/test.txt'
	sentences = open(fname, 'r').readlines()
	cluster(sentences)
