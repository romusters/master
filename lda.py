from sklearn.decomposition import NMF, LatentDirichletAllocation
from time import time
import sys
import logging
import utils

logging.basicConfig(stream=sys.stdout, level=logging.ERROR, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

peregrine = False

def LDA(data, tf, n_samples, n_features, bow):
	logger.info("Start LDA")
	n_topics = 5
	n_top_words = 309

	logger.info("Fitting LDA models with tf features, n_samples=%d and n_features=%d..."
		  % (n_samples, n_features))
	lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
									learning_method='online', learning_offset=50.,
									random_state=0)
	t0 = time()
	lda.fit(tf)
	logger.info("done in %0.3fs." % (time() - t0))

	logger.info("\nTopics in LDA model:")
	print_top_words(lda, bow, n_top_words)

def print_top_words(model, feature_names, n_top_words):
	for topic_idx, topic in enumerate(model.components_):
		print("Topic #%d:" % topic_idx)
		print(" ".join([feature_names[i]
						for i in topic.argsort()[:-n_top_words - 1:-1]]))
	print()

def main():
	#Load Twitter data
	if peregrine:
		tweets = utils.loadAllData('/data/s1774395/', filter=True, save=True)
	else:
		tweets = utils.loadAllData('/home/robert/data/2015123101/', filter=True, save=True)


	logger.debug("Number of tweets: %i", len(tweets))

	#get bag of words
	bow = list(utils.getBagOfwords(tweets))

	#get the features
	import numpy as np
	features = np.zeros((len(tweets), len(bow)))
	#for every tweet
	for i in range(0, len(tweets)):
		#for every word in a tweet
		for j in range(0, len(tweets[i])):
			#Stem data, normalize, remove duplicates and only use geo-enabled data
			word = utils.filter(tweets[i][j])
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


	# #find the distribution of the BOW
	# f = open('/home/cluster/Dropbox/Master/resources/bow.txt', 'w')
	# distr = features.sum(axis=0, dtype='int')
	# logger.info("The distribution of the Bag of words vector is:%s", distr)
	# for d in distr:
	# 	try:
	# 		f.write(str(d))
	# 	except TypeError:
	# 		import traceback
	# 		traceback.print_exc()
	# 		print d
	# f.close()

	#Train for Topic detection using LDA
	import lda
	lda.LDA(data, scipy.sparse.coo_matrix(features), len(data), len(bow), bow)

def lda():
	#load data
	import csv
	path = "/media/cluster/dataThesis/data_sample.csv"
	filtered_text = []
	vectors = []
	with open(path, 'rb') as csvfile:
		rows = csv.reader(csvfile, delimiter=',')
		for row in rows:
			dict[row[0]] = eval(row[1].replace('WrappedArray', "").replace("(", "[").replace(")", "]"))
	print dict.keys()

if __name__ == "__main__":
	#main()
	lda()