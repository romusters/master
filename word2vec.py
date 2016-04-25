#!/usr/bin/env python
# -*- coding: utf-8 -*-
#


#NOT USED
import gensim
import word2vecReader
import sys
import logging


#sys.setdefaultencoding()

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
#main.logger.setLevel(logging.INFO)
logger = logging.getLogger(__name__)

fname = ""
class Word2vec:
	fname = None
	def __init__(self, fname=None):
		self.fname = fname

	def loadModel(self, fname):
		try:
			return gensim.models.Word2Vec.load(fname)
		except RuntimeError:
			logger.error("Word2vec model could not be loaded, location was: %s", fname)
			raise
		#return model = gensim.models.Word2Vec.load_word2vec_format("word2vec_twitter.bin", binary=True)

	def trainModel(self, sentences, peregrine):
		try:
			if peregrine:
				import os
				# poolSize = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
				model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5)#, workers=poolSize)
				return model
			else:
				model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
				return model
		except:
			logger.error("W2v model could not be trained.")
			import traceback
			traceback.print_exc(file=sys.stdout)


	def saveModel(self, model, fname):
		try:
			model.save(fname)
		except:
			logger.error("W2v model could not saved.")
			import traceback
			traceback.print_exc(file=sys.stdout)

	def getSimilar(self, word, model):
		return model.most_similar(positive=word)

def main():
	# utils.encodeTextToUTF8(dname)
	dname = '/home/robert/data/201201.tar.gz'
	sentences = gensim.models.word2vec.LineSentence(dname, max_sentence_length=150)
	model = gensim.models.word2vec.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
	fname = "/home/robert/data/gensimLocalModel.bin"
	print model.most_similar(positive=['word'], negative=['ik'])
	model.save(fname)
	# w2v = Word2vec()
	# model = w2v.trainModel(sentences, False)
	#
	# mname = "/home/robert/data/gensimLocalModel.bin"
	# w2v.saveModel(model, mname)



	# w2v = word2vecReader.Word2Vec()
	#
	#
	# data = None
	# try:
	# 	data = pickle.load(open( "data.p", "rb" ))
	# except IOError:
	# 	#perhaps check the whole json object for the value? So -text?
	# 	logger.info("No data available, start training...")
	# 	#or open data from the tweet-stream
	# 	data = utils.getSpecificTextData('/media/robert/dataThesis/tweets/Tekst/2010/12', value)
	#
	# 	pickle.dump( data, open( "data.p", "wb" ) )
	# model = w2v.load_word2vec_format("word2vec_twitter.bin", binary=True)

def testModel():
	import utils
	tweet = 'ik heb geen zin meer in Twitter'.split()
	#tweet = tweet.encode('utf-8')
	#tweet = utils.filterTweet(tweet)
	#model = Word2vec.loadModel('/home/robert/data/word2vecModels/20151231.bin')

	import gensim
	model = gensim.models.Word2Vec.load('/media/robert/dataThesis/w2vModels/gensimLocalModel.bin')

	similar = model.most_similar(tweet)

	logger.info('Most similar words for: %s are %s', tweet, similar)


if __name__ == "__main__":
	#main()
	testModel()
