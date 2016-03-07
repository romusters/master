#NOT USED
import gensim
import word2vecReader
import sys
import logging


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
#main.logger.setLevel(logging.INFO)
logger = logging.getLogger(__name__)

fname = ""
class Word2vec:
	fname = None
	def __init__(self, fname):
		self.fname = fname

	def loadModel(self, fname):
		try:
			return gensim.models.Word2Vec.load(fname)
		except RuntimeError:
			logger.error("Word2vec model could not be loaded, location was: %s", fname)
			raise
		#return model = gensim.models.Word2Vec.load_word2vec_format("word2vec_twitter.bin", binary=True)

	def trainModel(self, sentences):
		try:
			model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
		except:
			logger.error("W2v model could not be trained.")
			import traceback
			traceback.print_exc(file=sys.stdout)
		return model

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
	w2v = word2vecReader.Word2Vec()
	data = None
	try:
		data = pickle.load(open( "data.p", "rb" ))
	except IOError:
		#perhaps check the whole json object for the value? So -text?
		logger.info("No data available, start training...")
		#or open data from the tweet-stream
		data = utils.getSpecificTextData('/media/robert/dataThesis/tweets/Tekst/2010/12', value)

		pickle.dump( data, open( "data.p", "wb" ) )
	model = w2v.load_word2vec_format("word2vec_twitter.bin", binary=True)

if __name__ == "__main__":
	main()
