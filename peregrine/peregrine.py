
import logging
import sys
import gensim

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class DirOfPlainTextCorpus(object):
	def __init__(self, fname):
		self.fname = fname

	def __iter__(self):
		text = open(self.fname).read()
		for sentence in text:
			yield sentence.split()


def train_model():
		dname = '/data/s1774395/text/input.txt'
		logger.info("Loading data")

		#Method 1
		#sentences = gensim.models.word2vec.LineSentence(dname)

		#Method 2
		#sentences = [d.split() for d in open(dname).readlines()]

		#Method 3

		logger.info("Training model")
		try:
			#model = gensim.models.word2vec.Word2Vec(sentences, size=100000, window=5, min_count=5, workers=24)
			model = gensim.models.word2vec.Word2Vec(DirOfPlainTextCorpus(dname), size=100000, window=5, min_count=5, workers=24)

		except UnicodeDecodeError:
			import traceback
			traceback.print_exc()

		logger.info("Saving model")
		fname = "/data/s1774395/fullGensimPeregrineModel.bin"
		model.save(fname)


def test_model():
	mname = '/data/s1774395/8GgensimPeregrineModel.bin'
	model = gensim.models.Word2Vec.load(mname)
	tweets = ['ik heb geen zin meer in Twitter', '@celine_maas ja joh mn zus kijkt het :(', '@RiannePathuis ik ook niet :P']
	for tweet in tweets:
		print model.most_similar(tweet.split())

if __name__ == "__main__":
		train_model()


