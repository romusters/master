import utils
import word2vec
import logging
import sys
import gensim

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def main():
	dname = '/data/s1774395/201201.tar.gz'
	sentences = gensim.models.word2vec.LineSentence(dname, max_sentence_length=150)
	model = gensim.models.word2vec.Word2Vec(sentences, size=100, window=5, min_count=5, workers=24)
	fname = "/data/s1774395/gensimPeregrineModel.bin"
	model.save(fname)
	print model.most_similar(positive=['word'], negative=['ik'])


if __name__ == "__main__":
	main()

