import utils
import word2vec
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

peregrine = False

def main():
    fname = '20151225to31.bin'
    if peregrine:
        tweets = utils.loadAllData('/data/s1774395/', filter=True, save=True)
    else:
        tweets = utils.loadAllData('/home/robert/data/20151225to31/', filter=True, save=True)

    w2v = word2vec.Word2vec(fname)

    word2vecModel = w2v.trainModel(tweets)

    if peregrine:
        w2v.saveModel(word2vecModel, '/data/s1774395/w2vModels/' + fname)
    else:
        w2v.saveModel(word2vecModel, '/home/robert/data/w2vModels/' + fname)

if __name__ == "__main__":
	main()

