from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import Word2Vec, HashingTF
import logging, sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


loc = '/user/rmusters/text/2010/12/20101231-23.out.gz.txt'
#loc = '/user/rmusters/text/2015/01/*'

conf = (SparkConf()
    .set("spark.driver.maxResultSize", "0"))

sc = SparkContext(appName='Word2Vec', conf=conf)

stopwords = ['de', 'en', 'van', 'ik', 'te', 'dat', 'die', 'in', 'een', 'hij', 'het', 'niet', 'zijn', 'is', 'was', 'op', 'aan', 'met', 'als', 'voor', 'had', 'er', 'maar', 'om', 'hem', 'dan', 'zo', 'of', 'wat', 'mijn', 'men', 'dit', 'zo', 'door', 'over', 'ze', 'zich', 'bij', 'ook', 'tot', 'je', 'mij', 'it', 'der', 'daar', 'haar', 'naar', 'heb', 'hoe', 'heeft', 'hebben', 'deze', '', 'want', 'nog', 'zal', 'me', 'zij', 'n', 'ge', 'geen', 'omdat', 'iets', 'worden', 'toch', 'al', 'waren', 'veel', 'meer', 'doen', 'toen', 'moet', 'ben', 'zonder', 'kan', 'hn', 'ds', 'alles', 'onder', 'ja', 'eens', 'hier', 'wie', 'werd', 'altijd', 'doch', 'wordt', 'wezen', 'knnen', 'ons', 'zelf', 'tegen', 'na', 'reeds', 'wil', 'kon', 'niets', 'w', 'iemand', 'geweest', 'andere']

def mv_tags(tweet):
	words = tweet.split(" ")
	for i, word in enumerate(words):
		if "http" in word:
			words[i] = "<URL>"
		if "@" in word:
			words[i] = "<MENTION>"
		if word in stopwords:
			words[i] = "<STOPWORD>"
	return " ".join(words)


def rm_encoding(s):
	return s.encode('ascii','ignore')


def filter(s):
	s = rm_encoding(s)
	s = mv_tags(s)
	return s

text_file = sc.textFile(loc)
counts = text_file.map(lambda line: filter(line)) \
             .flatMap(lambda line: line.split(" ")) \
             .map(lambda word: (word, 1)) \
             .reduceByKey(lambda a, b: a + b)\
			 .sortBy(lambda x:x[1], ascending=True)
unique_words = counts.count()
print "Unique words are: ", unique_words

inp = sc.textFile(loc).map(lambda line: filter(line).split(" "))\

# inp.cache()

# inp = sc.textFile(loc).map(lambda row: row.split(" "))
# hashingTF = HashingTF()
# tf = hashingTF.transform(inp)
# print tf.count()

vocab_size = unique_words
print "Vocabulary size is: ", vocab_size

max_int_size = 268435455
# vector_size = max_int_size / tf.count()
# print vector_size

word2vec = Word2Vec()
# word2vec.setMinCount(500)

vector_size = max_int_size / vocab_size
print "Vector size is: ", vector_size

word2vec.setVectorSize(vector_size)
model = word2vec.fit(inp)

model.save(sc, '/user/rmusters/pymodel_filtered5')

#model =  word2vec.load(sc, '/user/rmusters/pymodel.bin')

#print model.getVectors()


