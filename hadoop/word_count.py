
from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import Word2Vec, HashingTF


#spark-submit --master yarn --executor-memory 12g --deploy-mode cluster --num-executors 400 master/hadoop/word_count.py


stopwords = ['de', 'en', 'van', 'ik', 'te', 'dat', 'die', 'in', 'een', 'hij', 'het', 'niet', 'zijn', 'is', 'was', 'op', 'aan', 'met', 'als', 'voor', 'had', 'er', 'maar', 'om', 'hem', 'dan', 'zo', 'of', 'wat', 'mijn', 'men', 'dit', 'zo', 'door', 'over', 'ze', 'zich', 'bij', 'ook', 'tot', 'je', 'mij', 'it', 'der', 'daar', 'haar', 'naar', 'heb', 'hoe', 'heeft', 'hebben', 'deze', '', 'want', 'nog', 'zal', 'me', 'zij', 'n', 'ge', 'geen', 'omdat', 'iets', 'worden', 'toch', 'al', 'waren', 'veel', 'meer', 'doen', 'toen', 'moet', 'ben', 'zonder', 'kan', 'hn', 'ds', 'alles', 'onder', 'ja', 'eens', 'hier', 'wie', 'werd', 'altijd', 'doch', 'wordt', 'wezen', 'knnen', 'ons', 'zelf', 'tegen', 'na', 'reeds', 'wil', 'kon', 'niets', 'w', 'iemand', 'geweest', 'andere']

import nltk.stem.snowball
stemmer =  nltk.stem.snowball.DutchStemmer()
def mv_tags(tweet):
	words = tweet.split(" ")
	for i, word in enumerate(words):
		if "http" in word:
			words[i] = "<URL>"
		elif "@" in word:
			words[i] = "<MENTION>"
		elif word in stopwords:
			words[i] = "<STOPWORD>"
		else:
			word[i] = stemmer.stem(word)
	return " ".join(words)


def rm_encoding(s):
	return s.encode('ascii','ignore')

punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~'
def rm_punctuation(s):
	s = s.encode('utf-8')
	s = s.translate(None, punctuation)
	return s

def filter(s):
	s = rm_encoding(s)
	s = rm_punctuation(s)
	s = mv_tags(s)
	return s


loc = '/user/rmusters/text/2015/*/*'

conf = (SparkConf()
    .set("spark.driver.maxResultSize", "0"))

sc = SparkContext(appName='word_count_filtered', conf=conf)


text_file = sc.textFile(loc)

counts = text_file.map(lambda line: filter(line)) \
             .flatMap(lambda line: line.split(" ")) \
             .map(lambda word: (word, 1)) \
             .reduceByKey(lambda a, b: a + b)\
			 .sortBy(lambda x:x[1], ascending=True)



# counts = text_file.flatMap(lambda line: line.split(" ")) \
# 			.map(lambda word: (word, 1)) \
# 			.reduceByKey(lambda a, b: a + b)\
#             .map(lambda (a, b): (b, a))\
#             .sortByKey(True, 1)\
# 			.map(lambda (a, b): (b, a))

counts.saveAsTextFile('/user/rmusters/counts_taggedUrl_Mention_Stopwords_Punctuation_ignoreNonAscii_Stemmed')
print counts.count()