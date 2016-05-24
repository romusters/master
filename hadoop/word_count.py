
from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import Word2Vec, HashingTF


#spark-submit --master yarn --executor-memory 12g --deploy-mode cluster --num-executors 400 master/hadoop/word_count.py

def http(tweet):
	words = tweet.split(" ")
	for i, word in enumerate(words):
		if "http" in word:
			words[i] = "URL"
	return " ".join(words)


loc = '/user/rmusters/text/2015/*/*'

conf = (SparkConf()
    .set("spark.driver.maxResultSize", "0"))

sc = SparkContext(appName='word_count', conf=conf)


text_file = sc.textFile(loc)

counts = text_file.map(lambda line: http(line)) \
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

counts.saveAsTextFile('/user/rmusters/counts_no_url')
print counts.count()