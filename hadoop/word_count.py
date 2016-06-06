
def main():
	from pyspark import SparkContext, SparkConf
	import filter

	#spark-submit --py-files master/hadoop/stemmer.py,master/hadoop/filter.py --master yarn --executor-memory 12g --deploy-mode cluster --num-executors 400  master/hadoop/word_count.py

	loc = '/user/rmusters/text/2015/01/*'

	#spark-submit --py-files master/hadoop/stemmer.py,master/hadoop/filter.py --master yarn --executor-memory 32g --deploy-mode cluster --num-executors 1000  master/hadoop/word_count.py
	loc = '/user/rmusters/text/2015/*/*'

	conf = (SparkConf()
		.set("spark.driver.maxResultSize", "0"))

	sc = SparkContext(appName='word_count_filtered', conf=conf)

	text_file = sc.textFile(loc)

	threshold = 10
	counts = text_file.map(lambda line: filter.filter(line)) \
				 .flatMap(lambda line: line.split(" ")) \
				 .map(lambda word: (word, 1)) \
				 .reduceByKey(lambda a, b: a + b) \
				 .filter(lambda pair:pair[1] >= threshold)\
				 .sortBy(lambda x:x[1], ascending=True)


	counts.saveAsTextFile('/user/rmusters/counts_taggedUrl_Mention_Stopwords_Punctuation_ignoreNonAscii_StemmedThreshold10_haha_hashtag2015all')
	print counts.count()

	# counts = text_file.flatMap(lambda line: line.split(" ")) \
	# 			.map(lambda word: (word, 1)) \
	# 			.reduceByKey(lambda a, b: a + b)\
	#             .map(lambda (a, b): (b, a))\
	#             .sortByKey(True, 1)\
	# 			.map(lambda (a, b): (b, a))

if __name__ == "__main__":
	main()
