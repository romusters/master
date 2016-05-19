
from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import Word2Vec, HashingTF

loc = '/user/rmusters/text/2015/*/*'

conf = (SparkConf()
    .set("spark.driver.maxResultSize", "0"))

sc = SparkContext(appName='word_count', conf=conf)


text_file = sc.textFile(loc)
counts = text_file.flatMap(lambda line: line.split(" ")) \
			.map(lambda word: (word, 1)) \
			.reduceByKey(lambda a, b: a + b)\
            .map(lambda (a, b): (b, a))\
            .sortByKey(True, 1)\
			.map(lambda (a, b): (b, a))

counts.saveAsTextFile('/user/rmusters/counts')