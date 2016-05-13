
from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import Word2Vec, HashingTF

loc = '/user/rmusters/text/2015/*/*'

conf = (SparkConf()
    .set("spark.driver.maxResultSize", "0"))

sc = SparkContext(appName='Word2Vec', conf=conf)


text_file = sc.textFile(loc)
counts = text_file.flatMap(lambda line: line.split(" ")) \
             .map(lambda word: (word, 1)) \
             .reduceByKey(lambda a, b: a + b)\
             .sortByKey(False) 

counts.saveAsTextFile('/user/rmusters/counts')