from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import Word2Vec, HashingTF
from pyspark.sql import SQLContext
import logging, sys


#spark-submit --py-files master/hadoop/stemmer.py,master/hadoop/filter.py --packages com.databricks:spark-csv_2.10:1.4.0 --master yarn --executor-memory 20g --deploy-mode cluster  master/hadoop/w2v.py
#if in yarn mode, all cores are used

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

loc = '/user/rmusters/text/2015/01/*'
#
# conf = (SparkConf()
#     .set("spark.driver.maxResultSize", "0")\
# 	.set("spark.driver.memory", "50g")\
# 	.set("spark.executor.instances", "10")\
# 	.set("spark.executor.cores", "2")\
# 	.set("spark.rpc.askTimeout", "120000") \
# 	.set("spark.akka.frameSize", "300"))

sc = SparkContext(appName='Lambert')
sqlContext = SQLContext(sc)


def lambert():
	import sys
	max_int_size = sys.maxint
	print "maximum in integer: ", sys.maxint
	data = sqlContext.read.parquet('hdfs:///user/rmusters/data_jan').select("filtered_text")
	print "Amount of tweets: ", data.count()
	print "Show a line: ", data.take(1)

	# counts = data.flatMap(lambda line: line.filtered_text.split(" ")) \
	# 	.map(lambda word: (word, 1)) \
	# 	.reduceByKey(lambda a, b: a + b)


	# Threshold to limit words which occur less than the threshold
	# thresholds = [10, 20, 30, 40]
	# for threshold in thresholds:
	# 	d = counts.filter(lambda pair: pair[1] >= threshold)
	# 	print "vocab_size: ", d.count()
	#
	# vocab_size = counts.count()
	# print "Vocabulary size is: ", vocab_size

	inp = data.map(lambda line: line.filtered_text.split())

	vector_size = 70
	# vocab_size = vector_size / max_int_size
	# print "Vector size is: ", vector_size
	# print "Vocabsize is: ", vocab_size



	word2vec = Word2Vec()
	sample_frac = 0.01
	threshold = 10
	# because per batch, the thresholds are calculated, we need to adjust for sample fraction
	# word2vec.setMinCount(sample_frac * threshold)#40
	word2vec.setVectorSize(vector_size)#/100

	model_name = '/user/rmusters/lambert_jan_2015model'
	# for idx in range(1, 100, 1):
	# 	print idx
	# 	model = word2vec.fit(inp.sample(False, sample_frac))
	# 	# model.save(sc, model_name + str(idx))
	# 	lookup = sqlContext.read.parquet(model_name + str(idx) + '/data').alias("lookup")
	# 	print "vocabsize for iteration " + str(idx) + " : ", lookup.count()

	word2vec.setMinCount(20)
	try:
		model = word2vec.fit(inp)
	except:
		model.save(sc, model_name)
	model.save(sc, model_name)
lambert()
