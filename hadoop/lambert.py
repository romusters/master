from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import Word2Vec, HashingTF
from pyspark.sql import SQLContext
import logging, sys


#spark-submit --py-files master/hadoop/stemmer.py,master/hadoop/filter.py --packages com.databricks:spark-csv_2.10:1.4.0 --master yarn --executor-memory 20g --deploy-mode cluster  master/hadoop/lambert.py
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
	# print "Amount of tweets: ", data.count()
	# print "Show a line: ", data.take(1)

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

	# inp = data.map(lambda line: line.filtered_text.split())

	from pyspark.ml.feature import RegexTokenizer
	reTokenizer = RegexTokenizer(inputCol="filtered_text", outputCol="words")
	inp = reTokenizer.transform(data)
	counts = data.flatMap(lambda line: line.filtered_text.split(" ")).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)
	counts.toDF().write.format("com.databricks.spark.csv").mode("overwrite").save("counts.csv")

	inp.select("words").write.parquet("hdfs:///user/rmusters/data_jan_tokenizer", mode="overwrite")

	inp = sqlContext.read.parquet("hdfs:///user/rmusters/data_jan_tokenizer")
	inp = inp.map(lambda line: line[0])
	#
	# import re
	# def split_number_string(tokens):
	# 	res = []
	# 	for token in tokens:
	# 		token = token.strip()
	# 		if token[0:2] == "06" and len(token) == 10:
	# 			res.append("<MOBIEL>")
	# 		else:
	# 			parts = re.split('(\d+)',token)#match(r"([a-z]+)([0-9]+)", token, re.I)
	# 			for part in parts:
	# 				if part != '':
	# 					res.append(part)
	# 				# if parts is None:
	# 			# 	res.append(token)
	# 			# else:
	# 			# 	res.extend(list(parts.groups()))
	# 	return res
	#
	# inp = inp.map(lambda x: split_number_string(x[0]))

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

	model_name = '/user/rmusters/lambert_jan_2015model_tokenizer'
	# for idx in range(1, 100, 1):
	# 	print idx
	# 	model = word2vec.fit(inp.sample(False, sample_frac))
	# 	# model.save(sc, model_name + str(idx))
	# 	lookup = sqlContext.read.parquet(model_name + str(idx) + '/data').alias("lookup")
	# 	print "vocabsize for iteration " + str(idx) + " : ", lookup.count()

	word2vec.setMinCount(20)
	model = word2vec.fit(inp)
	model.save(sc, model_name)



def save_vectors(path):
	vectors = sqlContext.read.parquet(path + '/data')
	vectors.save(path + ".csv", "com.databricks.spark.csv")

def save_correct_text():
	inp = sqlContext.read.parquet("hdfs:///user/rmusters/data_jan_lowercase")

	import re
	def split_number_string(tokens):
		res = []
		for token in tokens:
			token = token.strip()
			if token[0:2] == "06" and len(token) == 10:
				res.append("<mobiel>")
			else:
				parts = re.split('(\d+)', token)  # match(r"([a-z]+)([0-9]+)", token, re.I)
				for part in parts:
					if part != '':
						res.append(part)
					# if parts is None:
					# 	res.append(token)
					# else:
					# 	res.extend(list(parts.groups()))
		return res

	inp = inp.map(lambda x: (x,split_number_string(x[0])))
	# geen id :( inp.toDF([""]).write.parquet("hdfs:///user/rmusters/data_jan_filtered")

# save_correct_text()
# lambert()
save_vectors("/user/rmusters/lambert_jan_2015model_tokenizer")
