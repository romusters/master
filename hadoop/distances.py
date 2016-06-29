from pyspark import SparkContext, SparkConf, SQLContext
import logging, sys
import numpy as np

# spark-submit --packages com.databricks:spark-csv_2.10:1.4.0 --py-files master/hadoop/stemmer.py,master/hadoop/filter.py --master yarn --deploy-mode cluster  master/hadoop/distances.py

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

conf = SparkConf()\
    .set("spark.driver.maxResultSize", "0")\
	.set("spark.driver.memory", "12g")\
	.set("spark.executor.memory", "12g")\
	.set("spark.executor.instances", "400")

sc = SparkContext(appName='distances', conf=conf)


def write_data(path):
	import filter
	from pyspark.mllib.feature import Word2Vec, Word2VecModel

	# load data
	loc = '/user/rmusters/text/2015/01/*'
	text_file = sc.textFile(loc)
	data = text_file.map(lambda line: filter.filter(line).split(" "))

	# load model
	word2vec = Word2Vec()
	model = Word2VecModel.load(sc, '/user/rmusters/2015model99')

	# get a tweet vector pair.
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	lookup = sqlContext.read.parquet('/user/rmusters/2015model99/data').alias("lookup")
	lookup_bd = sc.broadcast(lookup.rdd.collectAsMap())

	vectors = data.map(lambda ws: [lookup_bd.value.get(w) for w in ws])
	logger.info(vectors.count())

	data = text_file.map(lambda line: (line, filter.filter(line).split(" ")))\
							.map(lambda (text, filtered): (text, filtered, [lookup_bd.value.get(w) for w in filtered][0]))

	from pyspark.sql.functions import monotonicallyIncreasingId
	df = data.toDF(["text", "filtered_text", "vectors"])
	# This will return a new DF with all the columns + id
	res = df.withColumn("id", monotonicallyIncreasingId())
	res.write.parquet(path, mode="overwrite")

def load_model():
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	lookup = sqlContext.read.parquet('/user/rmusters/2015model99/data').alias("lookup")
	lookup_bd = sc.broadcast(lookup.rdd.collectAsMap())
	return lookup_bd

def load_data(path):
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	data = sqlContext.read.parquet(path)
	return data

def normal_similarity(vectors, vector):
	import numpy as np
	try:
		#HAS abs() to be inside?
		return float(abs(np.sum(np.subtract(np.asarray(vectors), np.asarray(vector)))))
	except TypeError:
		return None
	except Exception as e:
		logging.info(e)


def cos_similarity(tweet, other_tweet, model):
	import numpy
	sims = []
	for word in tweet:
		tmp_cos_sim = 0
		try:
			word = model.value.get(word)
		except TypeError:
			continue
		for other_word in other_tweet:
			other_word = model.value.get(other_word)
			#cos_sim = numpy.dot(model.transform(word), model.transform(other_word)) / (numpy.linalg.norm(model.transform(word)) * numpy.linalg.norm(model.transform(other_word)))
			try:
				cos_sim = numpy.dot(word, other_word) / (numpy.linalg.norm(word) * numpy.linalg.norm(other_word))
			except TypeError:
				continue
			if cos_sim > tmp_cos_sim:
				tmp_cos_sim = cos_sim
		sims.append(tmp_cos_sim)
	return float(sum(sims)/len(sims))



def main():
	path = 'hdfs:///user/rmusters/data'
	#write_data(path)

	#load filtered tweets
	data = load_data(path)

	#take a sample, because else it takes to long
	data = data.sample(False, 0.01)


	#load w2v model
	model = load_model()

	tweet = data.take(1)[0].filtered_text
	#calculate distances from a tweet to the others.
	data_rdd = data.rdd.map(lambda (text, filtered_text, vectors, id): (text, filtered_text, vectors, id, cos_similarity(tweet, text, model)))

	#
	logging.info(data_rdd.count())

	cluster = data_rdd.filter(lambda (text, filtered_text, vectors, id, similarity): similarity > 0.6)
	logging.info(cluster.count())

def write_sims():
	path =  'hdfs:///user/rmusters/sims'
	df = data_rdd.toDF(["text", "filtered_text", "vectors", "id", "sims"])
	df.write.parquet(path, mode="overwrite")

def average_vector(data):
	from pyspark.sql.functions import col
	vectors = data.select("vectors").where(col("vectors").isNotNull())

	from pyspark.mllib.linalg import Vectors
	vectors_v = vectors.map(lambda line: Vectors.dense(line))

	from pyspark.mllib.stat import Statistics
	summary = Statistics.colStats(vectors_v)
	mean = summary.mean()
	logger.info(mean)
	return mean

def find_similar_tweet_from_vector():
	from pyspark import SQLContext
	sqlContext = SQLContext(sc)

	path = 'hdfs:///user/rmusters/data'

	#load filtered tweets
	data = sqlContext.read.parquet(path)

	#take a sample, because else it takes to long
	data = data.sample(False, 0.01)
	data.persist()

	vector = average_vector(data)

	#calculate distances from a tweet to the others.
	data_rdd = data.rdd.map(lambda (text, filtered_text, vectors, id): (text, filtered_text, vectors, id, normal_similarity(vectors, vector)))

	df = data_rdd.toDF(["text", "filtered_text", "vectors", "id", "sims"])
	df = df.sort(df.sims.asc()) #is one column sorted or all columns

	from pyspark.sql.functions import col
	df = df.where(col("sims").isNotNull())
	df.sort(df.sims.desc())
	#df = df.select("sims")
	path = 'hdfs:///user/rmusters/average_tweet_distances'
	df.write.parquet(path, mode="overwrite")

def sort():
	path =  'hdfs:///user/rmusters/sims'
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	data = sqlContext.read.parquet(path)
	data_sample = data.sample(False, 0.01)
	data_sample_sort = data_sample.sort(data_sample.sims.desc())
	df = data_sample_sort.toDF("text", "filtered_text", "vectors", "id", "sims")
	path = "hdfs:///user/rmusters/data_sample_sort"
	df.write.parquet(path, mode="overwrite")

def load_sort():
	#pyspark --packages com.databricks:spark-csv_2.10:1.4.0 --num-executors 10
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	dist = sqlContext.read.parquet('/user/rmusters/data_sample_sort')
	dist = dist.select("sims")
	return dist
	dist.save("data_sample_sort.csv", "com.databricks.spark.csv", "overwrite")
	#hdfs dfs -getmerge '/user/rmusters/data_sample_sort.csv' dddd
	#res = dist.filter(dist.sims > 0.28).filter(dist.sims < 0.29)

def save_dist(df_path, fname):
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	df = sqlContext.read.parquet(df_path)
	df.save(fname, "com.databricks.spark.csv", "overwrite")


#vectors = data.select("vectors").collect()





# assume that tweets with the same topic, have words with the same semantic meaning at the same location
# take a tweet and calculate the distance to all the other tweets. Be sure to take the tweet with the most words and compare each word.



# Sort of learning vector quantization!
# present a tweet to the user which is not that far from the original tweet.
	# if the tweet is correct
		# construct a prototype
				# repeat until convergence
	# if the tweet is not correct, present a tweet which is less far from the original tweet
		# repeat


###########################
#OLD distance function
# take a tweet
# base = data.take(1)
# # calculate the distance to all other tweets
# data = data.map(lambda line: abs(np.sum(np.subtract(np.asarray(eval(line[1])), np.asarray(eval(base[0][1]))))))
# path =  'hdfs:///user/rmusters/distances.txt'
# data.saveAsTextFile(path)

if __name__ == "__main__":
	#main()
	#sort()

	#determine the average vector and calculate distances from it
	find_similar_tweet_from_vector()

	save_dist("hdfs:///user/rmusters/average_tweet_distances", "normal_distance.csv")
