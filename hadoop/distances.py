from pyspark import SparkContext, SparkConf, SQLContext
import logging, sys
import numpy as np

# spark-submit --py-files master/hadoop/stemmer.py,master/hadoop/filter.py --master yarn --deploy-mode cluster  master/hadoop/distances.py

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def load_model():
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	lookup = sqlContext.read.parquet('/user/rmusters/2015model99/data').alias("lookup")
	lookup_bd = sc.broadcast(lookup.rdd.collectAsMap())
	return lookup_bd

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


def load_data(path):
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	data = sqlContext.read.parquet(path)
	return data


def similarity(tweet, other_tweet, model):
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


conf = SparkConf()\
    .set("spark.driver.maxResultSize", "0")\
	.set("spark.driver.memory", "12g")\
	.set("spark.executor.memory", "12g")\
	.set("spark.executor.instances", "200")

sc = SparkContext(appName='distances', conf=conf)

path = 'hdfs:///user/rmusters/data'
write_data(path)
data = load_data(path)
model = load_model()

tweet = data.take(1)[0].filtered_text
data_rdd = data.rdd.map(lambda (text, filtered_text, vectors, id): (text, filtered_text, vectors, id, similarity(tweet, text, model)))
path =  'hdfs:///user/rmusters/sims'
df = data_rdd.toDF(["text", "filtered_text", "vectors", "id", "sims"])
df.write.parquet(path, mode="overwrite")


#vectors = data.select("vectors").collect()





# assume that tweets with the same topic, have words with the same semantic meaning at the same location
# take a tweet and calculate the distance to all the other tweets. Be sure to take the tweet with the most words and compare each word.

# res = vectors.take(2)
#
# vectors = data.map(lambda line: model.transform(line[0]))
# model.transform("ik")

# data = sc.textFile(loc).map(lambda line: line.split(",", 1))\
# 	.map(lambda line: (line[0], eval(line[1].replace('WrappedArray', "").replace("(", "[").replace(")", "]"))))




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
base = data.take(1)
# calculate the distance to all other tweets
data = data.map(lambda line: abs(np.sum(np.subtract(np.asarray(eval(line[1])), np.asarray(eval(base[0][1]))))))
path =  'hdfs:///user/rmusters/distances.txt'
data.saveAsTextFile(path)

