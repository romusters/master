from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.sql import SQLContext
import logging, sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

#spark-submit --master yarn --deploy-mode cluster master/hadoop/lda.py

conf = (SparkConf()\
		.set("spark.driver.maxResultSize", "0")\
		.set("spark.driver.memory", "50g")\
		.set("spark.executor.memory", "50g")\
		.set("spark.executor.instances", "400")\
		.set("spark.rpc.askTimeout", "120000"))

# Larger file
loc = '/user/rmusters/text/2015/*/*'
loc = '/user/rmusters/text/2015/01/*'

sc = SparkContext(appName='lda_train', conf=conf)

sqlContext = SQLContext(sc)



def train_model():
	model_vectors = sqlContext.read.parquet('/user/rmusters/threshold20_2015model56/data')
	logger.info("model loaded")

	rdd_words = model_vectors.map(lambda line: line[0])
	# words = rdd_words.collect() #15919
	logger.info("words collected: ", int(rdd_words.count()))

	#bag of words is used to train LDA
	path = 'hdfs:///user/rmusters/bow_data'
	data = load_data(path)
	# logger.info("data loaded")
	# data = data.sample(False, 0.01)
	# logger.info("data sampled")

	#check if sum of vector is zero 13 times. This indicates the datasample does not contain certain words and thus the sparse vector removes them

	corpus = data.map(lambda (text, filtered_text, vector, id): [id, vector])

	logger.info("Training the lda model")
	ldaModel = LDA.train(corpus, k=100)
	logger.info("Vocab size is: ", ldaModel.vocabSize())

	ldaModel.save(sc, 'hdfs:///user/rmusters/ldaModel')


def load_data(path):
		sqlContext = SQLContext(sc)
		data = sqlContext.read.parquet(path)
		return data


def load_model():
	from pyspark.mllib.clustering import LDA, LDAModel
	#sc = SparkContext(appName='lda_load', conf=conf)
	path = "/user/rmusters/ldaModel2"
	ldaModel = LDAModel.load(sc, path)
	return ldaModel

def predict_cluster(tweet, topics, topicsMatrix, word_dict):
	import numpy as np
	topics_hits = [0]*len(topics)
	topics_hits_scores = [0]*len(topics)
	for w in tweet:
		try:
			topic_scores = topicsMatrix[word_dict[w]]
		except:
			continue
		max_topic_score = max(topicsMatrix[word_dict[w]])
		max_topic_index = np.argmax(topic_scores)
		topics_hits[max_topic_index] += 1
		topics_hits_scores[max_topic_index] = max_topic_score
	best_cluster = max(topics_hits)
	if best_cluster == 1:
		return int(np.argmax(topics_hits_scores))
	else:
		return int(np.argmax(topics_hits))


def predict_old():
	# probabilities for each word are extraced from each topic. The probabilities for all the words per cluster are summed.
	# the cluster with the heighest sum wins.
	ldaModel = load_model()
	path = 'hdfs:///user/rmusters/data_sample'
	data = load_data(path)
	data = data.drop("vectors")
	topics = ldaModel.describeTopics() #every topics is a vector with sorted probabilities and their indices.
	topicsMatrix = ldaModel.topicsMatrix()
	#make a dictionary containing the word and their index
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	model_vectors = sqlContext.read.parquet('/user/rmusters/2015model99/data')
	#logger.info("model loaded")
	rdd_words = model_vectors.map(lambda line: line[0])
	words = rdd_words.collect()  # 15919
	word_dict = {}
	for i, w in enumerate(words):
		word_dict[w] = i
	data_cluster = data.map(lambda (text, filtered_text, id): (text, filtered_text, id, predict_cluster(filtered_text, topics, topicsMatrix, word_dict)))
	df = data_cluster.toDF(["text", "filtered_text", "id", "cluster"])
	path = 'hdfs:///user/rmusters/lda_cluster_data'
	df.write.parquet(path, mode="overwrite")

def predict():
	path = 'hdfs:///user/rmusters/data_sample'
	data = load_data(path)
	data = data.drop("vectors")

	ldaModel = load_model()
	topics = ldaModel.describeTopics()  # every topics is a vector with sorted probabilities and their indices.
	topicsMatrix = ldaModel.topicsMatrix()

	# make a dictionary containing the word and their index
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	model_vectors = sqlContext.read.parquet('/user/rmusters/2015model99/data')
	# logger.info("model loaded")
	rdd_words = model_vectors.map(lambda line: line[0])
	words = rdd_words.collect()  # 15919
	word_dict = {}
	for i, w in enumerate(words):
		word_dict[w] = i

	data_cluster = data.map(lambda (text, filtered_text, id): (text, filtered_text, id, predict_cluster(filtered_text, topics, topicsMatrix, word_dict)))
	df = data_cluster.toDF(["text", "filtered_text", "id", "cluster"])
	path = 'hdfs:///user/rmusters/lda_cluster_data'
	df.write.parquet(path, mode="overwrite")


def save_lda_data_csv():
	path = 'hdfs:///user/rmusters/lda_cluster_data'
	data = load_data(path)
	data = data.sort(data.cluster.asc())
	data.save("lda_cluster_data.csv", "com.databricks.spark.csv", "overwrite")


if __name__ == "__main__":
	import logging, sys
	train_model()
	#model = load_model()
	#logger.info(model.vocabSize())
	# predict()
	#save_lda_data_csv()