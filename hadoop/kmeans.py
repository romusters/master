from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark import SparkContext, SparkConf
from math import sqrt
import logging, sys

#spark-submit --py-files master/hadoop/lda.py --packages com.databricks:spark-csv_2.10:1.4.0 --master yarn --deploy-mode cluster --driver-memory 50g master/hadoop/kmeans.py

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


conf = (SparkConf()
    .set("spark.driver.maxResultSize", "0")\
	.set("spark.driver.memory", "50g")\
	.set("spark.executor.memory", "10g") \
	.set("spark.executor.cores", "2") \
	.set("spark.executor.instances", "400")\
	.set("spark.rpc.askTimeout", "120000"))

def kmeans_w2v():
	sc = SparkContext(appName='kmeans_w2v', conf=conf)
	from pyspark.sql import SQLContext

	sqlContext = SQLContext(sc)

	df_path = "hdfs:///user/rmusters/w2v_data"
	df = sqlContext.read.parquet(df_path)
	data = df.select("mean_vector")
	print data.take(1)
	from numpy import array
	parsedData = data.map(lambda line: line[0]).filter(lambda line: line is not None)
	print parsedData.take(1)

	for n_clusters in range(500,700,20):
		# Build the model (cluster the data)
		clusters = KMeans.train(parsedData, n_clusters, maxIterations=10, runs=10, initializationMode="random")

		# Evaluate clustering by computing Within Set Sum of Squared Errors
		def error(point):
			center = clusters.centers[clusters.predict(point)]
			return sqrt(sum([x**2 for x in (point - center)]))

		WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
		logger.info("Within Set Sum of Squared Error = " + str(n_clusters) + "&" +  str(WSSSE) + "\\")

	# Save and load model
	clusters.save(sc, "hdfs:///user/rmusters/kmeans_w2v")
	#sameModel = KMeansModel.load(sc, "myModelPath")

def kmeans_bow(path):
	sc = SparkContext(appName='kmeans_bow', conf=conf)
	from pyspark.sql import SQLContext

	sqlContext = SQLContext(sc)
	data = sqlContext.read.parquet(path + "bow_data").select("vectors")

	from numpy import array
	data = data.map(lambda line: line[0])
	logging.info(data.take(1))
	logging.info(len(data.take(1)))


	clusters = None
	for n_clusters in range(400,500,20):
		# Build the model (cluster the data)
		clusters = KMeans.train(data, n_clusters, maxIterations=10, runs=10, initializationMode="random")

		# Evaluate clustering by computing Within Set Sum of Squared Errors
		def error(point):
			center = clusters.centers[clusters.predict(point)]
			# http://stackoverflow.com/questions/32977641/index-out-of-range-in-spark-mllib-k-means-with-tfidf-for-text-clutsering
			return sqrt(sum([x**2 for x in (point.toArray() - center)]))

		WSSSE = data.map(lambda point: error(point)).reduce(lambda x, y: x + y)
		logger.info("Within Set Sum of Squared Error = " + str(n_clusters) + "&" +  str(WSSSE) + "\\")

	# Save and load model
	clusters.save(sc, path + "bow_data")
	#sameModel = KMeansModel.load(sc, "myModelPath")


def kmeans_lda():
	from pyspark.mllib.clustering import KMeans, KMeansModel
	from pyspark import SparkContext, SparkConf
	from pyspark.sql import SQLContext
	sc = SparkContext(appName='kmeans_lda', conf=conf)
	sqlContext = SQLContext(sc)
	path = "/user/rmusters/lda_doc_topic"
	data = sqlContext.read.parquet(path)
	data = data.select("_2")
	data = data.map(lambda line: line[0])
	#data.write.format('com.databricks.spark.csv').save('lda_doc_topic.csv')
	clusters = None
	for n_clusters in range(560, 700, 20):
		# Build the model (cluster the data)
		clusters = KMeans.train(data, n_clusters, maxIterations=10, runs=10, initializationMode="random")

		# Evaluate clustering by computing Within Set Sum of Squared Errors
		def error(point):
			center = clusters.centers[clusters.predict(point)]
			# http://stackoverflow.com/questions/32977641/index-out-of-range-in-spark-mllib-k-means-with-tfidf-for-text-clutsering
			return sqrt(sum([x ** 2 for x in (point.toArray() - center)]))

		WSSSE = data.map(lambda point: error(point)).reduce(lambda x, y: x + y)
		logger.info(str(n_clusters) + "&" + str(WSSSE))

	# Save and load model
		clusters.save(sc, "/user/rmusters/kmeans_lda")

def kmeans_lda_predict():
	from pyspark.mllib.clustering import KMeans, KMeansModel
	sc = SparkContext(appName='kmeans_lda_predict', conf=conf)
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	data = sqlContext.read.parquet("hdfs:///user/rmusters/lda_data")
	# data = df.rdd
	model = KMeansModel.load(sc, "hdfs:///user/rmusters/kmeans_lda")
	data = data.map(lambda (text, filtered_text, vectors, id): (text, filtered_text, vectors, model.predict(vectors), id))
	df = data.toDF(["text", "filtered_text", "vectors", "cluster", "id"])
	df.write.parquet("hdfs:///user/rmusters/lda_data_cluster", mode= "overwrite")


def w2v_predict(vector, model):
	if vector == None:
		return None
	else:
		return model.predict(vector)


def kmeans_w2v_predict():
	sc = SparkContext(appName='kmeans_w2v_predict', conf=conf)
	from pyspark.sql import SQLContext
	from pyspark.mllib.clustering import KMeans, KMeansModel
	sqlContext = SQLContext(sc)
	data = sqlContext.read.parquet("hdfs:///user/rmusters/w2v_data")
	df = data.toDF("text", "filtered_text", "mean_vector", "id")
	df = df.where(df.mean_vector.isNotNull())
	data = df.rdd
	# def none(x):
	# 	if type(x) != NoneType:
	# 		return True
	# 	else:
	# 		return False
	#
	# data = data.map(lambda (text, filtered_text, mean_vector, id): (text, filtered_text, str(mean_vector), id))
	# data = data.filter(lambda (text, filtered_text, mean_vector, id): none(mean_vector))

	model = KMeansModel.load(sc, "hdfs:///user/rmusters/kmeans_w2v")
	#model.predict(eval(
	data = data.map(lambda (text, filtered_text, mean_vector, id): (text, filtered_text, mean_vector, model.predict(mean_vector), id))
	df = data.toDF(["text", "filtered_text", "mean_vector", "cluster", "id"])
	df.write.parquet("hdfs:///user/rmusters/w2v_data_cluster", mode= "overwrite")


# def kmeans_lda_predict(tweet, topics, topicsMatrix, word_dict):
# 	sc = SparkContext(appName='kmeans_lda_predict', conf=conf)
# 	from pyspark.sql import SQLContext
# 	from pyspark.mllib.clustering import LDA, LDAModel
# 	sqlContext = SQLContext(sc)
# 	path = 'hdfs:///user/rmusters/lda_data'
# 	data = sqlContext.read.parquet(path)
#
# 	path = "/user/rmusters/ldaModel2"
# 	ldaModel = LDAModel.load(sc, path)
# 	topics = ldaModel.describeTopics()  # every topics is a vector with sorted probabilities and their indices.
# 	topicsMatrix = ldaModel.topicsMatrix()
#
# 	# make a dictionary containing the word and their index
# 	from pyspark.sql import SQLContext
# 	sqlContext = SQLContext(sc)
# 	model_vectors = sqlContext.read.parquet('/user/rmusters/2015model99/data')
# 	# logger.info("model loaded")
# 	rdd_words = model_vectors.map(lambda line: line[0])
# 	words = rdd_words.collect()  # 15919
# 	word_dict = {}
# 	for i, w in enumerate(words):
# 		word_dict[w] = i
#
# 	from lda import predict_cluster
# 	data_cluster = data.map(lambda (text, filtered_text, id):
# 							(text, filtered_text, predict_cluster(filtered_text, topics, topicsMatrix, word_dict), id))
# 	df = data_cluster.toDF(["text", "filtered_text", "cluster", "id"])
# 	path = 'hdfs:///user/rmusters/lda_data_cluster'
	df.write.parquet(path, mode="overwrite")


def bow(filtered_text, words):
	word_dict = {}
	vector_dict = {}
	for i, v in enumerate(words):
		word_dict[v] = i
		vector_dict[i] = 0
	for w in filtered_text:
		if w in words:
			vector_dict[word_dict[w]] = vector_dict[word_dict[w]]  + 1
	return vector_dict

def save_bow_data(path):
	sc = SparkContext(appName='save_bow_data', conf=conf)
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)

	model_vectors = sqlContext.read.parquet(path + 'threshold20_2015model56/data')
	rdd_words = model_vectors.map(lambda line: line[0])
	words = rdd_words.collect()  # 15919
	size = len(words)
	print size
	from pyspark.mllib.linalg import SparseVector
	data = sqlContext.read.parquet(path + "data_sample")
	data = data.map(lambda (text, filtered_text, id): (text, filtered_text, SparseVector(size, bow(filtered_text, words)), id))
	df = data.toDF(["text", "filtered_text", "vectors", "id"])
	df.write.parquet(path + "bow_data", mode="overwrite")

def save_bow_data_csv(path):
	sc = SparkContext(appName='save_bow_data_csv', conf=conf)
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	data = sqlContext.read.parquet(path + "bow_data")
	data.write.format('com.databricks.spark.csv').save('bow_data.csv')

if __name__ == "__main__":
	path = 'hdfs:///user/rmusters/'
	#kmeans_w2v()
	#kmeans_bow(path)
	#save_bow_data(path)
	#save_bow_data_csv(path)
	#kmeans_w2v_predict()
	kmeans_lda_predict()