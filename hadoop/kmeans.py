from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark import SparkContext, SparkConf
from math import sqrt
import logging, sys

#spark-submit --master yarn --deploy-mode cluster master/hadoop/kmeans.py

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


conf = (SparkConf()
    .set("spark.driver.maxResultSize", "0")\
	.set("spark.driver.memory", "50g")\
	.set("spark.executor.memory", "20g") \
	.set("spark.executor.cores", "4") \
	.set("spark.executor.instances", "400"))


def kmeans_w2v():
	sc = SparkContext(appName='kmeans_w2v', conf=conf)
	from pyspark.sql import SQLContext

	sqlContext = SQLContext(sc)

	df_path = "hdfs:///user/rmusters/average_tweet_distances"
	df = sqlContext.read.parquet(df_path)
	rddData = df.select("vectors").rdd

	from numpy import array
	parsedData = rddData.map(lambda line: array(line[0]))

	for n_clusters in range(0,150,1):
		# Build the model (cluster the data)
		clusters = KMeans.train(parsedData, n_clusters, maxIterations=10, runs=10, initializationMode="random")

		# Evaluate clustering by computing Within Set Sum of Squared Errors
		def error(point):
			center = clusters.centers[clusters.predict(point)]
			return sqrt(sum([x**2 for x in (point - center)]))

		WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
		logger.info("Within Set Sum of Squared Error = " + str(n_clusters) + "&" +  str(WSSSE) + "\\")

	# Save and load model
	#clusters.save(sc, "myModelPath")
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
	for n_clusters in range(220,400,20):
		logger.info("Cluster amount is:"  + str(n_clusters))
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
	kmeans_bow(path)
	#save_bow_data(path)
	#save_bow_data_csv(path)