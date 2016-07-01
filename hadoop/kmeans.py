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
	.set("spark.executor.memory", "20g")\
	.set("spark.executors.instances", "400"))


def kmeans_w2v():
	sc = SparkContext(appName='kmeans_w2v', conf=conf)
	from pyspark.sql import SQLContext

	sqlContext = SQLContext(sc)

	df_path = "hdfs:///user/rmusters/average_tweet_distances"
	df = sqlContext.read.parquet(df_path)
	rddData = df.select("vectors").rdd

	from numpy import array
	parsedData = rddData.map(lambda line: array(line[0]))

	for n_clusters in range(100,150,1):
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

def kmeans_bow():
	sc = SparkContext(appName='kmeans_bow', conf=conf)
	from pyspark.sql import SQLContext

	sqlContext = SQLContext(sc)

	model_vectors = sqlContext.read.parquet('/user/rmusters/2015model99/data')

	rdd_words = model_vectors.map(lambda line: line[0])
	words = rdd_words.collect()  # 15919

	def load_data(path):
		sqlContext = SQLContext(sc)
		data = sqlContext.read.parquet(path)
		return data

	path = 'hdfs:///user/rmusters/data'
	data = load_data(path)
	data = data.sample(False, 0.01)
	data = data.drop("vectors")

	def bow(filtered_text):
		vector = [0] * len(words)
		for i, w in enumerate(words):
			if w in filtered_text:
				vector[i] = vector[i] + 1
		v_dict = {}
		for i, v in enumerate(vector):
			if v >= 1:
				v_dict[i] = v
			else:
				v_dict[i] = 0				#pas dit aan denk ik
		return v_dict

	from pyspark.mllib.linalg import SparseVector
	size = len(words)
	data = data.map(lambda (text, filtered_text, id): (text, filtered_text, SparseVector(size, bow(filtered_text)), id))
	data = data.map(lambda (text, filtered_text, vector, id): vector)
	logging.info(data.take(1))
	logging.info(len(data.take(1)))

	for n_clusters in range(111,150,1):
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
	clusters.save(sc, "/user/rmusters/kmeans_bow")
	#sameModel = KMeansModel.load(sc, "myModelPath")


if __name__ == "__main__":
	#kmeans_w2v()
	kmeans_bow()