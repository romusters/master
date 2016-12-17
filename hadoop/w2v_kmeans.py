from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark import SparkContext, SparkConf
from math import sqrt
import logging, sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def kmeans_w2v():
	sc = SparkContext(appName='kmeans_w2v')
	from pyspark.sql import SQLContext

	sqlContext = SQLContext(sc)

	df_path = "hdfs:///user/rmusters/lambert_w2v_data_jan"
	df = sqlContext.read.parquet(df_path)
	data = df.select("vectors")
	parsedData = data.dropna().map(lambda line: line[0])


	for n_clusters in range(500,540,20):
		# Build the model (cluster the data)
		clusters = KMeans.train(parsedData, n_clusters, maxIterations=10, runs=10, initializationMode="random")

		# Evaluate clustering by computing Within Set Sum of Squared Errors
		def error(point):
			center = clusters.centers[clusters.predict(point)]
			return sqrt(sum([x**2 for x in (point - center)]))

		WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
		logger.info("Within Set Sum of Squared Error = " + str(n_clusters) + "&" +  str(WSSSE))

	# Save and load model
		if n_clusters == 500:
			clusters.save(sc, "hdfs:///user/rmusters/lambert_kmeans_w2v_jan")

def w2v_predict(vector, model):
	if vector == None:
		return None
	else:
		return model.predict(vector)


def kmeans_w2v_predict():
	appName='kmeans_w2v_predict'
	sc = SparkContext(appName=appName)
	from pyspark.sql import SQLContext
	from pyspark.mllib.clustering import KMeans, KMeansModel
	sqlContext = SQLContext(sc)
	data = sqlContext.read.parquet("hdfs:///user/rmusters/lambert_w2v_data_jan")
	df = data.toDF("text", "filtered_text", "split_text", "vectors", "id")
	df = df.where(df.vectors.isNotNull())
	data = df.rdd
	model = KMeansModel.load(sc, "hdfs:///user/rmusters/lambert_kmeans_w2v_jan")

	data = data.map(lambda (text, filtered_text, split_text, vectors, id): (text, filtered_text, split_text, vectors, model.predict(vectors), id))
	df = data.toDF(["text", "filtered_text", "split_text", "vectors", "cluster", "id"])
	df = df.sort(df.cluster.asc())
	df.write.parquet("hdfs:///user/rmusters/lambert_w2v_data_cluster", mode= "overwrite")
	logger.info(appName)

def test_cluster_entities():
	appName='show cluster entities'
	sc = SparkContext(appName=appName)
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	data = sqlContext.read.parquet("hdfs:///user/rmusters/lambert_w2v_data_cluster")
	ids = [9712, 32444, 163208784628, 11095, 32444, 343597394914]
	for id in ids:
		print id
		cluster = data.where(data.id == id).select("cluster").collect()[0][0]
		tweets = data.where(data.cluster == cluster).select("text").sample(False, 0.1).take(5)
		for t in tweets:
			print t

		print

def test_cluster_separation():
	appName='test cluster seperation'
	sc = SparkContext(appName=appName)
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	data = sqlContext.read.parquet("hdfs:///user/rmusters/lambert_w2v_data_cluster")
	voetbal = data.where(data.id == 9712).select("cluster").collect()[0][0]
	voetbal = data.where(data.cluster == voetbal).select("vectors", "cluster")
	moslim = data.where(data.id == 32444).select("cluster").collect()[0][0]
	moslim = data.where(data.cluster == moslim).select("vectors", "cluster")
	moslim.save("moslim_vectors.csv", "com.databricks.spark.csv")
	voetbal.save("voetbal_vectors.csv", "com.databricks.spark.csv")

def save_cluster_ids():
	appName='save cluster ids'
	sc = SparkContext(appName=appName)
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	data = sqlContext.read.parquet("hdfs:///user/rmusters/lambert_w2v_data_cluster")
	data = data.select("cluster", "id")
	data.save("cluster_id.csv", "com.databricks.spark.csv")

# kmeans_w2v()
# kmeans_w2v_predict()
# test_cluster_entities()
# test_cluster_separation()
save_cluster_ids()