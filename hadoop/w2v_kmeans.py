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

	df_path = "hdfs:///user/rmusters/w2v_data_jan"
	df = sqlContext.read.parquet(df_path)
	data = df.select("vectors")
	print data.take(1)
	parsedData = data.map(lambda line: line[0]).filter(lambda line: line is not None)
	print parsedData.take(1)

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
			clusters.save(sc, "hdfs:///user/rmusters/kmeans_w2v_jan")

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
	data = sqlContext.read.parquet("hdfs:///user/rmusters/w2v_data_jan")
	df = data.toDF("text", "filtered_text", "vectors", "id")
	df = df.where(df.vectors.isNotNull())
	data = df.rdd
	model = KMeansModel.load(sc, "hdfs:///user/rmusters/kmeans_w2v_jan")

	data = data.map(lambda (text, filtered_text, vectors, id): (text, filtered_text, vectors, model.predict(vectors), id))
	df = data.toDF(["text", "filtered_text", "vectors", "cluster", "id"])
	df = df.sort(df.cluster.asc())
	df.write.parquet("hdfs:///user/rmusters/w2v_data_cluster", mode= "overwrite")
	logger.info(appName)


#kmeans_w2v()
kmeans_w2v_predict()