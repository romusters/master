from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark import SparkContext, SparkConf
from math import sqrt
import logging, sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def kmeans_lda():

	from pyspark.mllib.clustering import KMeans, KMeansModel
	from pyspark import SparkContext, SparkConf
	from pyspark.sql import SQLContext

	sc = SparkContext(appName='kmeans_lda')

	sqlContext = SQLContext(sc)
	path = "/user/rmusters/lda_doc_topic"
	data = sqlContext.read.parquet(path)
	data = data.select("_2")
	data = data.map(lambda line: line[0])
	#data.write.format('com.databricks.spark.csv').save('lda_doc_topic.csv')
	clusters = None
	for n_clusters in range(500, 501, 1):
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
		clusters.save(sc, "/user/rmusters/kmeans_lda_jan")

#first do lda training, then do scala lda doc topic, then do kmeans training and then kmeans prediction
def kmeans_lda_predict():
	appName = 'kmeans_lda_predict'
	from pyspark.mllib.clustering import KMeans, KMeansModel
	sc = SparkContext(appName=appName)
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	data = sqlContext.read.parquet("hdfs:///user/rmusters/lda_doc_topic") #lda_data_jan
	# data = df.rdd
	model = KMeansModel.load(sc, "hdfs:///user/rmusters/kmeans_lda_jan")
	data = data.map(lambda (id, vectors): (id, vectors, model.predict(vectors)))
	df = data.toDF(["id", "vectors", "cluster"])
	df = df.sort(df.cluster.asc())
	df.write.parquet("hdfs:///user/rmusters/lda_data_cluster", mode= "overwrite")
	logger.info(appName)

def merge_data():
	sc = SparkContext(appName='merge_data')
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	data = sqlContext.read.parquet("hdfs:///user/rmusters/data_jan_sample")
	# data = data.drop("vectors")
	lda_data = sqlContext.read.parquet("hdfs:///user/rmusters/lda_data_cluster")
	joined = lda_data.join(data, data.id == lda_data.id)
	joined = joined.map(lambda (id2, vectors, cluster, text, filtered_text, id): (text, filtered_text, vectors, cluster, id))
	df = joined.toDF(["text", "filtered_text", "vectors", "cluster", "id"])
	df.write.parquet("hdfs:///user/rmusters/lda_data_jan_cluster_merged", mode="overwrite")



def test():
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	data = sqlContext.read.parquet("hdfs:///user/rmusters/lda_data_jan")
	print data.count()
	from pyspark.sql.functions import col
	vectors = data.select("vectors").where(col("vectors").isNotNull())
	print vectors.count()
#kmeans_lda()
# kmeans_lda_predict()
# test()
merge_data()