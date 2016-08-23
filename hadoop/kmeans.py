from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark import SparkContext, SparkConf
from math import sqrt
import logging, sys

#spark-submit --py-files master/hadoop/lda.py --packages com.databricks:spark-csv_2.10:1.4.0 --master yarn --deploy-mode cluster --driver-memory 50g master/hadoop/kmeans.py

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# conf = (SparkConf()
#     .set("spark.driver.maxResultSize", "0")\
# 	.set("spark.driver.memory", "50g")\
# 	.set("spark.executor.memory", "10g") \
# 	.set("spark.executor.cores", "2") \
# 	.set("spark.executor.instances", "10")\
# 	.set("spark.rpc.askTimeout", "120000"))

def kmeans_bow(path):
	sc = SparkContext(appName='kmeans_bow')
	from pyspark.sql import SQLContext

	sqlContext = SQLContext(sc)
	data = sqlContext.read.parquet(path + "bow_data").select("vectors")

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


def to_csv(paths):
	appName = "to_csv"
	sc = SparkContext(appName=appName, conf=conf)
	for path in paths:
		from pyspark.sql import SQLContext
		sqlContext = SQLContext(sc)
		data = sqlContext.read.parquet(path)
		data.write.format('com.databricks.spark.csv').save(path + '.csv')

def clusters():
	import numpy
	appName = "clusters"
	sc = SparkContext(appName=appName)
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)


	#choose 20 random clusters
	import random
	clusters = range(0, 500, 1)
	rand_clust = []
	for i in range(0, 20, 1):
		rand_clust.append(random.choice(clusters))


	# random_clusters = list(numpy.random.choice(clusters, 20, replace=False).tolist())
	w2v_path = "hdfs:///user/rmusters/w2v_data_cluster"
	w2v_data = sqlContext.read.parquet(w2v_path)
	lda_path = "hdfs:///user/rmusters/lda_data_jan_cluster_merged"
	lda_data = sqlContext.read.parquet(lda_path)

	#from the 20 random clusters, find the w2v data.
	for i in rand_clust:
		print "cluster number: %i", i
		d = w2v_data.where(w2v_data.cluster == i)
		#d.write.format('com.databricks.spark.csv').save(w2v_path + "_" + str(i) + '.csv')

		#take 20 datapoints, which are going to be labeled.
		d = d.take(20)

		#collect the ids which are used to find the LDA clusters.
		ids = []
		for el in d:
			ids.append(el[4])

		lda_data.rdd.filter(lambda x: x.id in ids)
		data = lda_data.toDF("text", "filtered_text", "vectors", "cluster", "id").select("cluster").map(lambda x: x[0]).collect()
		print data
		sys.exit(0)
		# lda_data.rdd.filter(lambda x: x.id in ids).collect()

	logger.info(appName)

def save_to_csv():
	appName = "save_to_csv"
	sc = SparkContext(appName=appName)
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	w2v_path = "hdfs:///user/rmusters/w2v_data_cluster"
	w2v_data = sqlContext.read.parquet(w2v_path)
	w2v_data.write.format('com.databricks.spark.csv').save(w2v_path + ".csv")
	lda_path = "hdfs:///user/rmusters/lda_data_jan_cluster_merged"
	lda_data = sqlContext.read.parquet(lda_path)
	lda_data.write.format('com.databricks.spark.csv').save(lda_path + ".csv")

if __name__ == "__main__":
	path = 'hdfs:///user/rmusters/'

	#kmeans_bow(path)
	#save_bow_data(path)
	#save_bow_data_csv(path)
	#kmeans_w2v_predict()

	paths = ["hdfs:///user/rmusters/lda_data_cluster", "hdfs:///user/rmusters/w2v_data_cluster"]
	#to_csv(paths)

	#kmeans_lda_predict()

	clusters()

	#save_to_csv()
