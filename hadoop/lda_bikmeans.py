import logging, sys
# from pyspark.mllib.clustering import BisectingKMeans, BisectingKMeansModel
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)



def bikmeans_lda():

	conf = (SparkConf()
			.set("spark.driver.maxResultSize", "10g"))
	sc = SparkContext(appName='bi-kmeans_lda', conf=conf)

	sqlContext = SQLContext(sc)
	path = "/user/rmusters/lda_doc_topic"
	data = sqlContext.read.parquet(path)
	data = data.select("_2").rdd
	data = data.map(lambda line: line[0])
	# data = data.sample(False, 0.001)
	model = BisectingKMeans.train(data, 500, maxIterations=10)

	cost = model.computeCost(data)
	print "Bisecting K-means Cost = " + str(cost)

	# data = sqlContext.read.parquet("hdfs:///user/rmusters/lda_doc_topic")
	vectors = data.collect()
	clusters = []
	for i in range(0, len(vectors)):
		cluster = model.predict(vectors[i])
		print cluster
		clusters.append(cluster)
	clusters_df = sc.parallelize(clusters).toDF()
	clusters_df.write.parquet("hdfs:///user/rmusters/lda_data_cluster", mode="overwrite")

#def predict can not be implemented, because the model is not on the workers.

def join_clusters_with_lda_data():
	from pyspark import SparkContext, SparkConf
	from pyspark.sql import SQLContext, Row
	from pyspark.sql.functions import monotonicallyIncreasingId

	conf = (SparkConf()
			.set("spark.driver.maxResultSize", "20g")\
			.set("spark.rpc.askTimeout", "120000")\
			.set("spark.executor.heartbeatInterval", "120000"))

	sc = SparkContext(appName='join_clusters_with_lda_data', conf=conf)
	sqlContext = SQLContext(sc)

	data_path = "/user/rmusters/lda_doc_topic"
	cluster_path = "/user/rmusters/lda_clusters"

	data = sqlContext.read.parquet(data_path)
	data.show()
	data = data.withColumn("id", monotonicallyIncreasingId())

	cluster = sqlContext.read.parquet(cluster_path)  # .sample(False, 0.00001)#.map(lambda x: Row(cluster=x[0]))
	cluster.show()
	cluster = cluster.withColumn("id", monotonicallyIncreasingId())

	result = data.join(cluster, on="id")
	result = result.drop("id")
	result = result.withColumnRenamed("_1", "id").withColumnRenamed("_2", "vectors")
	result.write.parquet("hdfs:///user/rmusters/bisecting_lda_data_cluster", mode="overwrite")
	result.show()






def merge_data():
	sc = SparkContext(appName='merge_data')
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	data = sqlContext.read.parquet("hdfs:///user/rmusters/data_jan_sample")
	lda_data = sqlContext.read.parquet("hdfs:///user/rmusters/bisecting_lda_data_cluster")
	joined = lda_data.join(data, data.id == lda_data.id)
	joined = joined.map(
		lambda (id2, vectors, cluster, text, filtered_text, id): (text, filtered_text, vectors, cluster, id))
	df = joined.toDF(["text", "filtered_text", "vectors", "cluster", "id"])
	df.write.parquet("hdfs:///user/rmusters/bisecting_lda_data_jan_cluster_merged", mode="overwrite")

	# data = data.rdd.map(lambda (id, vectors): (id, vectors, model.predict(vectors)))
	# df = data.toDF(["id", "vectors", "cluster"])
	# df = df.sort(df.cluster.asc())
	# df.write.parquet("hdfs:///user/rmusters/bisecting_lda_data_cluster", mode="overwrite")

def get_lda_doc_topic():
	sc = SparkContext(appName='get_lda_doc_topic')
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	path = "/user/rmusters/lda_doc_topic"
	data = sqlContext.read.parquet(path)
	data.write.format('com.databricks.spark.csv').save('lda_doc_topic.csv')

# bikmeans_lda()
# get_lda_doc_topic()
# join_clusters_with_lda_data()
merge_data()