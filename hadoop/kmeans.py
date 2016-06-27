from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark import SparkContext, SparkConf
from numpy import array
from math import sqrt
import logging, sys
import filter

#spark-submit --py-files master/hadoop/stemmer.py,master/hadoop/filter.py --master yarn --deploy-mode cluster --num-executors 400  master/hadoop/kmeans.py

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


conf = (SparkConf()
    .set("spark.driver.maxResultSize", "0")\
	.set("driver-memory", "50g")\
	.set("executor-memory", "6g")\
	.set("num-executors", "400"))

sc = SparkContext(appName='kmeans', conf=conf)


# loc = '/user/rmusters/text/2015/*/*'

# Load and parse the data
# data = sc.textFile(loc)
# parsedData = data.map(lambda line: filter.filter(line))

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