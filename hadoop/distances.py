from pyspark import SparkContext, SparkConf, SQLContext
import logging, sys
import numpy as np


# spark-submit --py-files master/hadoop/stemmer.py,master/hadoop/filter.py --master yarn --deploy-mode cluster  master/hadoop/distances.py

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

loc = '/user/rmusters/vectors.csv'

conf = (SparkConf()
    .set("spark.driver.maxResultSize", "0")\
	.set("driver-memory", "6g")\
	.set("executor-memory", "6g")\
	.set("num-executors", "100"))

sc = SparkContext(appName='distances', conf=conf)


import filter
from pyspark.mllib.feature import Word2Vec, Word2VecModel
# load data
loc = '/user/rmusters/text/2015/01/*'
text_file = sc.textFile(loc)
data = text_file.map(lambda line: filter.filter(line).split(" "))

# load model
word2vec = Word2Vec()
model =  Word2VecModel.load(sc, '/user/rmusters/2015model99')

# get a tweet vector pair.
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
lookup = sqlContext.read.parquet('/user/rmusters/2015model99/data').alias("lookup")
lookup_bd = sc.broadcast(lookup.rdd.collectAsMap())


vectors = data.map(lambda ws: [lookup_bd.value.get(w) for w in ws])
logger.info(vectors.count())
path =  'hdfs:///user/rmusters/distances.txt'
vectors.saveAsTextFile(path)
# res = vectors.take(2)
#
# vectors = data.map(lambda line: model.transform(line[0]))
# model.transform("ik")

# data = sc.textFile(loc).map(lambda line: line.split(",", 1))\
# 	.map(lambda line: (line[0], eval(line[1].replace('WrappedArray', "").replace("(", "[").replace(")", "]"))))




# Sort of learning vector quantization!
# present a tweet to the user which is not that far from the original tweet.
	# if the tweet is correct
		# construct a prototype
				# repeat until convergence
	# if the tweet is not correct, present a tweet which is less far from the original tweet
		# repeat


###########################
#OLD distance function
# take a tweet
base = data.take(1)
# calculate the distance to all other tweets
data = data.map(lambda line: abs(np.sum(np.subtract(np.asarray(eval(line[1])), np.asarray(eval(base[0][1]))))))
path =  'hdfs:///user/rmusters/distances.txt'
data.saveAsTextFile(path)
