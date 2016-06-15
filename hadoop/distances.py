from pyspark import SparkContext, SparkConf
import logging, sys
import numpy as np

# spark-submit --master yarn --deploy-mode cluster  master/hadoop/distances.py

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

loc = '/user/rmusters/vectors.csv'

conf = (SparkConf()
    .set("spark.driver.maxResultSize", "0")\
	.set("driver-memory", "6g")\
	.set("executor-memory", "6g")\
	.set("num-executors", "100"))

sc = SparkContext(appName='distances', conf=conf)

data = sc.textFile(loc).map(lambda line: line.split(",", 1))\
	.map(lambda line: (line[0], eval(line[1].replace('WrappedArray', "").replace("(", "[").replace(")", "]"))))

base = data.take(1)
base = base[1]
logger.info(base)
logger.info(type(base))

base = np.asarray(base)

data = data.map(lambda line: abs(np.sum(np.subtract(np.asarray(line[1]), base))))

path =  'hdfs:///user/rmusters/distances.txt'
data.saveAsTextFile(path)
# distances = sc.pickleFile(loc)
# data = distances.map(lambda line: eval(line))
#logger.info(data.values().take(1))