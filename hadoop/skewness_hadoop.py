from pyspark import SparkContext, SparkConf
import pyspark.sql.functions
import logging, sys


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

conf = (SparkConf()
    .set("spark.driver.maxResultSize", "40g"))

sc = SparkContext(appName='skewness', conf=conf)
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

path = 'hdfs:///user/rmusters/lda_data_cluster'

data = sqlContext.read.parquet(path)
lists = data.select("vectors").collect()
print len(lists)
columns = sc.parallize(lists)
print pyspark.sql.functions.skewness[columns[0]]
