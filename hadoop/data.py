from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import logging, sys
import filter

#spark-submit --py-files master/hadoop/stemmer.py,master/hadoop/filter.py --packages com.databricks:spark-csv_2.10:1.4.0 --master yarn --deploy-mode cluster  master/hadoop/data.py

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


conf = (SparkConf()
    .set("spark.driver.maxResultSize", "0")\
	.set("spark.driver.memory", "10g")\
	.set("spark.executor.memory", "10g")\
	.set("spark.executor.instances", "50")\
	.set("spark.executor.cores", "4")\
	.set("spark.rpc.askTimeout", "120000"))

sc = SparkContext(appName='Data', conf=conf)
sqlContext = SQLContext(sc)

def data_jan():
	from pyspark.sql.functions import monotonicallyIncreasingId
	loc = '/user/rmusters/text/2015/01/*'
	text_file = sc.textFile(loc)
	data = text_file.map(lambda (text): (text, filter.filter(text)))
	data = data.toDF(["text", "filtered_text"]).withColumn("id", monotonicallyIncreasingId())

	path = '/user/rmusters/data_jan'
	data.write.parquet(path, mode="overwrite")


def save_data_sample(path):
	data = sqlContext.read.parquet(path)
	data = data.sample(False, 0.01)
	data.write.parquet("hdfs:///user/rmusters/data_sample", mode="overwrite")

def save_data_csv(path):
	data = sqlContext.read.parquet(path)#.select("filtered_text")
	data.save(path + ".csv", "com.databricks.spark.csv", "overwrite")

path = 'hdfs:///user/rmusters/data_jan'
# save_data_csv(path)
data_jan()
save_data_sample(path)
save_data_csv(path)