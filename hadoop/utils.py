from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import logging, sys


#spark-submit --packages com.databricks:spark-csv_2.10:1.4.0 --master yarn --executor-memory 10g --deploy-mode cluster hadoop/utils.py


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

conf = (SparkConf()
	.set("spark.driver.memory", "10g")\
	.set("spark.executor.instances", "100"))

sc = SparkContext(appName='Get jan data', conf=conf)
sqlContext = SQLContext(sc)

def join():
	sims_name = 'lambert_w2v_data_jan_sims_1.csv'
	df = sqlContext.read.format("com.databricks.spark.csv").load(sims_name)
	print "before"
	df.printSchema()
	df = df.select(["C0"]).map(lambda x: (int(x[0]),)).toDF()  # .collect()
	print "after"
	df.printSchema()

	all_name = "w2v_data_jan"
	all_df = sqlContext.read.parquet(all_name)
	print "all_df"
	all_df.printSchema()

	res = df.join(all_df, df._1 == all_df.id)

	print "result"
	res.printSchema()

	res = res.sort("_1", ascending=True)
	print res.select("_1").show()

	res.save("lambert_w2v_data_jan_all_columns2.csv", "com.databricks.spark.csv")


