from pyspark import SparkContext, SparkConf
import sys

# spark-submit --master yarn --executor-memory 2g --driver-memory 2g --deploy-mode cluster --num-executors 4 master/hadoop/saveFileTest.py

conf = (SparkConf()
		.set("spark.driver.maxResultSize", "0")
		# .set('spark.executor.memory','32g')
		# .set('spark.driver.memory','32g')
		)

sc = SparkContext(appName='saveFileTest', conf=conf)

path =  'hdfs:///user/rmusters/vectors.txt'
rdd = sc.parallelize([1,2])
rdd.saveAsTextFile(path)