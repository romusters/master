from pyspark import SparkContext, SparkConf

conf = (SparkConf().set("spark.driver.maxResultSize", "0"))
sc = SparkContext(appName='stemmerTest', conf=conf)
import nltk

#http://help.mortardata.com/technologies/spark/load_and_transform_data