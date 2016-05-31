from pyspark import SparkContext, SparkConf
import sys

#spark-submit --master yarn --executor-memory 32g --driver-memory 32g --deploy-mode cluster --num-executors 400 master/hadoop/loadw2v.py

conf = (SparkConf()
		.set("spark.driver.maxResultSize", "0")
		#.set('spark.executor.memory','32g')
		#.set('spark.driver.memory','32g')
		)


sc = SparkContext(appName='loadw2v', conf=conf)

from pyspark.mllib.feature import Word2VecModel
model = Word2VecModel.load(sc, '/user/rmusters/pymodel_filtered3.bin')

vec = model.getVectors()
vec = str(vec)

path =  '/user/rmusters/vectors.txt'
with open(path, 'w') as textfile:
	textfile.write(vec)


print sys.getsizeof(model.getVectors)