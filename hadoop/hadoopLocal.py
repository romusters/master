

import os
import sys
import utils

# Path for spark source folder
os.environ['SPARK_HOME']="/home/robert/spark-1.6.1-bin-hadoop2.6"

# Append pyspark  to Python Path
sys.path.append("/home/robert/spark-1.6.1-bin-hadoop2.6/bin")

try:
	from pyspark import SparkContext
	from pyspark import SparkConf
	from pyspark import SparkContext
	from pyspark.mllib.feature import Word2Vec, Word2VecModel
	print ("Successfully imported Spark Modules")

except ImportError as e:
	print ("Can not import Spark Modules", e)
	sys.exit(1)

dir = '/user/rmusters/'
# utils.removeCharFromFilename(dir)
# utils.concatFiles(dir)
word2vec = Word2Vec()
sc = SparkContext(appName='Word2Vec')

filename = "text8.zip"
filename = "12.txt"
inp = sc.textFile(dir + filename).map(lambda row: row.split(" "))

model = word2vec.fit(inp)

model.save(sc, dir + "pymodel.bin")

model =  Word2VecModel.load(sc, dir + "pymodel.bin")

print model.getVectors()
#model.train(inp)