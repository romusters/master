from pyspark import SparkContext, SparkConf

conf = (SparkConf()
		.set("spark.driver.maxResultSize", "0")
		.set('spark.executor.memory','32g')
		.set('spark.driver.memory','32g'))


sc = SparkContext(appName='loadw2v', conf=conf)

from pyspark.mllib.feature import Word2VecModel
model = Word2VecModel.load(sc, '/user/rmusters/pymodel_filtered3.bin')
print model.getVectors()
