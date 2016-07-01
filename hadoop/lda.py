from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import LDA, LDAModel
import logging, sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

#spark-submit --master yarn --deploy-mode cluster master/hadoop/lda.py

conf = (SparkConf()\
		.set("spark.driver.maxResultSize", "0")\
		.set("spark.driver.memory", "20g")\
		.set("spark.executor.memory", "20g")\
		.set("spark.executor.instances", "400"))

# Larger file
loc = '/user/rmusters/text/2015/*/*'


def train_model():
	sc = SparkContext(appName='lda_train', conf=conf)
	from pyspark.sql import SQLContext

	sqlContext = SQLContext(sc)

	model_vectors = sqlContext.read.parquet('/user/rmusters/2015model99/data')
	logger.info("model loaded")
	rdd_words = model_vectors.map(lambda line: line[0])
	words = rdd_words.collect() #15919
	logger.info("words collected")

	def load_data(path):
		sqlContext = SQLContext(sc)
		data = sqlContext.read.parquet(path)
		return data

	path = 'hdfs:///user/rmusters/data'
	data = load_data(path)
	logger.info("data loaded")
	data = data.sample(False, 0.01)
	logger.info("data sampled")
	data = data.drop("vectors")

	def bow(filtered_text):
		vector = [0]*len(words)
		for i, w in enumerate(words):
			if w in filtered_text:
				vector[i] = vector[i]+1
		v_dict = {}
		for i, v in enumerate(vector):
			if v >= 1:
				v_dict[i] = v
			else:
				v_dict[i] = 0 # don't forget this or else the vocabsize and vector size will differ.
		return v_dict

	#check if sum of vector is zero 13 times. This indicates the datasample does not contain certain words and thus the sparse vector removes them
	from pyspark.mllib.linalg import SparseVector
	size = len(words)
	data = data.map(lambda (text, filtered_text, id): (text, filtered_text, SparseVector(size, bow(filtered_text)), id))
	logger.info("bag of words data")

	corpus = data.map(lambda (text, filtered_text, vector, id): [id, vector])

	logger.info("Training the lda model")
	ldaModel = LDA.train(corpus, k=100)
	logger.info(ldaModel.vocabSize())

	ldaModel.save(sc, 'hdfs:///user/rmusters/ldaModel2')

def load_model():
	from pyspark.mllib.clustering import LDA, LDAModel
	sc = SparkContext(appName='lda_load', conf=conf)
	path = "/user/rmusters/ldaModel"
	ldaModel = LDAModel.load(sc, path)
	return ldaModel

def predict():
	# probabilities for each word are extraced from each topic. The probabilities for all the words per cluster are summed.
	# the cluster with the heighest sum wins.
	ldaModel = load_model()

	topics = ldaModel.describeTopics()

	topics = ldaModel.topicsMatrix()
	for topic in range(100):
		print "Topic " + str(topic) + ":"
		for word in range(0, ldaModel.vocabSize()):
			print " " + str(topics[word][topic])


if __name__ == "__main__":
	import logging, sys
	#train_model()
	model = load_model()
	logger.info(model.vocabSize())
	#predict()