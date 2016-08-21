from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import Word2Vec, HashingTF
from pyspark.sql import SQLContext
import logging, sys
import filter

#spark-submit --py-files master/hadoop/stemmer.py,master/hadoop/filter.py --packages com.databricks:spark-csv_2.10:1.4.0 --master yarn --executor-memory 20g --deploy-mode cluster  master/hadoop/w2v.py
#if in yarn mode, all cores are used

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

loc = '/user/rmusters/text/2015/01/*'

conf = (SparkConf()
    .set("spark.driver.maxResultSize", "0")\
	.set("spark.driver.memory", "50g")\
	.set("spark.executor.instances", "10")\
	.set("spark.executor.cores", "2")\
	.set("spark.rpc.askTimeout", "120000") \
	.set("spark.akka.frameSize", "300"))

sc = SparkContext(appName='Word2Vec', conf=conf)
sqlContext = SQLContext(sc)

def main():
	# Threshold to limit words which occur less than the threshold
	threshold = 10#10
	text_file = sc.textFile(loc)

	data = text_file.map(lambda line: filter.filter(line))

	counts = data.flatMap(lambda line: line.split(" ")) \
				 .map(lambda word: (word, 1)) \
				 .reduceByKey(lambda a, b: a + b) \
				.filter(lambda pair: pair[1] >= threshold)
				#.sortBy(lambda x:x[1], ascending=True) #only use for inspection
	counts.cache()
	vocab_size = counts.count()
	print "Vocabulary size is: ", vocab_size

	inp = data.map(lambda line: line.split(" "))
	inp.cache()


	max_int_size = 268435455
	vector_size = max_int_size / vocab_size
	print "Vector size is: ", vector_size
	word2vec = Word2Vec()
	word2vec.setMinCount(threshold)#40
	word2vec.setVectorSize(vector_size)#/100

	for idx in range(1, 100, 1):
		print idx
		model = word2vec.fit(inp.sample(False, 0.01))
		# if idx == 1 or idx == 2:
		# 	print "Vector size of current model:  ", word2vec.getVectorSize()
		# 	inputcol = word2vec.getInputCol()
		# 	outputcol =  word2vec.getOutputCol()
		# 	print "input column: ", inputcol
		# 	try:
		# 		print len(inputcol)
		# 		print len(outputcol)
		# 	except:
		# 		pass
		# 	print "output column", outputcol

		model.save(sc, '/user/rmusters/threshold20_2015model' + str(idx))

def load_model(path):
	lookup = sqlContext.read.parquet(path + '/data').alias("lookup")
	lookup_bd = sc.broadcast(lookup.rdd.collectAsMap())
	return lookup_bd

def average_word_vectors(vectors):
	def mean(a):
		return sum(a) / len(a)
	vector = []
	for v in vectors:
		if v == None:
			continue
		else:
			vector.append(v)
	return map(mean, zip(*vector))

def average_word_vectors():
	import numpy as np
	lookup_bd = load_model('/user/rmusters/jan_threshold20_2015model99')
	data = sqlContext.read.parquet("/user/rmusters/data_jan_sample")
	data = data.map(lambda (text, filtered_text, id):  (text, filtered_text, np.mean([lookup_bd.value.get(w) for w in filtered_text if not lookup_bd.value.get(w) == None], axis=0).tolist(), id))
	df = data.toDF(["text", "filtered_text", "vectors", "id"])
	df.write.parquet("/user/rmusters/w2v_data_jan", mode="overwrite")

def save_vectors(path):
	#pyspark --packages com.databricks:spark-csv_2.10:1.4.0
	vectors = sqlContext.read.parquet(path + '/data')
	vectors.save("w2v_vectors.csv", "com.databricks.spark.csv")

def train_w2v():
	threshold = 20

	data = sqlContext.read.parquet('hdfs:///user/rmusters/data_jan').select("filtered_text")

	counts = data.flatMap(lambda line: line) \
		.map(lambda word: (word, 1)) \
		.reduceByKey(lambda a, b: a + b) \
		.filter(lambda pair: pair[1] >= threshold)

	vocab_size = 67585#counts.count()
	print "Vocabulary size is: ", vocab_size

	data = data.map(lambda line: line.filtered_text.split())

	max_int_size = 268435455
	vector_size = max_int_size / vocab_size
	print "Vector size is: ", vector_size
	word2vec = Word2Vec()
	word2vec.setMinCount(threshold)
	word2vec.setVectorSize(vector_size)

	for idx in range(1, 100, 1):
		print idx
		model = word2vec.fit(data.sample(False, 0.01))
		model.save(sc, '/user/rmusters/jan_threshold20_2015model' + str(idx))

#load the word2vec model
def load_w2v_model():
	from pyspark.sql import SQLContext
	sqlContext = SQLContext(sc)
	lookup = sqlContext.read.parquet('/user/rmusters/2015model99/data').alias("lookup")
	lookup_bd = sc.broadcast(lookup.rdd.collectAsMap())
	return lookup_bd

#old
def cos_similarity(tweet, other_tweet, model):
	import numpy
	sims = []
	for word in tweet:
		tmp_cos_sim = 0
		try:
			word = model.value.get(word)
		except TypeError:
			continue
		for other_word in other_tweet:
			other_word = model.value.get(other_word)
			#cos_sim = numpy.dot(model.transform(word), model.transform(other_word)) / (numpy.linalg.norm(model.transform(word)) * numpy.linalg.norm(model.transform(other_word)))
			try:
				cos_sim = numpy.dot(word, other_word) / (numpy.linalg.norm(word) * numpy.linalg.norm(other_word))
			except TypeError:
				continue
			if cos_sim > tmp_cos_sim:
				tmp_cos_sim = cos_sim
		sims.append(tmp_cos_sim)
	return float(sum(sims)/len(sims))

if __name__ == "__main__":
	path = '/user/rmusters/threshold20_2015model56'
	#main()
	#save_vectors(path)
	#save_w2v_data()
	average_word_vectors()
	#train_w2v()


#from pyspark.mllib.feature import Word2VecModel
#model = Word2VecModel.load(sc,"word_vec_from_cleaned_query.model")