

from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import Word2Vec, HashingTF
from pyspark.sql import SQLContext
import logging, sys
# import filter

#spark-submit --py-files master/hadoop/stemmer.py,master/hadoop/filter.py --packages com.databricks:spark-csv_2.10:1.4.0 --master yarn --executor-memory 20g --deploy-mode cluster  master/hadoop/w2v.py
#spark-submit --master yarn --deploy-mode cluster --packages com.databricks:spark-csv_2.10:1.4.0 --driver-memory 50g --executor-memory 10g --num-executors 100 w2v.py

#if in yarn mode, all cores are used

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

loc = '/user/rmusters/text/2015/01/*'

conf = (SparkConf()
    .set("spark.driver.maxResultSize", "0")\
	.set("spark.driver.memory", "50g")\
	.set("spark.executor.instances", "200")\
	.set("spark.executor.cores", "2")\
	.set("spark.rpc.askTimeout", "600000") \
	.set("spark.akka.frameSize", "600")\
	.set("spark.executor.heartbeatInterval", "600000") \
	.set("spark.executor.heartbeat", "600000"))

sc = SparkContext(appName='Word2Vec')
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


def mean_slice(slice):
	import numpy as np
	if len(slice) == 0:
		print "zero slice found"
		return None
	else:
		return np.mean(slice, axis = 0).tolist()

import re
def process(text):
	tokens = text.split()
	res = []
	for token in tokens:
		# token = token.lower()
		token = token.strip()
		if token not in ["<stopword>", "<mention>", "<url>", "rt"]:
			# if token[0:2] == "06" and len(token) == 10:
			# 	res.append("<mobiel>")
			# else:
			# 	parts = re.split('(\d+)', token)  # match(r"([a-z]+)([0-9]+)", token, re.I)
			# 	for part in parts:
			# 		if part != '':
			# 			res.append(part)
			res.append(token)
	return res

def average_word_vectors2():
	print "average vectors 2"
	lookup_bd = load_model('/user/rmusters/lambert_jan_2015model')
	keys = lookup_bd.value.keys()
	# data = sqlContext.read.parquet("/user/rmusters/data_jan_sample")
	data = sqlContext.read.parquet("/user/rmusters/data_jan")#.sample(False, 0.0001)

	data = data.dropna()
	# text = data.select("filtered_text", "id")
	# text = text.map(lambda (filtered_text, id): (filtered_text, process(filtered_text), id))
	text = data.map(lambda (text, filtered_text, id): (text, filtered_text, process(filtered_text), id))
	df = text.toDF(["text", "filtered_text", "tokens", "id"])
	df.write.parquet("hdfs:///user/rmusters/processed", mode="overwrite")
	df.write.format("com.databricks.spark.csv").mode("overwrite").save("processed.csv")
	# # mean_vectors = text.map(lambda (filtered_text, processed, id): (filtered_text, processed, mean_slice([lookup_bd.value.get(w) for w in processed if w in keys and lookup_bd.value.get(w) is not None ]), id))
	# mean_vectors = text.map(lambda (text, filtered_text, tokens, id): (text, filtered_text, tokens, mean_slice([lookup_bd.value.get(w) for w in tokens if w in keys and lookup_bd.value.get(w) is not None]), id))
	# #
	# # df = mean_vectors.toDF(["pre_processed", "tokens", "vectors", "id"])
	# df = mean_vectors.toDF(["text", "filtered_text", "tokens", "vectors", "id"])
	# # result = df.select("tokens", "vectors", "id")
	# df.write.parquet("hdfs:///user/rmusters/lambert_w2v_data_jan", mode="overwrite")
	# df.write.format("com.databricks.spark.csv").mode("overwrite").save("lambert_w2v_data_jan.csv")
	# # data = data.map(lambda (text, filtered_text, id):  (text, filtered_text, [w for w in filtered_text.split() if not w == None and w != "<stopword>" and w != "<mention>" and w !="<URL>"and w in keys] , id))



# def average_word_vectors():
# 	lookup_bd = load_model('/user/rmusters/lambert_jan_2015model')
# 	keys = lookup_bd.value.keys()
# 	data = sqlContext.read.parquet("/user/rmusters/data_jan_sample")
# 	data = data.map(lambda (text, filtered_text, id):  (text, filtered_text, [w for w in filtered_text.split() if not w == None and w != "<stopword>" and w != "<mention>" and w !="<url>"and w in keys] , id))
# 	data = data.map(lambda (text, filtered_text, split_text, id):  (text, filtered_text, split_text, mean_slice([lookup_bd.value.get(w) for w in split_text ]), id))
# 	df = data.toDF(["text", "filtered_text", "split_text", "vectors", "id"])
# 	df.write.parquet("hdfs:///user/rmusters/lambert_w2v_data_jan", mode="overwrite")
# 	df.select("text", "filtered_text", "vectors", "id").save("lambert_w2v_data_jan.csv", "com.databricks.spark.csv", "overwrite")



def save_vectors(path):
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

def mean_vectors():
	print "mean vectors"
	lookup_bd = load_model('/user/rmusters/lambert_jan_2015model_tokenizer')
	keys = lookup_bd.value.keys()
	data = sqlContext.read.parquet("hdfs:///user/rmusters/processed").sample(False, 0.01)
	data = data.select(["id", "tokens"])
	df = data.map(lambda (id, tokens): (id, tokens, mean_slice([lookup_bd.value.get(w) for w in tokens if w in keys and lookup_bd.value.get(w) is not None]))).toDF(["id", "tokens", "vectors"])
	df.write.parquet("hdfs:///user/rmusters/lambert_w2v_data_jan", mode="overwrite")
	df.write.format("com.databricks.spark.csv").mode("overwrite").save("lambert_w2v_data_jan.csv")

if __name__ == "__main__":
	import numpy as np
	# average_word_vectors2()
	mean_vectors()
# 	path = '/user/rmusters/threshold20_2015model56'
# 	#main()
# 	#save_vectors(path)
# 	#save_w2v_data()
# 	average_word_vectors2()
# 	vectors = sqlContext.read.parquet("hdfs:///user/rmusters/lambert_w2v_data_jan")
# 	vectors.write.format("com.databricks.spark.csv").mode("overwrite").save("lambert_w2v_data_jan.csv")
# 	#train_w2v()
# 	# lambert()


#from pyspark.mllib.feature import Word2VecModel
#model = Word2VecModel.load(sc,"word_vec_from_cleaned_query.model")

