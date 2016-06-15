import sys
sys.path.insert(0, '..')
from hadoop import filter
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# pyspark --packages com.databricks:spark-csv_2.10:1.4.0
# vectors = sqlContext.read.parquet("testModel2/data")
# vectors.save("vectors2.csv", "com.databricks.spark.csv")

def rm_wrapped_array():
	import csv
	path = "/home/cluster/vectors.csv"
	dict = {}
	with open(path, 'rb') as csvfile:
		rows = csv.reader(csvfile, delimiter=',')
		for row in rows:
			dict[row[0]] = eval(row[1].replace('WrappedArray', "").replace("(", "[").replace(")", "]"))
	print dict.keys()

def load_vectors():
	import pickle
	path = "/data/s1774395/p_vectors"
	vectors = pickle.load(open(path, "rb"))
	return vectors

def tweet_to_vector(tweet, vectors):
	data = []
	for word in tweet.split(" "):
		try:
			data.append(vectors[word])
		except:
			pass
	return [sum(i) for i in zip(*data)]


from functools import partial
from multiprocessing import Pool
import numpy as np
import pickle

def edit_distance(base_tweet, tweet):
	base_tweet = np.asarray(base_tweet)
	tweet = np.asarray(tweet)
	return abs(np.sum(np.subtract(base_tweet, tweet)))

def dist_mp(base_tweet, tweet):
	return tweet, edit_distance(base_tweet, tweet)


def distances():
	sequence_list = []
	idx = 0
	with open("/data/s1774395/vectors.txt","r") as v_file:
		for line in v_file:
			sequence_list.append(eval(line))
			if idx > 10:
				continue
			idx += 1

	base_tweet = sequence_list[0]
	logger.info(base_tweet)

	results = []

	for s in sequence_list:
		results.append(edit_distance(base_tweet, s))

	#
	# pool = Pool()  # use all CPUs
	# for tweet, d in pool.imap_unordered(partial(dist_mp, base_tweet), sequence_list[1:]):
	# 	results.append(d)
	# pool.close()
	# pool.join()
	pickle.dump(open("/data/s1774395/distances", "rb"), results)


def main():
	import os,sys
	import pickle

	vectors = load_vectors()

	#dname = "/home/cluster/201501/"
	dname = "/data/s1774395/text/2015/01/"
	files = os.listdir(dname)

	vname = "/data/s1774395/vectors.txt"

	idx = 0
	print "starting to read the files"
	for file in files:
		idx = idx + 1
		logger.info(idx)
		v_pass = 0
		for line in open(dname + file):
			# take every tenth of tweet, to decrease the dataset
			if v_pass > 10:
				vector = tweet_to_vector(filter.filter(line), vectors)
				with open(vname, 'a') as vfile:
					vfile.write(str(vector) + "\n")
				v_pass = 0
			else:
				v_pass += 1
				continue

if __name__ == "__main__":
	distances()
