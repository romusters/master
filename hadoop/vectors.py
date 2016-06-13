
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
	path = "/home/cluster/p_vectors"
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




def main():
	import os,sys
	import filter
	import pickle

	vectors = load_vectors()

	#dname = "/home/cluster/201501/"
	dname = "/data/s1774395/text/2015/01"
	files = os.listdir(dname)

	vname = "/data/s1774395/vectors.txt"

	idx = 0
	for file in files:
		idx = idx + 1
		print idx
		for line in open(dname + file):
			vector = tweet_to_vector(filter.filter(line), vectors)
			with open(vname, 'a') as vfile:
				vfile.write(str(vector))

if __name__ == "__main__":
	main()