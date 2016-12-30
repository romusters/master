def data_sample_tweet_id(vector_name, tweet_name):
	import pandas as pd
	tweets = pd.read_csv(vector_name, header=None, iterator=True, chunksize=1000, usecols=[0, 1])
	chunk = tweets.get_chunk()
	while chunk is not None:
		chunk.index = chunk[0]
		chunk[1].to_csv(tweet_name, mode="a")
		chunk = tweets.get_chunk()

# when collecting the lookup table from hadoop through the terminal, we get a dataframe which we convert to dict: dict(df)
def dict_to_csv(dict, fname):
	import csv
	with open(fname, 'wb') as f:
		w = csv.writer(f)
		w.writerows(dict.items())


def convert_lookup_to_hdf(fname):
	import pandas as pd
	data = pd.read_csv(fname, names=["words", "vectors"])
	data["vectors"] = data["vectors"].apply(lambda x: eval(x))
	data.to_hdf(fname + ".h5", "data")


def data_sample_vector_id_2(data_name, vector_name):
	import pandas as pd
	chunksize = 100000
	tweets = pd.read_csv(data_name, header=None, iterator=True, index_col=False, chunksize=chunksize, usecols=[1,2], names=["vector", "id"])
	store = pd.HDFStore(vector_name + ".clean.h5")
	chunk = tweets.get_chunk()
	idx = 0
	while chunk is not None:
		print idx
		chunk["vector"] = chunk["vector"].apply(lambda x: eval(x.replace("WrappedArray(", "[").replace(")", "]").replace("null", "None")))
		print "Amount of None is: ", chunksize-chunk["vector"].dropna().count()
		vector = chunk["vector"].apply(pd.Series, 1)
		id = chunk["id"].to_frame()
		vector["id"] = id
		store.append("data", vector)
		chunk = tweets.get_chunk()
		idx+=1
	store.close()

def data_sample_vector_id(data_name, vector_name):
	import pandas as pd
	chunksize = 100000
	tweets = pd.read_csv(data_name, header=None, iterator=True, index_col=False, chunksize=chunksize, usecols=[2,3], names=["vector", "id"])
	store = pd.HDFStore(vector_name + ".clean.h5")
	chunk = tweets.get_chunk()
	idx = 0
	while chunk is not None:
		print idx
		chunk["vector"] = chunk["vector"].apply(lambda x: eval(x.replace("WrappedArray(", "[").replace(")", "]").replace("null", "None")))
		print "Amount of None is: ", chunksize-chunk["vector"].dropna().count()
		vector = chunk["vector"].apply(pd.Series, 1)
		id = chunk["id"].to_frame()
		vector["id"] = id
		store.append("data", vector)
		chunk = tweets.get_chunk()
		idx+=1
	store.close()

def data_sample_tokens_id2(data_name):
	import pandas as pd
	tweets = pd.read_csv(data_name, header=None, iterator=True, index_col=False, chunksize=1000, usecols=[0,2], names=["token", "id"])
	chunk = tweets.get_chunk()
	while chunk is not None:
		chunk["token"] = chunk["token"].apply(lambda x:x.replace("WrappedArray(", "[").replace(")", "]").replace("null", "None"))
		chunk.index = chunk["id"]
		chunk["token"].to_csv("/media/cluster/data1/lambert/data_sample_tokens_id.csv", mode="a")
		chunk = tweets.get_chunk()

def data_sample_tokens_id(data_name):
	import pandas as pd
	tweets = pd.read_csv(data_name, header=None, iterator=True, index_col=False, chunksize=1000, usecols=[0,2],
						 names=["id", "token"])
	chunk = tweets.get_chunk()
	while chunk is not None:
		chunk.index = chunk["id"]
		chunk["token"].to_csv("/media/cluster/data1/lambert/data_sample_tokens_id.csv", mode="a")
		chunk = tweets.get_chunk()


def rm_wrappedarray():
	import pandas as pd
	data_name = "/media/cluster/data1/lambert/voetbal_moslim_vectors.csv"
	data = pd.read_csv(data_name, header=None)
	data["vectors"]  = data[0].apply(lambda x: eval(x.replace("WrappedArray(", "[").replace(")", "]")))
	return data


def find_subject_tweets(word):
	import pandas as pd
	import re
	data_name = "/media/cluster/data1/lambert/data_sample_tokens_id.csv"
	data = pd.read_csv(data_name, header=None, names=[ "id", "tokens"])

	data =data[data["tokens"].str.contains(word)]
	ids = data["id"].values.tolist()
	print len(ids)
	cluster_name = "/media/cluster/data1/lambert/cluster_id.csv"
	cluster = pd.read_csv(cluster_name, header=None, names=["cluster", "id"])
	# which cluster has the max number of word occurrences
	c_ids = pd.merge(data, cluster, on="id")
	print c_ids["cluster"].value_counts().index.tolist()[0:2]
	return c_ids["cluster"].value_counts().index.tolist()[0]

def show_cluster_tweets(cluster_id):
	import pandas as pd
	cluster_name = "/media/cluster/data1/lambert/cluster_id.csv"
	cluster = pd.read_csv(cluster_name, header=None, names=["cluster", "id"])
	tweet_ids = cluster[cluster["cluster"] == cluster_id].drop_duplicates()
	print tweet_ids
	tweet_name = "/media/cluster/data1/lambert/lambert_w2v_data_jan_tweet_id.csv"
	# tweet_name = "/media/cluster/data1/lambert/data_sample_tokens_id.csv"

	tweet_data = pd.read_csv(tweet_name, header=None, names=["id", "text"])
	print tweet_data
	res = pd.merge(tweet_data, tweet_ids)
	for tweet in res["text"]:
		if "voetbal" not in tweet:
			print tweet

def show_cluster_homogenity():
	import pandas as pd
	cluster_name = "/media/cluster/data1/lambert/cluster_id.csv"
	cluster_data = pd.read_csv(cluster_name, header=None, names=["cluster", "id"])
	vector_name = "/media/cluster/data1/lambert/data_sample_vector_id"
	store = pd.HDFStore(vector_name + ".clean.h5")
	vector_data = store["data"]
	hopkins_list = []
	for i in range(520):
		print i
		# get the ids for each cluster
		ids = cluster_data[cluster_data.cluster == i]
		# get the vectors for each cluster
		data = pd.merge(ids, vector_data, on="id")[range(70)]
		# Calculate Hopkins for each cluster
		import hopkins
		import numpy as np
		hop = hopkins.hopkins(np.array(data.values.tolist()), i)
		print hop["hopkins"]
		hopkins_list.append(hop["hopkins"])
		print
	print hopkins_list


def input_cluster_topics():
	g = open("/media/cluster/data1/lambert/topics", "a")
	for i in range(500):
		show_cluster_tweets(i)
		x = raw_input()
		g = open("/media/cluster/data1/lambert/topics", "a")
		g.write(str(i) + "," + x + "\n")
		g.close()

def input_cluster_topics_hopkins():
	f = open("/media/cluster/data1/lambert/hopkins_result")
	g = open("/media/cluster/data1/lambert/topics", "a")
	g.writelines("cluster, topic\n")
	g.close()
	hopkins = eval(f.read())
	hopkins.reverse()
	hopkins =hopkins[32:500]
	for h in hopkins:
		show_cluster_tweets(h)
		x = raw_input()
		g = open("/media/cluster/data1/lambert/topics", "a")
		g.write(str(h) + "," + x + "\n")
		g.close()

if __name__ == "__main__":

	data_name = "/media/cluster/data1/lambert/lambert_w2v_data_jan.csv"
	vector_name = "/media/cluster/data1/lambert/data_sample_vector_id"
	# data_name = "/media/cluster/data1/lambert/lambert_w2v_data_jan_all_columns.csv"
	# vector_name = "/media/cluster/data1/lambert/data_sample_vector_id"
	# data_sample_vector_id(data_name, vector_name)
	# data_sample_vector_id_2(data_name, vector_name)

	tweet_name = "/media/cluster/data1/lambert/lambert_w2v_data_jan_tweet_id.csv"
	# data_sample_tweet_id(data_name, tweet_name)
	# data_sample_tokens_id(data_name)
	# data_sample_tokens_id2(data_name)
	model_name = "/media/cluster/data1/lambert/lambert_model.csv"
	# convert_lookup_to_hdf(model_name)

	word = "vegan"
	# cluster_id = find_subject_tweets(word)
	cluster_id = 224
	# show_cluster_tweets(cluster_id)

	# show_cluster_homogenity()
	input_cluster_topics()