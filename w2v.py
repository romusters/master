import pandas as pd
import numpy as np
import math
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# remove WrappedArray from embeddings from hadoop model
model_name = "/media/cluster/data1/vectors_threshold20_2015model56.csv.clean"
data_name = "/media/cluster/data1/data_sample.csv"
# data = pd.read_csv(data_name + ".clean", header=None, iterator=True, chunksize=1)
# chunk = data.get_chunk()
# print chunk

def clean_data_column(data_name, header=None):
	data = pd.read_csv(data_name, header=header)
	f = lambda vector: eval(vector.replace("WrappedArray(", "[").replace(")", "]"))
	data[1] = data[1].map(f)
	data.to_csv(data_name + ".clean")
	return data


def clean_data():
	data = pd.read_csv(data_name, header=None, iterator=True, chunksize=1)
	idx = 0
	chunk = data.get_chunk()
	# chunk.rename(columns={0: 'words', 1: 'tokens', 2: 'vector', 3: 'cluster'}, inplace=True)
	while chunk is not None:
		if idx % 1000 == 0:
			print idx
		try:
			f = lambda vector: eval(vector.replace("WrappedArray(", "[").replace(")", "]"))
			# chunk = chunk[chunk.vector != 'null']
			tmp = chunk[2] != 'null'
			if tmp[0]:
				clean_vectors = chunk[2].map(f)
				clean_data = chunk
				clean_data[2] = clean_vectors
				# clean_data = clean_data.drop("cluster", 1)
				clean_data.to_csv(data_name + ".clean", index=None, mode="a", header=False)
			chunk = data.get_chunk()
			# chunk.rename(columns={0: 'words', 1: 'tokens', 2: 'vector', 3: 'cluster'}, inplace=True)
			idx += 1
		except StopIteration:
			break
# clean_data()

	#


# chunk_size = 1000
# def predict():
# 	data_name = "/media/cluster/data1/data_sample.csv.clean"
#
# 	data = pd.read_csv(data_name, header=None, iterator=True, chunksize=chunk_size)
# 	chunk = data.get_chunk()
# 	print chunk.keys()
# 	vector_seed = eval(chunk[2][0])
#
# 	idx = 0
# 	sims_name = data_name + ".sims"
# 	while chunk is not None:
# 		print idx
#
# 		try:
# 			sims = chunk[2].apply(lambda x: cosine_similarity(eval(x), vector_seed))
# 			chunk[4] = sims
# 			chunk.to_csv(sims_name, index=None, mode="a", header=None)
# 			chunk = data.get_chunk()
# 		except StopIteration:
# 			break
# 		idx += 1

def test(v1, v2):
	print len(v1), len(v2)
	import sys
	sys.exit(0)

def cosine_similarity(v1,v2):
	import math
	# compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
	sumxx, sumxy, sumyy = 0, 0, 0
	for i in range(len(v1)):
		x = v1[i]; y = v2[i]
		sumxx += x*x
		sumyy += y*y
		sumxy += x*y
	sim = sumxy/math.sqrt(sumxx*sumyy)
	# sim = sumxy/math.sqrt(sumyy)

	return sim

def predict(fname, seed_id):
	logging.info("Start predicting for " + str(seed_id))
	import os.path
	if os.path.isfile("/media/cluster/data1/lambert/lambert_w2v_data_jan_sims_" + str(seed_id) + ".csv"):
		return
	else:
		import pandas as pd
		data = pd.read_hdf(fname, "data", iterator=True, chunksize=1000000)
		store = pd.HDFStore(fname)
		seed = np.array(store.select("data",  where=store["data"].id.isin([seed_id])).drop("id", axis=1).values.tolist())[0]
		print seed
		dim = len(seed)
		count = 0
		for chunk in data:
			print count
			ids = chunk["id"]
			vectors = chunk[range(dim)].values.tolist()
			sims = []
			for row in vectors:
				sim = cosine_similarity(seed, row)
				sims.append(sim)
			sims = pd.DataFrame(sims)
			sims.index=ids.values.tolist()
			sims.to_csv("/media/cluster/data1/lambert/lambert_w2v_data_jan_sims_" + str(seed_id) + ".csv",  mode='a',header=None)
			count += 1



def sort_predicted():
	data = pd.read_csv(data_name + ".clean.sims", header=None, usecols=[4], nrows=10000)
	data = data.sort_values([4], ascending=False)
	return data

def plot_sims(fname, id):
	data = pd.read_csv(fname, header=None)
	data = data.sort_values([1], ascending=False)
	print data.values.tolist()
	from plotly.offline import init_notebook_mode, plot
	init_notebook_mode()
	from plotly.graph_objs import Scatter, Figure, Layout
	trace = Scatter(x=range(data[1].count()), y=np.array(data[1].values.tolist()).ravel(), mode="lines+markers")
	data = [trace]
	layout = Layout(title="Distance of tweet id " + str(id) + " to other tweets", xaxis=dict(title="tweet number"),
					yaxis=dict(title="Cosine similarity"))
	fig = Figure(data=data, layout=layout)
	plot(fig)
	print data
# id = 194
# fname = "/media/cluster/data1/w2v_data_jan_sims_"+ str(id) + ".csv"
# plot_sims(fname, id)

def get_nearest():
	sorted = sort_predicted()
	result = sorted[sorted[4] > 0.5].index.values
	print result
	return result

def get_data_by_index(data, indices):
	print data.loc[indices]

# used to only get the tweet and the id to speed up the tweet selection by id procedure
def data_sample_tweet_id():
	import pandas as pd
	data_name = "/media/cluster/data1/data_sample.csv"

	tweets = pd.read_csv(data_name, header=None, iterator=True, index_col=False, chunksize=1000, usecols=[0, 3],
						 names=["tweet", "id"])
	chunk = tweets.get_chunk()
	while chunk is not None:
		chunk.index = chunk["id"]
		chunk["tweet"].to_csv("/media/cluster/data1/data_sample_tweet_id_emb3971.csv", mode="a")
		chunk = tweets.get_chunk()


def data_sample_vector_id():
	import pandas as pd
	data_name = "/media/cluster/data1/w2v_data_jan.csv"
	chunksize = 1000
	tweets = pd.read_csv(data_name, header=None, iterator=True, index_col=False, chunksize=chunksize, usecols=[2, 3], names=["vector", "id"])
	store = pd.HDFStore(data_name + ".clean.h5")
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







def test_model(fname):
	data = pd.HDFStore(fname)["data"]
	# word = "1"
	# word = "moslim"
	words = data["words"].values.tolist()
	print words[0]
	print word in words
	A = data[data["words"] == word].vectors.values.tolist()[0]
	print A
	sims = data.vectors.apply(lambda x: cosine_similarity(np.array(x), np.array(A)))
	sims.to_csv(fname + ".sims")

	sims.sort(ascending=False)
	els = sims.head(10).keys()

	print els

	for el in els:
		print data.words[el]


def test_similar_tweets(fname, data_name):
	tweet_sims = pd.read_csv(fname + ".sims", header=None)
	# get all the indices, the embeddings will be checked on uniqueness
	yes_idx = tweet_sims[tweet_sims[1] > 0.60][0].values.tolist()
	print len(yes_idx)
	tweets = pd.HDFStore(data_name )["data"]
	for i in range(3):
		print tweets[tweets[0].isin([yes_idx[i]])][1].values.tolist()


def test_embedding_separation():
	#are these tweets exactly the same?
	##
	ids = [103079215880,  515396089413]
	tweets = pd.read_csv("/media/cluster/data1/data_sample_tweet_id.csv", header=None)
	tokens = pd.read_csv("/media/cluster/data1/data_sample_tokens_id.csv", header=None)
	embs_store = pd.HDFStore("/media/cluster/data1/vectors_threshold20_2015model56.csv.clean.h5")
	store = pd.HDFStore("/media/cluster/data1/data_sample_embs_id.h5")
	vocab  = embs_store["table"]["words"].values.tolist()
	for id in ids:
		tweet = tweets[tweets[0].isin([id])]
		tweet_tokens = tokens[tokens[0].isin([id])][1].values.tolist()[0].replace("[", "").replace("]", "").replace(" ", "").split(",")
		print tweet[0]
		print tweet_tokens
		print [token if token in vocab else "*" + token + "*" for token in tweet_tokens]

		embs = store.select("data", where=store["data"].index.isin([id]))
		print embs[range(3)].values.tolist()
		print "\n"



	# are the embs in the tweets the same? yes
	# test_embs = store.select("data", where=store["data"].index.isin(ids))
def test_correct_mean_embedding():
	import numpy as np
	ids = [103079215880, 515396089413]
	tweets = pd.read_csv("/media/cluster/data1/data_sample_tweet_id.csv", header=None)
	tokens = pd.read_csv("/media/cluster/data1/data_sample_tokens_id.csv", header=None)
	embs_store = pd.HDFStore("/media/cluster/data1/vectors_threshold20_2015model56.csv.clean.h5")
	# embs_store = pd.HDFStore("/media/cluster/data1/vectors_jan_threshold20_2015model99.csv.clean.h5")
	store = pd.HDFStore("/media/cluster/data1/data_sample_embs_id.h5")
	vocab = embs_store["table"]["words"].values.tolist()
	mean_vector = []
	for id in ids:
		tweet = tweets[tweets[0].isin([id])]
		print tweet.values.tolist()
		tweet_tokens = tokens[tokens[0].isin([id])][1].values.tolist()[0].replace("[", "").replace("]", "").replace(" ","").split(",")
		for token in tweet_tokens:
			print token
			mean_vector.append(embs_store["table"]["vectors"][embs_store["table"]["words"].isin([token])].values.tolist())
		print np.mean(mean_vector[0], axis=0).tolist()
		mean_vector = []
# test_correct_mean_embedding()

def test_similar_embeddings():
	store = pd.HDFStore("/media/cluster/data1/data_sample_embs_id.h5")


def get_vocab():
	store = pd.HDFStore("/media/cluster/data1/lambert/lambert_model.csv.h5")
	vocab = store['data']["words"].values.tolist()
	return vocab
fname = "/media/cluster/data1/lambert/lambert_model.csv.h5"
# test_model(fname)

tweet_name = "/media/cluster/data1/lambert/w2v_data_jan_tweet_id.csv"
data_name = "/media/cluster/data1/lambert/lambert_w2v_data_jan.csv.clean.h5"
# test_similar_tweets(data_name, tweet_name)

fname = "/media/cluster/data1/lambert/lambert_w2v_data_jan.csv.clean.h5"
# predict(fname, 194)