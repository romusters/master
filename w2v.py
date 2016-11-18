import pandas as pd
import numpy as np
import math

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
def cosine_similarity(v1,v2):
	# compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
	sumxx, sumxy, sumyy = 0, 0, 0
	for i in range(len(v1)):
		x = v1[i]; y = v2[i]
		sumxx += x*x
		sumyy += y*y
		sumxy += x*y
	return sumxy/math.sqrt(sumxx*sumyy)
	#return sumxy/math.sqrt(sumyy)

chunk_size = 1000
def predict():
	data_name = "/media/cluster/data1/data_sample.csv.clean"

	data = pd.read_csv(data_name, header=None, iterator=True, chunksize=chunk_size)
	chunk = data.get_chunk()
	print chunk.keys()
	vector_seed = eval(chunk[2][0])

	idx = 0
	sims_name = data_name + ".sims"
	while chunk is not None:
		print idx

		try:
			sims = chunk[2].apply(lambda x: cosine_similarity(eval(x), vector_seed))
			chunk[4] = sims
			chunk.to_csv(sims_name, index=None, mode="a", header=None)
			chunk = data.get_chunk()
		except StopIteration:
			break
		idx += 1


def sort_predicted():
	data = pd.read_csv(data_name + ".clean.sims", header=None, usecols=[4], nrows=10000)
	data = data.sort_values([4], ascending=False)
	return data

def plot_sort_predicted(data):
	from plotly.offline import init_notebook_mode, plot
	init_notebook_mode()
	from plotly.graph_objs import Scatter, Figure, Layout
	trace = Scatter(x=range(data.count()), y=np.array(data.values.tolist()).ravel(), mode="lines+markers")
	data = [trace]
	layout = Layout(title="", xaxis=dict(title=""),
					yaxis=dict(title=""))
	fig = Figure(data=data, layout=layout)
	plot(fig)
	print data


def get_nearest():
	sorted = sort_predicted()
	result = sorted[sorted[4] > 0.5].index.values
	print result
	return result

def get_data_by_index(data, indices):
	print data.loc[indices]



def test():
	data_name = "/media/cluster/data1/test.csv"
	f = open(data_name + ".clean", "a")
	data = pd.read_csv(data_name, header=None, iterator=True, chunksize=1)
	chunk = data.get_chunk().drop(3, 1)
	while chunk is not None:
		try:
			tmp = ";".join(chunk.values.tolist()[0])
			print tmp
			f.write(tmp + "\n")
			chunk = data.get_chunk().drop(3, 1)
		except StopIteration:
			break
	f.close()
# test()

def test2():
	data_name = "/media/cluster/data1/test.csv"
	f = open(data_name + "2.clean", "a")
	data = pd.read_csv(data_name, header=None, iterator=True, chunksize=2)
	chunk = data.get_chunk().drop(3, 1)
	while chunk is not None:
		try:
			tmp = ";".join(chunk.values.tolist()[0])
			print tmp
			f.write(tmp + "\n")
			chunk = data.get_chunk().drop(3, 1)
		except StopIteration:
			break
	f.close()
# test2()

# X = pd.read_csv(fname, header=None)
# f = lambda vector: eval(vector.replace("WrappedArray(", "[").replace(")", "]"))
# clean_vectors = X[1].map(f)
#
# X[1] = clean_vectors
# X.columns = ["words", "vectors"]
# print X
# X.to_csv(fname + ".clean", index=None)

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

tweets = pd.read_csv(data_name, header=None, iterator=True, index_col=False, chunksize=10, usecols=[2, 3], names=["vector", "id"])
store = pd.HDFStore(data_name + ".clean.h5")
chunk = tweets.get_chunk()
idx = 0
while chunk is not None:
	if idx > 2:
		store.close()
		break
	print idx
	# chunk.index = chunk["id"]
	chunk["vector"] = chunk["vector"].apply(lambda x: eval(x.replace("WrappedArray(", "[").replace(")", "]").replace("null", "None")))
	# store.append("data", chunk["vector"])
	print "Amount of None is: ", 1000-chunk["vector"].dropna().count()
	vector = chunk["vector"].apply(pd.Series, 1)
	id = chunk["id"].to_frame()
	vector["id"] = id#.to_hdf(data_name + ".clean.h5", "data")
	store.append("data", vector)
	chunk = tweets.get_chunk()
	idx+=1
store.close()


def data_sample_tokens_id():
	import pandas as pd
	data_name = "/media/cluster/data1/data_sample.csv"

	tweets = pd.read_csv(data_name, header=None, iterator=True, index_col=False, chunksize=1000, usecols=[1, 3],
						 names=["token", "id"])
	chunk = tweets.get_chunk()
	while chunk is not None:
		f = lambda vector: vector.replace("WrappedArray(", "[").replace(")", "]")
		chunk["token"] = chunk["token"].apply(f)
		chunk.index = chunk["id"]
		chunk["token"].to_csv("/media/cluster/data1/data_sample_tokens_id.csv", mode="a")
		chunk = tweets.get_chunk()


def AL():
	# jan = False
	# f = lambda x: eval(x)
	# dim = 0
	# if jan:
	# 	emb_name = "/media/cluster/data1/vectors_jan_threshold20_2015model99.csv.clean"
	# 	# embs = pd.read_csv(emb_name)
	# 	# embs[1] = embs[1].apply(f)
	# 	# dim = len(embs[1][0])
	# else:
	# 	emb_name = "/media/cluster/data1/vectors_threshold20_2015model56.csv.clean"
	# 	embs = pd.read_csv(emb_name)
	# 	embs["vectors"] = embs["vectors"].apply(f)
	# 	dim = len(embs["vectors"][0])
	# print dim
	# print embs.keys()

	emb_name = "/media/cluster/data1/vectors_threshold20_2015model56.csv.clean.h5"
	# get the tweet ids with the highest similarities and ...
	data_name = "/media/cluster/data1/data_sample.csv"
	# data = pd.read_csv(data_name + ".clean.sims", header=None, nrows=10000)
	# sorted = sort_predicted()
	# yes = data.loc[sorted[sorted[4] > 0.5].index.values.tolist()].sample(20)
	# yes[2] = yes[2].apply(lambda x: eval(x))
	# yes[5] = pd.DataFrame(data=[dim*[[0,1]]]).T
	#
	# no = data.loc[sorted[(sorted[4] < 0.5) & (sorted[4] > 0.0)].index.values.tolist()].sample(20)
	# no[2] = no[2].apply(lambda x: eval(x))
	# no[5] = pd.DataFrame(data=[dim*[[1,0]]]).T
	# batch = yes.append(no)


	#get the tweet with similarities
	tweet_sims = pd.read_csv("/media/cluster/data1/data_sample.sims")
	# get all the indices, the embeddings will be checked on uniqueness
	yes_idx = tweet_sims[tweet_sims["1"] > 0.60]["0"].values.tolist()
	print len(yes_idx)
	# get the embedding belonging to ids
	store = pd.HDFStore("/media/cluster/data1/data_sample_embs_id.h5")
	yes_embs = store.select("data", where=store["data"].index.isin(yes_idx))
	print yes_embs.shape[0]
	try:
		yes_embs = yes_embs.drop_duplicates().sample(20)
	except:
		yes_embs = yes_embs.drop_duplicates()[0:20]
	print yes_embs.shape[0]
	tweets = pd.read_csv("/media/cluster/data1/data_sample_tweet_id.csv", header=None)
	yes_tweets = tweets[tweets[0].isin(yes_idx)].sample(20)
	tokens = pd.read_csv("/media/cluster/data1/data_sample_tokens_id.csv", header=None)



	# check if dimension of model and data are the same
	model_store = pd.HDFStore("/media/cluster/data1/vectors_threshold20_2015model56.csv.clean.h5")
	dim_model = len(yes_embs["table"]["vectors"][0])
	dim_data = yes_embs.shape[1]
	assert(dim_model == dim_data)
	dim = yes_embs.shape[0]

	yes_embs = yes_embs.values.tolist()
	data = yes_embs
	labels = []
	labels.extend([[0,1]]*len(yes_embs))
	# yes.append(pd.DataFrame(data=[dim * [[0, 1]]]).T)

	no_idx = tweet_sims[tweet_sims["1"] < 0.5]["0"].values.tolist()
	no_embs = store.select("data", where=store["data"].index.isin(no_idx))
	try:
		no_embs = no_embs.drop_duplicates().sample(20)
	except:
		no_embs = no_embs.drop_duplicates()[0:20]
		no_embs = no_embs.values.tolist()
		labels.extend([[1, 0]] * len(no_embs))
	data.extend(no_embs)
	# no.append(pd.DataFrame(data=[dim * [[1, 0]]]).T)
	# batch = yes.append(no)

	import tensorflow as tf
	sess = tf.InteractiveSession()

	n_classes = 2
	x = tf.placeholder(tf.float32, shape=[None, dim], name="Input")
	y_ = tf.placeholder(tf.float32, shape=[None, n_classes], name="Output")
	W = tf.Variable(tf.zeros([dim, n_classes]))
	b = tf.Variable(tf.zeros([n_classes]))
	sess.run(tf.initialize_all_variables())

	y = tf.matmul(x, W) + b

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	correct_prediction.eval(feed_dict={x: data, y_: labels})

	for i in range(3):
		train_step.run(feed_dict={x: data, y_: labels})
			# train_step.run(feed_dict={x: batch[2].values.tolist(), y_: batch[5].values.tolist()})
		print(accuracy.eval(feed_dict={x: data, y_: labels}))


def main():
	data = pd.read_csv(fname)

	A = np.array(eval(data[data["words"] == "moslim"].vectors.values.tolist()[0]))
	sims = data.vectors.apply(lambda x: cosine_similarity(eval(x), A))


	sims.sort(ascending=False)
	els = sims.head(10).keys()

	print els

	for el in els:
		print data.words[el]

	# B = np.array(eval(data[data["words"] == "drie"].vectors.values.tolist()[0]))
	# print A
	# print B
	# from scipy import spatial
	#
	# result = 1 - spatial.distance.cosine(A, B)
	# print result

#
# print cosine_similarity(A, B)

def test_similar_tweets():
	tweet_sims = pd.read_csv("/media/cluster/data1/data_sample.sims")
	# get all the indices, the embeddings will be checked on uniqueness
	yes_idx = tweet_sims[tweet_sims["1"] > 0.60]["0"].values.tolist()
	print len(yes_idx)
	# get the embedding belonging to ids
	store = pd.HDFStore("/media/cluster/data1/data_sample_embs_id.h5")
	yes_embs = store.select("data", where=store["data"].index.isin(yes_idx))
	tweets = pd.read_csv("/media/cluster/data1/data_sample_tweet_id.csv", header=None)
	for i in range(3):
		print tweets[tweets[0].isin([yes_idx[i]])][1].values.tolist()
# test_similar_tweets()

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
