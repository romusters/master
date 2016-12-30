import pandas as pd

def write_sim_file(word):
	data = pd.read_csv("/media/cluster/data1/lambert/lambert_w2v_data_jan_tweet_id.csv", header=None)

	flags = data[1].str.contains(r"\b%s\b" % word, case=False)
	ids = data[0][flags]

	# get average of tweet with subject
	store = pd.HDFStore("/media/cluster/data1/lambert/data_sample_vector_id.clean.h5")
	spec_vectors = store.select("data", where=store["data"]["id"].isin(ids))
	spec_vectors = spec_vectors[range(70)]
	mean = spec_vectors.mean().values.tolist()

	import w2v
	vectors = store["data"][range(70)].values.tolist()
	sims = []
	for row in vectors:
		sims.append(w2v.cosine_similarity(row, mean))

	res = pd.DataFrame()
	res["id"] = store["data"]["id"]
	res["sim"] = sims
	res.to_csv("/media/cluster/data1/lambert/" + word + ".csv")

def get_sim_data(word):
	data = pd.read_csv("/media/cluster/data1/lambert/" + word + ".csv")
	data.dropna(inplace=True)
	# data = data.sort("sim", axis=0, ascending=False)
	# ids = data.where(data.sim>0.5)
	# ids = ids[ids.sim.notnull()].id
	# return ids
	return data

def plot_sim_data():
	from plotly.offline import init_notebook_mode, plot
	init_notebook_mode()
	from plotly.graph_objs import Scatter, Figure, Layout
	trace = Scatter(x=range(len(sims)), y=sims)
	data = [trace]
	layout = Layout(title="Similarities for word " + word, xaxis=dict(title="n tweets"),
					yaxis=dict(title="Cosine similarity"))
	fig = Figure(data=data, layout=layout)
	plot(fig)


def show_sample_tweet(ids):
	sample = ids.sample(20).values.tolist()
	data = pd.read_csv("/media/cluster/data1/lambert/lambert_w2v_data_jan_tweet_id.csv", header=None)
	tweets = data[data[0].isin(sample)]
	for tweet in tweets[1]:
		print tweet

def count_wordnotin(ids):
	data = pd.read_csv("/media/cluster/data1/lambert/lambert_w2v_data_jan_tweet_id.csv", header=None)
	tweets = data[data[0].isin(ids)]
	flags = tweets[1].str.contains(r"\b%s\b" % word, case=False)

def compare_all():
	import w2v
	store = pd.HDFStore("/media/cluster/data1/lambert/data_sample_vector_id.clean.h5")
	vectors = store.select("data")[range(70)].values.tolist()
	idx = 0
	for seed in vectors:
		print idx
		for vector in vectors:
			w2v.cosine_similarity(seed, vector)
		idx = idx + 1


# def sims_gradient(word):
# 	data = pd.read_csv("/media/cluster/data1/lambert/" + word + ".csv")
# 	data.dropna(inplace=True)
# 	data = data.sort("sim", axis=0, ascending=False)
# 	data = data.where(data.sim>0.5).values.tolist()
# 	dx = 5
# 	diff = data.diff(dx)/(dx+1)


def get_tweet_blocks():
	data = pd.read_csv("/media/cluster/data1/lambert/" + word + ".csv")
	all_tweets = pd.read_csv("/media/cluster/data1/lambert/lambert_w2v_data_jan_tweet_id.csv", header=None)
	block = [1.0, 0.6,  0, -0.5, -0.9]
	for i, b in enumerate(block):
		tmp_data = data[(data.sim > block[i] -0.1) & (data.sim < block[i])]
		try:
			ids = tmp_data.sample(20).id
		except ValueError:
			ids = tmp_data.id
		tweets = all_tweets[all_tweets[0].isin(ids)][1]
		for tweet in tweets:
			print tweet

		print "done"

def annotate(word):
	data = pd.read_csv("/media/cluster/data1/lambert/" + word + ".csv")
	all_tweets = pd.read_csv("/media/cluster/data1/lambert/lambert_w2v_data_jan_tweet_id.csv", header=None)
	block = [1.0, 0.6, 0, -0.5, -0.9]

	for b in block:
		tmp_data = data[(data.sim > b - 0.1) & (data.sim < b)]
		try:
			ids = tmp_data.sample(10).id
		except ValueError:
			ids = tmp_data.id
		tweets = all_tweets[all_tweets[0].isin(ids)][1]
		for i,tweet in enumerate(tweets):
			print tweet
			g = open("/media/cluster/data1/lambert/annotated", "a")
			x = raw_input()
			g.write(str(ids.iloc[i]) + "," + x + "," + word+ "," + str(b) + "\n")
			g.close()


def nn():
	data = pd.read_csv("/media/cluster/data1/lambert/annotated", header=None, names=["id", "label", "subject"])
	store = pd.HDFStore("/media/cluster/data1/lambert/data_sample_vector_id.clean.h5")
	vectors = store.select("data",  where=store["data"].id.isin(data["id"])).drop("id", axis=1).values.tolist()
	tmp_labels = data["label"].values.tolist()
	labels = []
	for l in tmp_labels:
		if l == 1:
			labels.append([0,1])
		else:
			labels.append([1,0])
	import tensorflow as tf
	sess = tf.InteractiveSession()

	n_classes = 2
	dim = 70
	x = tf.placeholder(tf.float32, shape=[None, dim], name="Input")
	y_ = tf.placeholder(tf.float32, shape=[None, n_classes], name="Output")
	W = tf.Variable(tf.zeros([dim, n_classes]))
	b = tf.Variable(tf.zeros([n_classes]))
	sess.run(tf.initialize_all_variables())

	y = tf.matmul(x, W) + b

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	import numpy as np
	correct_prediction.eval(feed_dict={x: vectors, y_: labels})

	for i in range(3000):
		train_step.run(feed_dict={x: np.array(vectors), y_: np.array(labels)})
		# train_step.run(feed_dict={x: batch[2].values.tolist(), y_: batch[5].values.tolist()})
		print(accuracy.eval(feed_dict={x: vectors, y_: labels}))

# which tweets are similar to average of previous?
word = "voetbal"
# write_sim_file(word)
# ids = get_sim_data(word)
#
#
# show_sample_tweet(ids)
# count_wordnotin(ids)
# plot_sim_data(sims)
# compare_all()
# get_tweet_blocks()

# annotate(word)
nn()
