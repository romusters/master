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

	sims = pd.read_csv("/media/cluster/data1/lambert/" + word + ".csv")
	# threshold = 0.5
	# yes_ids = sims[sims.sim > threshold+0.2].id
	# no_ids = sims[sims.sim <= threshold-0.2].id
	#
	# yes_data = store.select("data",  where=store["data"].id.isin(yes_ids)).drop("id", axis=1).values.tolist()
	# no_data = store.select("data",  where=store["data"].id.isin(no_ids)).drop("id", axis=1).values.tolist()
	# train_data = yes_data + no_data

	vectors = store["data"]
	res = pd.merge(vectors, sims, on="id")
	res.sim = res.sim.apply(lambda x: 0.5*(x+1))
	train_data = res[range(70)]
	labels = res.sim.values.tolist()
	ids = res.id.values.tolist()
	# print ids
	train_labels = [[1-x, x] for x in labels]

	# train_labels = [[0,1]] * yes_ids.count()
	# train_labels.extend([[1,0]] * no_ids.count())

	test_data = store.select("data",  where=store["data"].id.isin(data["id"])).drop("id", axis=1).values.tolist()
	tmp_labels = data["label"].values.tolist()
	test_labels = []
	for l in tmp_labels:
		if l == 1:
			test_labels.append([0,1.0])
		else:
			test_labels.append([1.0,0])
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

	# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)

	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	import numpy as np
	# correct_prediction.eval(feed_dict={x: test_data, y_: test_labels})

	exp_data = pd.read_csv("/media/cluster/data1/lambert/first_iter.csv")
	train_data = exp_data["data"].apply(lambda x: eval(x)).values.tolist()
	train_labels = exp_data["labels"].apply(lambda x: eval(x)).values.tolist()

	acc_list = []
	for i in range(100):
		train_step.run(feed_dict={x: np.array(train_data), y_: np.array(train_labels)})
		# train_step.run(feed_dict={x: batch[2].values.tolist(), y_: batch[5].values.tolist()})
		acc = accuracy.eval(feed_dict={x: test_data, y_: test_labels})
		print(acc)
		acc_list.append(acc)

	print acc_list

	# print correct_prediction.eval(feed_dict={x: train_data, y_: train_labels})
	prediction = tf.argmax(y,1)
	predictions = prediction.eval(feed_dict={x: train_data})
	print predictions


	#
	# probs = y
	# probs = probs.eval(feed_dict={x: train_data})
	# file = open("/media/cluster/data1/lambert/probs", "wb")
	# import pickle
	# pickle.dump(probs, file)
	# # the indices which have value 1
	# indices = [i for i, x in enumerate(predictions) if x == 1]
	# # the ids which have value 1
	# ids = []
	# for i in indices:
	# 	ids.append(i)
	# print ids
	# print len(ids)
	#
	# all_tweets = pd.read_csv("/media/cluster/data1/lambert/lambert_w2v_data_jan_tweet_id.csv", header=None)
	# tweets = all_tweets[all_tweets[0].isin(ids)][1][0:10]
	# for tweet in tweets:
	# 	print tweet


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


def get_training_dataset():
	calc_sims()

def calc_sims():
	annotated_data = pd.read_csv("/media/cluster/data1/lambert/annotated", header=None,
								 names=["id", "label", "subject"])
	store = pd.HDFStore("/media/cluster/data1/lambert/data_sample_vector_id.clean.h5")
	seed_ids = annotated_data.id
	import os
	for seed_id in seed_ids:
		if os.path.isfile("/media/cluster/data1/lambert/" + str(seed_id) + ".csv"):
			continue
		seed_vector = store.select("data", where=store["data"].id.isin([seed_id])).drop("id", axis=1).values.tolist()[0]

		import w2v
		vectors = store["data"][range(70)].values.tolist()
		sims = []
		for row in vectors:
			sims.append(w2v.cosine_similarity(row, seed_vector))

		res = pd.DataFrame()
		res["id"] = store["data"]["id"]
		res["sim"] = sims
		res.to_csv("/media/cluster/data1/lambert/" + str(seed_id) + ".csv")

def expand_annotated():
	annotated_data = pd.read_csv("/media/cluster/data1/lambert/annotated", header=None,
								 names=["id", "label", "subject"])
	store = pd.HDFStore("/media/cluster/data1/lambert/data_sample_vector_id.clean.h5")
	seed_ids = annotated_data.id
	seed_labels = annotated_data.label
	training_ids = []
	training_data = []
	training_labels = []
	for seed_id, seed_label in zip(seed_ids, seed_labels):
		sims_data = pd.read_csv("/media/cluster/data1/lambert/sims/" + str(seed_id) + ".csv")
		ids = sims_data[sims_data.sim > 0.9].id
		vectors = store.select("data",  where=store["data"].id.isin(ids))[range(70)].values.tolist()
		print len(vectors)
		if seed_label == 1:
			training_labels.extend([[0,1]]*len(vectors))
		else:
			training_labels.extend([[1,0]]*len(vectors))
		training_data.extend(vectors)
		training_ids.extend(ids)

	df = pd.DataFrame({"id": training_ids, "data": training_data, "labels": training_labels})
	df.to_csv("/media/cluster/data1/lambert/first_iter.csv")



def show_tweet_id():
	id = 1314260047264
	sims_data = pd.read_csv("/media/cluster/data1/lambert/" + str(id) + ".csv")
	pos_ids = sims_data[sims_data.sim > 0.8].id
	all_tweets = pd.read_csv("/media/cluster/data1/lambert/lambert_w2v_data_jan_tweet_id.csv", header=None)
	tweets = all_tweets[all_tweets[0].isin(pos_ids)][1]
	for tweet in tweets:
		print tweet


def show_expanded_tweets():
	data = pd.read_csv("/media/cluster/data1/lambert/first_iter.csv")
	ids = data["id"]
	all_tweets = pd.read_csv("/media/cluster/data1/lambert/lambert_w2v_data_jan_tweet_id.csv", header=None)
	tweets = all_tweets[all_tweets[0].isin(ids)][1][0:1000]
	for tweet in tweets:
		print tweet


def test():
	data = pd.read_csv("/media/cluster/data1/lambert/voetbal.csv")
	threshold = 0.7
	ids = data[(data.sim > threshold) & (data.sim < threshold + 1)].id
	print len(ids)
	all_tweets = pd.read_csv("/media/cluster/data1/lambert/lambert_w2v_data_jan_tweet_id.csv", header=None)
	tweets = all_tweets[all_tweets[0].isin(ids)][1][0:10000]
	for tweet in tweets:
		print tweet

def test2():
	data = pd.read_csv("/media/cluster/data1/lambert/annotated", header=None)
	all_tweets = pd.read_csv("/media/cluster/data1/lambert/lambert_w2v_data_jan_tweet_id.csv", header=None)
	for id, label in zip(data[0], data[1]):
		annotated_tweet = pd.read_csv("/media/cluster/data1/lambert/sims/"+ str(id) + ".csv")
		ids = annotated_tweet[annotated_tweet.sim > 0.9].id
		print ids.count()
		print id
		tweets = all_tweets[all_tweets[0].isin(ids)][1]

		with open("/media/cluster/data1/lambert/expanded_tweets/" + str(label) + "/" + str(id) + ".csv", "wb") as file:
			for tweet in tweets:
				print tweet
				file.write(tweet + "\n")

def test_predictions():
	import pickle
	fname = "/media/cluster/data1/lambert/probs"
	file = open(fname)
	data = pickle.load(file)
	yes = []
	no = []
	diff = []
	# predictions for training data
	for a,b in data:
		yes.append(b)
		no.append(a)
		diff.append(abs(abs(a) - abs(b)))
	list_id = yes.index(max(yes))

	import numpy as np
	diff = np.convolve(yes, np.ones((2000,))/2000)

	import pandas as pd
	from plotly.offline import init_notebook_mode, plot

	init_notebook_mode()
	from plotly.graph_objs import Scatter, Figure, Layout
	# yes.sort()
	trace = Scatter(x=range(len(yes)), y=yes)

	data = [trace]
	layout = Layout(title="",
					yaxis=dict(title=""))
	fig = Figure(data=data, layout=layout)
	# plot(fig)


	import numpy as np
	# indices of the list of training data
	sorted_ids = np.array(yes).argsort()

	# reverse the list because we dont want the smallest but the largest differences.
	sorted_ids = list(reversed(sorted_ids))

	# get training data ids as list so the previous indices can be used to select the correct tweets
	exp_data = pd.read_csv("/media/cluster/data1/lambert/first_iter.csv")
	ids = exp_data.id.values.tolist()
	df = pd.DataFrame({"id": ids, "prob": yes})
	print df[0:20]
	df = df.sort(columns=["prob"], ascending=False)
	print df[0:20]

	# now get use ....
	new_ids = [ids[x] for x in sorted_ids]
	print yes[0]
	# print new_ids
	import pandas as pd



	all_tweets = pd.read_csv("/media/cluster/data1/lambert/lambert_w2v_data_jan_tweet_id.csv", header=None, names=["id", "text"])
	# print all_tweets[all_tweets["id"] == new_ids[0]]["text"].values.tolist()[0]
	all_tweets.index = all_tweets["id"]

	# tmp = pd.merge(pd.DataFrame({"id": new_ids}), all_tweets, on="id")
	tmp = pd.merge(pd.DataFrame({"id": df.id[df.prob > -1.0]}), all_tweets, on="id")

	trace = Scatter(x=range(1900), y=df.prob[df.prob > -1.0][0:1900])

	data = [trace]
	layout = Layout(title="",
					yaxis=dict(title=""))
	fig = Figure(data=data, layout=layout)
	plot(fig)

	for id, text in zip(tmp["id"], tmp["text"]):
		print id, text
	# print df.count()
	# print df.prob.iloc[1880]
	# ids = df.id.iloc[0:1880]
	# ids.to_csv("/media/cluster/data1/lambert/second_iter.csv")

def second_nn():
	data = pd.read_csv("/media/cluster/data1/lambert/annotated", header=None, names=["id", "label", "subject"])
	store = pd.HDFStore("/media/cluster/data1/lambert/data_sample_vector_id.clean.h5")

	second_iter = pd.read_csv("/media/cluster/data1/lambert/second_iter.csv", names=["tmp", "id"])
	all_ids = store["data"]["id"]
	second_ids = second_iter["id"]
	no_ids = all_ids[~all_ids.isin(second_ids)]
	yes_vectors = store.select("data", where=store["data"].id.isin(second_ids)).drop("id", axis=1).values.tolist()
	no_vectors = store.select("data", where=store["data"].id.isin(no_ids)).drop("id", axis=1).values.tolist()

	train_data = []
	train_labels = []
	for e in yes_vectors:
		train_data.append(e)
		train_labels.append([0, 1.0])
	for e in no_vectors:
		train_data.append(e)
		train_labels.append([1.0, 0])

	exp_data = pd.read_csv("/media/cluster/data1/lambert/first_iter.csv")
	train_data.extend(exp_data["data"].apply(lambda x: eval(x)).values.tolist())
	train_labels.extend(exp_data["labels"].apply(lambda x: eval(x)).values.tolist())

	print train_data[0]
	print train_labels[0]



	test_data = store.select("data", where=store["data"].id.isin(data["id"])).drop("id", axis=1).values.tolist()
	tmp_labels = data["label"].values.tolist()
	test_labels = []
	for l in tmp_labels:
		if l == 1:
			test_labels.append([0, 1.0])
		else:
			test_labels.append([1.0, 0])
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


	train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)

	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	import numpy as np


	acc_list = []
	iter = 100
	size = len(train_data)/iter
	print size
	for i in range(iter):
		print i
		train_step.run(feed_dict={x: np.array(train_data),
								  y_: np.array(train_labels)})

		# train_step.run(feed_dict={x: np.array(train_data[i*size : (i+1)*size]), y_: np.array(train_labels[i*size : (i+1)*size])})
		# # train_step.run(feed_dict={x: batch[2].values.tolist(), y_: batch[5].values.tolist()})
		acc = accuracy.eval(feed_dict={x: test_data, y_: test_labels})
		print(acc)


# get all the tweets containing voetbal as traininset and test them
def test_nn():
	all_tweets = pd.read_csv("/media/cluster/data1/lambert/lambert_w2v_data_jan_tweet_id.csv", names=["id", "text"])
	print "%i tweets read" % all_tweets.count()[0]

	voetbal = all_tweets[all_tweets.text.str.contains("voetbal")]
	print "%i tweets contain voetbal" % voetbal.count()[0]

	store = pd.HDFStore("/media/cluster/data1/lambert/data_sample_vector_id.clean.h5")
	all_ids = store["data"]["id"]
	no_ids = all_ids[~all_ids.isin(voetbal.id)]
	print "%i tweets do not contain voetbal" % no_ids.count()
	yes_vectors = store.select("data", where=store["data"].id.isin(voetbal.id)).drop("id", axis=1).values.tolist()
	print len(yes_vectors)
	no_vectors = store.select("data", where=store["data"].id.isin(no_ids)).drop("id", axis=1).values.tolist()
	print len(no_vectors)
	print "yes_vectors is %f times as large as no_vectors" % (len(no_vectors)/len(yes_vectors) )

	print "oversampling no_vectors"
	yes_vectors = yes_vectors* (len(no_vectors)/len(yes_vectors) )
	print "yes_vectors is %f times as large as no_vectors" % (len(no_vectors)/len(yes_vectors) )


	train_data = []
	train_labels = []
	for e in yes_vectors:
		train_data.append(e)
		train_labels.append([0, 1.0])
	for e in no_vectors:
		train_data.append(e)
		train_labels.append([1.0, 0])

	print "randomize training data"
	import random
	tmp = zip(train_data, train_labels)
	random.shuffle(tmp)
	train_data, train_labels = zip(*tmp)


	data = pd.read_csv("/media/cluster/data1/lambert/annotated", header=None, names=["id", "label", "subject"])
	test_data = store.select("data", where=store["data"].id.isin(data["id"])).drop("id", axis=1).values.tolist()
	tmp_labels = data["label"].values.tolist()
	test_labels = []
	for l in tmp_labels:
		if l == 1:
			test_labels.append([0, 1.0])
		else:
			test_labels.append([1.0, 0])
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

	train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)

	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	import numpy as np

	acc_list = []
	iter = 10000
	size = len(train_data) / iter
	print size
	for i in range(iter):
		train_step.run(feed_dict={x: np.array(train_data[i * size: (i + 1) * size]),
								  y_: np.array(train_labels[i * size: (i + 1) * size])})
		# train_step.run(feed_dict={x: batch[2].values.tolist(), y_: batch[5].values.tolist()})
		acc = accuracy.eval(feed_dict={x: test_data, y_: test_labels})
		if i %10 == 0:
			print(acc)
			# print i




# test2()
# sim_id()
# show_tweet_id()
# get_training_dataset()
# expand_annotated()
# show_expanded_tweets()
# nn()

# test_predictions()

# second_nn()
test_nn()