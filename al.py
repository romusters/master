import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
def active_learning(seed_id, threshold):
	logging.info("Start Active Learning")
	import pandas as pd

	#get the tweet with similarities
	sim_name = "/media/cluster/data1/lambert/lambert_w2v_data_jan_sims_" + str(seed_id) + ".csv"
	tweet_sims = pd.read_csv(sim_name, header=None)

	# get all the indices, the embeddings will be checked on uniqueness
	yes_idx = tweet_sims[tweet_sims[1] > threshold][0].values.tolist()
	print yes_idx
	logging.info("Amount of positive tweets found: %i", len(yes_idx))

	# get the embedding belonging to ids
	store = pd.HDFStore("/media/cluster/data1/lambert/lambert_w2v_data_jan_all_columns.csv.clean.h5")
	yes_data = store.select("data", where=store["data"]["id"].isin(yes_idx))
	logging.info("Amount of tweets found in the store: %i", yes_data.shape[0])
	dim = yes_data.shape[1]-1
	# get the tweet ids with the highest similarities and ...
	#note that the row indices in the df below are not tweet ids.

	if len(yes_idx) > 20:
		logging.info( "sample of 20")
		yes_sample = yes_data.sample(20)
		yes_idx = yes_sample["id"]
		yes_embs = yes_sample[range(dim)]
	else:
		logging.info("sample smaller than 20")
		yes_idx = yes_data["id"]
		yes_embs = yes_data[range(dim)]

	tweets = pd.read_csv("/media/cluster/data1/lambert/lambert_w2v_data_jan_tweet_id.csv", header=None)
	yes_tweets = tweets[tweets[0].isin(yes_idx)].values.tolist()
	for tweet in yes_tweets:
		logging.info(tweet)
	yes_tokens = pd.read_csv("/media/cluster/data1/lambert/data_sample_tokens_id.csv", header=None)
	yes_tokens = yes_tokens[yes_tokens[0].isin(yes_idx)].values.tolist()
	vocab = w2v.get_vocab()
	for token in yes_tokens:
		token = token[1].split()
		res = []
		for t in token:
			if t in vocab:
				res.append(t)
		logging.info(res)


	# check if dimension of model and data are the same
	model_name = "/media/cluster/data1/lambert/lambert_model.csv.h5"
	model_store = pd.HDFStore(model_name)
	dim_model = len(model_store["data"]["vectors"][0])
	dim_data = yes_embs.shape[1]

	logging.info(dim_model)
	logging.info(dim_data)

	assert(dim_model == dim_data)
	dim = dim_data

	yes_embs = yes_embs.values.tolist()
	logging.info("Number of datapoints is %i and the dimension is %i", len(yes_embs), len(yes_embs[0]))
	data = yes_embs
	labels = []
	labels.extend([[0,1]]*len(yes_embs))
	# yes.append(pd.DataFrame(data=[dim * [[0, 1]]]).T)

	no_idx = tweet_sims[tweet_sims[1] < 0.5][0].values.tolist()
	logging.info(no_idx[0])
	no_data = store.select("data", where=store["data"]["id"].isin(no_idx))

	if no_data.shape[0]>20:
		logging.info("sample of 20")
		no_sample = no_data.sample(20)
		# no_idx = no_sample["id"]
		logging.debug(no_idx)
		no_embs = no_sample[range(dim)]
	else:
		logging.info( "sample smaller than 20")
		# no_idx = no_data["id"]
		no_embs = no_data[range(dim)]

	no_tweets = tweets[tweets[0].isin(no_idx)].values.tolist()
	# tokens = pd.read_csv("/media/cluster/data1/data_sample_tokens_id.csv", header=None)

	no_embs = no_embs.values.tolist()
	# for emb in no_embs:
	# 	logging.debug(no_embs)
	logging.debug(no_embs[0])
	logging.debug(no_embs[-1])
	labels.extend([[1, 0]] * len(no_embs))
	data.extend(no_embs)
	import numpy as np
	labels = np.array(labels)
	data = np.array(data)

	for n in data:
		logging.debug(n)
	logging.info("Length of data is %i and the dimension of first element is %i and the last element is %i", len(data), len(data[0]), len(data[-1]))

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
	import numpy as np
	correct_prediction.eval(feed_dict={x: data, y_: labels})

	for i in range(3):
		train_step.run(feed_dict={x: np.array(data), y_: np.array(labels)})
			# train_step.run(feed_dict={x: batch[2].values.tolist(), y_: batch[5].values.tolist()})
		print(accuracy.eval(feed_dict={x: data, y_: labels}))

def get_seed(word):
	logging.info("Seed word is: %s", word)
	with open("/media/cluster/data1/lambert/lambert_w2v_data_jan_tweet_id.csv", 'rb') as f:
		for line in f:
			if word in line:
				print line
				idx = int(line.split(",")[0])
				break
	logging.info("Seed id is %i", idx)
	return idx

import w2v
word = "moslim"
# word = "vegan"
# word = "rutte"
# word = "nieuwjaar"
# word = "1"
# word = "overheid"
seed_id = get_seed(word)
embs_name = "/media/cluster/data1/lambert/lambert_w2v_data_jan_all_columns.csv.clean.h5"
w2v.predict(embs_name, seed_id)

threshold = 0.993
active_learning(seed_id, threshold)
