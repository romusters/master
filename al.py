import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def active_learning(seed_id, threshold):
	logging.info("Start Active Learning")
	import pandas as pd
	store = pd.HDFStore("/media/cluster/data1/lambert/data_sample_vector_id.clean.h5")
	tweets = pd.read_csv("/media/cluster/data1/lambert/lambert_w2v_data_jan_tweet_id.csv", header=None)
	tokens = pd.read_csv("/media/cluster/data1/lambert/data_sample_tokens_id.csv", header=None)
	#get the tweet with similarities
	sim_name = "/media/cluster/data1/lambert/lambert_w2v_data_jan_sims_" + str(seed_id) + ".csv"
	tweet_sims = pd.read_csv(sim_name, header=None)

	# get all the indices of the tweets which are more similar than the threshold
	yes_idx = tweet_sims[tweet_sims[1] > threshold][0].values.tolist()
	logging.info("Amount of positive tweets found: %i", len(yes_idx))

	# get the embedding belonging to the indices
	yes_data = store.select("data", where=store["data"]["id"].isin(yes_idx))
	logging.info("Amount of tweets found in the store: %i", yes_data.shape[0])

	# make sure that the similarity file did not get appended by accident
	assert(len(yes_idx) == yes_data.shape[0])
	dim = yes_data.shape[1]-1

	#get the sample of the tweets which are similar
	yes_idx, yes_embs = get_sample(yes_idx, yes_data, dim)

	yes_tweets = tweets[tweets[0].isin(yes_idx)].values.tolist()
	# log_tweets(yes_tweets)

	import single_char
	getch = single_char._Getch()
	print "Press y (yes) or n (no)"
	input_labels = []
	for tweet in yes_tweets:
		print tweet
		val = getch.__call__()
		if val == "y":
			val = [0, 1]
		elif val == "n":
			val = [1, 0]

		input_labels.append( val )
	logging.info("input labels %s" , str(input_labels))


	yes_tokens = tokens[tokens[0].isin(yes_idx)].values.tolist()
	vocab = w2v.get_vocab()
	# log_tokens(yes_tokens, vocab)

	# check if dimension of model and data are the same
	check_dims(yes_embs)

	yes_embs = yes_embs.values.tolist()
	logging.info("Number of datapoints is %i and the dimension is %i", len(yes_embs), len(yes_embs[0]))

	# create variables to hold the Tensorflow data
	data = yes_embs
	labels = []
	labels.extend([[0,1]]*len(yes_embs))

	no_idx = tweet_sims[(tweet_sims[1] < 0.0) & (tweet_sims[1] > -1.0)][0].values.tolist()
	no_data = store.select("data", where=store["data"]["id"].isin(no_idx))


	no_idx, no_embs = get_sample(no_idx, no_data, dim)
	no_tweets = tweets[tweets[0].isin(no_idx)].values.tolist()
	log_tweets(no_tweets)
	no_tokens = tokens[tokens[0].isin(no_idx)].values.tolist()
	# log_tokens(no_tokens, vocab)
	no_embs = no_embs.values.tolist()
	labels.extend([[1, 0]] * len(no_embs))
	data.extend(no_embs)

	import numpy as np
	labels = np.array(labels)
	data = np.array(data)

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
			# if word in line:
			import re
			if re.search(r"\b%s\b" % word, line):
				print line
				idx = int(line.split(",")[0])
				break
	logging.info("Seed id is %i", idx)
	return idx

def log_tweets(tweets):
	for tweet in tweets:
		logging.info(tweet)


def log_tokens(tokens, vocab):
	for token in tokens:
		token = token[1].split()
		res = []
		for t in token:
			if t in vocab:
				res.append(t)
		logging.info(res)

def get_sample(idx, data, dim):
	if len(idx) > 20:
		logging.info( "sample of 20")
		sample = data.sample(20)
		idx = sample["id"]
		embs = sample[range(dim)]
	else:
		logging.info("sample smaller than 20")
		idx = data["id"]
		embs = data[range(dim)]
	return idx, embs

def check_dims(embs):
	import pandas as pd
	model_name = "/media/cluster/data1/lambert/lambert_model.csv.h5"
	model_store = pd.HDFStore(model_name)
	dim_model = len(model_store["data"]["vectors"][0])
	dim_data = embs.shape[1]

	logging.info(dim_model)
	logging.info(dim_data)

	assert (dim_model == dim_data)
	dim = dim_data


# import w2v
# words = ["voetbal", "racisme", "moslim", "veganisme", "politiek", "seks", "duurzame energie"]
# words = ["moslim"]
# for word in words:
#
# 	# get a topic using a word
# 	seed_id = get_seed(word)
# 	import plot
#
#
#
# 	# create the similarity file corresponding to the topic
# 	embs_name = "/media/cluster/data1/lambert/data_sample_vector_id.clean.h5"
# 	# w2v.predict(embs_name, seed_id)
# 	# raw_input("Press enter to continue...")
# 	# print "Continuing"
#
# 	# plot.plot_sims(seed_id, word)
#
# 	# find tweets corresponding to the topic and build a classifier
# 	threshold = 0.8
# 	# active_learning(seed_id, threshold)
# 	# break


'''
Determine which tweet it still about voetbal
'''
def threshold_subject(hashtag):
	import pandas as pd
	data = pd.read_csv("/media/cluster/data1/lambert/results/" + hashtag + ".csv", header=None, low_memory=False)
	data = data.dropna()
	data[2] = data[2].apply(lambda x: float(x))
	data = data.sort(columns=[2])
	probs = data[2]
	t = 0#(max(probs)-min(probs))/2
	tmp = data[data[2] > t]
	print tmp.head()[3].values
	# if probs are large enough to be voetbal, lower t.


	# assume the list is ordered from voetbal -2 to not voetbal +10 or sth.
	while True:
		tmp = data[data[2] > t]
		print tmp.head()[3].values
		flag = True
		inp = int(input())
		t = 0
		while flag:
			print t
			if inp == 0:
				print "not about the subject"
				t +=1
				tmp = data[data[2] > t]
				print tmp.head()[3].values
				inp = int(input())
			if inp == 1:
				print "about the subject!"
				flag = False
				prev_t = t-1
		print "the threshold is %i" % t
		print "the previous threshold was %i" % prev_t
		high = t

		low = prev_t
		center = high-(high-low)/2.0

		def test(high,low, center):
			tmp = data[data[2] > center]
			print tmp.head()[3].values
			inp = int(input())
			if inp == 1:
				new_low = high-(high-center)/2.0
				return high, new_low, high-(center-new_low)/2.0
			elif inp == 0:
				return center, low, center - (center - low)/2.0

		while True:
			high, low, center = test(high, low, center)
			print high, low, center




		while True:

			if inp == 1:
				t = t - ((t-prev_t) / 2.0)
				print "the new threshold is %f" % t
				tmp = data[data[2] > t]
				print tmp.head()[3].values

			elif inp == 0:
				t = t + ((t-prev_t) / 2.0)
				print "the new threshold is %f" % t
				tmp = data[data[2] > t]
				print tmp.head()[3].values
			inp = int(input())


		# if probs are too small, thus it is not about footbal, increase the probs until you found voetbal

'''
Determine which tweet it still about voetbal
'''
def threshold_subject_tmp(hashtag):
	import pandas as pd
	data = pd.read_csv("/media/cluster/data1/lambert/results/" + hashtag + ".csv", header=None, low_memory=False)
	data = data.dropna()
	data[2] = data[2].apply(lambda x: float(x))
	data = data.sort(columns=[2])
	probs = data[2]
	t = 0#(max(probs)-min(probs))/2
	tmp = data[data[2] > t]
	print tmp.head()[3].values
	# if probs are large enough to be voetbal, lower t.

	delta = 0.8
	t = 1
	occ = 1
	first = 1
	count = 0 # if too many annotations done, stop
	prev_t = 100
	prev_zero = 0
	# assume the list is ordered from voetbal -2 to not voetbal +10 or sth.
	while True:
		flag = False
		inp = int(input())
		# if probs are too small, thus it is not about footbal, increase the probs until you found voetbal
		if inp == 0:
			flag = True
			while flag:

				print "initial conditions"
				print t
				# begin condition,
				if inp == 0:
					t = t + 1
					tmp = data[data[2] > t]
					print tmp.head()[3].values
					inp = int(input())
					count += 1
				if inp == 1:
					flag = False
					tmp = data[data[2] > t]
					print tmp.head()[3].values
		# you found football, now decrease the values
		print t
		import math
		count += 1
		if inp == 1:
			occ = occ + 1
			t = t - delta * occ
			print "you are in voetbal now"
			flag = True
			print t
			while flag:


				# great you are still looking at voetbal
				if inp == 1:
					count += 1
					tmp = data[data[2] > t]
					print tmp.head()[3].values
					occ = occ + 1
					t = t-math.pow(delta, occ)
					prev_zero = False
					print t

					inp = int(input())
				# too bad, you are not looking at voetbal anymore, increase the threshold a bit
				if inp == 0:
					count += 1
					tmp = data[data[2] > t]
					print tmp.head()[3].values
					occ = occ + 1
					if prev_zero:
						t = t + math.pow(delta, occ) + 0.4 * prev_zero
					prev_zero +=1
					print t

					inp = int(input())
				if count > 10: #if too many annotations needed, stop
					print "too many iterations"
					flag = 0
				diff = abs(prev_t -t)
				print "the difference between current and previous threshold is %i", diff
				if  diff < 0.1: # if difference too small, stop
					print "difference very small %i", prev_t - t
					flag = 0
				prev_t = t
		tmp = data[data[2] > t]
		print tmp.head()[3]
		f = open("/media/cluster/data1/lambert/results/" + hashtag, "w")
		f.write(str(t))
		f.close()
		# t= 3.5
'''
Determine tweets which do no contain voetbal but are about voetbal
'''
def show_voetbal():
	import pandas as pd
	data = pd.read_csv("/media/cluster/data1/lambert/results/voetbal_moslim.csv", header=None, low_memory=False)
	data = data.dropna()
	data[2] = data[2].apply(lambda x: float(x))
	data = data.sort(columns=[2])
	tweets = data[data[2] > 3.5]
	print tweets
	tweets = tweets[~tweets[3].str.contains("voetbal")]
	print len(tweets.index)
	print len(data.index)
	# print tweets

# show_voetbal()
threshold_subject("moslim")