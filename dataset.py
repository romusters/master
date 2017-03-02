# creating and manipulating datasets
import pandas as pd
import numpy as np
import tensorflow as tf

hashtags = ["voetbal", "moslim", "werk", "economie", "jihad", "seks", "politiek"]
variables = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
store = pd.HDFStore("/media/cluster/data1/lambert/data1_sample_vector_id.clean.h5")

def create_subject_sets():
	all_tweets = pd.read_csv("/media/cluster/data1/lambert/lambert_w2v_data_jan_tweet_id.csv", names=["id", "text"])
	print "%i tweets read" % all_tweets.count()[0]

	for idx, hashtag in enumerate(hashtags):
		hashtag_set = all_tweets[all_tweets.text.str.contains(hashtag)]
		print len(hashtag_set.index)
		vectors = store.select("data", where=store["data"].id.isin(hashtag_set.id))
		print "%i tweets found for subject %s" % (len(vectors.index), hashtag)
		# create label

		vectors["labels"] = idx
		print vectors
		vectors.to_hdf("/media/cluster/data1/lambert/data1sets/"+ hashtag + ".h5", "data", format="table")
		print "Subject %s written." % hashtag


def load_all_data():
	dict = {}
	all = pd.HDFStore("/media/cluster/data1/lambert/data1sets/"+ hashtags[0] + ".h5")["data"]
	for hashtag in hashtags[1:]:
		print hashtag
		dict[hashtag] = pd.HDFStore("/media/cluster/data1/lambert/data1sets/"+ hashtag + ".h5")["data"]
		all = all.append(dict[hashtag])
		print len(all.index)
	all = all.sample(frac=1)
	return all

	# trainset.to_hdf("/media/cluster/data1/lambert/data1sets/trainset.h5", "data", format="table")
	# testset.to_hdf("/media/cluster/data1/lambert/data1sets/testset.h5", "data", format="table")



def save_meta_subject(hashtag):
	import pandas as pd
	path = "/media/cluster/data1/lambert/results/" + hashtag
	data = pd.read_csv(path + ".csv", low_memory=False)
	data = data.dropna()
	data["probs"] = data["probs"].apply(lambda x: float(x))
	data = data.sort(columns=["probs"])
	probs = data["probs"]
	threshold = float(open(path, "r").readline())
	print threshold
	positive = data[data["probs"] >= threshold]
	negative = data[data["probs"] < threshold]
	print negative
	positive.to_csv("/media/cluster/data1/lambert/datasets/meta_subjects/" + hashtag + "_meta", columns=["id", "text"])
	return positive

def load_meta_subject(hashtag):
	data = pd.read_csv("/media/cluster/data1/lambert/datasets/meta_subjects/"+ hashtag + "_meta")
	store = pd.HDFStore("/media/cluster/data1/lambert/data_sample_vector_id.clean.h5")
	ids = data.id
	pos_vectors = store.select("data", where=store["data"].id.isin(ids))
	all_ids = store["data"]
	neg_ids = all_ids[~all_ids.id.isin(ids)].id
	neg_vectors = store.select("data", where=store["data"].id.isin(neg_ids))

	pos_vectors["labels"] = 1
	neg_vectors = neg_vectors.sample(n=len(pos_vectors.index))
	neg_vectors["labels"] = 0
	result = pos_vectors.append(neg_vectors).dropna()
	return result

def load_meta_subjects(hashtags):
	print "load meta subjects"
	store = pd.HDFStore("/media/cluster/data1/lambert/data_sample_vector_id.clean.h5")
	result = pd.DataFrame()
	for i, hashtag in enumerate(hashtags[0:-1]):
		# load previous saved classes
		data = pd.read_csv("/media/cluster/data1/lambert/datasets/meta_subjects/" + hashtag + "_meta")
		pos_vectors = store.select("data", where=store["data"].id.isin(data.id))
		pos_vectors["labels"] = i
		result = result.append(pos_vectors)

	positive = save_meta_subject(hashtags[-1])
	positive["labels"] = len(hashtags)
	result = result.append(positive)
	print result
	# balance datasets!!!!



# create_subject_sets()
# import onehot

def nn():
	import balance_categories
	dataset = balance_categories.load_balanced_data()
	print "dataset"

	trainset = dataset.sample(frac=0.8, random_state=200)
	testset = dataset.drop(trainset.index)
	print "trainset"
	import numpy as np
	train_data = np.array(trainset[range(70)].values.tolist())
	train_labels = np.array(trainset["labels"].apply(lambda x: onehot(x)).values.tolist())
	print "testset"
	test_data = np.array(testset[range(70)].values.tolist())
	test_labels = np.array(testset["labels"].apply(lambda x: onehot(x)).values.tolist())
	for t in test_labels:
		if sum(t) != 1.0:
			print "broken"


	for lr in variables:
		fname = "/media/cluster/data1/lambert/results/ada_result_" + str(lr)

		sess = tf.InteractiveSession()

		n_classes = len(hashtags)
		dim = 70
		x = tf.placeholder(tf.float32, shape=[None, dim], name="Input")
		y_ = tf.placeholder(tf.float32, shape=[None, n_classes], name="Output")
		W = tf.Variable(tf.zeros([dim, n_classes]))
		b = tf.Variable(tf.zeros([n_classes]))


		y = tf.matmul(x, W) + b
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
		train_step = tf.train.AdagradOptimizer(lr).minimize(cross_entropy)
		sess.run(tf.initialize_all_variables())


		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		import numpy as np

		batch_size = 1000
		batch_count = len(train_data) / batch_size
		epochs = 8
		losses = []
		train_accs = []
		test_accs = []
		f1s = []

		for i in range(batch_count * epochs):
			begin = (i % batch_count) * batch_size
			end = (i % batch_count + 1) * batch_size
			print begin, end
			batch_data = np.array(train_data[begin : end])
			batch_labels = np.array(train_labels[begin : end])
			_, loss = sess.run([train_step, cross_entropy],feed_dict={x: batch_data, y_: batch_labels})

			test_acc = sess.run(accuracy, feed_dict={x: test_data, y_: test_labels})
			train_acc = sess.run(accuracy, feed_dict={x: batch_data, y_: batch_labels})

			prediction = tf.argmax(y, 1)
			y_pred = prediction.eval(feed_dict={x: test_data})
			gold = []
			for l in test_labels:
				label = list(l).index(1)
				gold.append(label)


			from sklearn import metrics
			f1 = metrics.f1_score(gold, list(y_pred), average="weighted")
			print loss, train_acc, test_acc, f1
			losses.append(loss)
			train_accs.append(train_acc)
			test_accs.append(test_acc)
			f1s.append(f1)
		result = pd.DataFrame({"loss": losses, "train_acc": train_accs, "test_acc": test_accs, "f1": f1s})
		result.to_csv(fname)
		sess.close()

# save optimal nn
def save_nn():

	import balance_categories
	dataset = balance_categories.load_balanced_data()
	print "dataset"

	trainset = dataset.sample(frac=0.8, random_state=200)
	testset = dataset.drop(trainset.index)
	print "trainset"
	import numpy as np
	train_data = np.array(trainset[range(70)].values.tolist())
	train_labels = np.array(trainset["labels"].apply(lambda x: onehot(x)).values.tolist())
	print "testset"
	test_data = np.array(testset[range(70)].values.tolist())
	test_labels = np.array(testset["labels"].apply(lambda x: onehot(x)).values.tolist())
	for t in test_labels:
		if sum(t) != 1.0:
			print "broken"


	sess = tf.InteractiveSession()

	n_classes = len(hashtags)
	dim = 70
	x = tf.placeholder(tf.float32, shape=[None, dim], name="Input")
	y_ = tf.placeholder(tf.float32, shape=[None, n_classes], name="Output")
	W = tf.Variable(tf.zeros([dim, n_classes]))
	b = tf.Variable(tf.zeros([n_classes]))

	y = tf.matmul(x, W) + b
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
	train_step = tf.train.AdagradOptimizer(1.0).minimize(cross_entropy)
	sess.run(tf.initialize_all_variables())
	saver = tf.train.Saver()
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	import numpy as np

	batch_size = 1000
	batch_count = len(train_data) / batch_size
	epochs = 8
	losses = []
	train_accs = []
	test_accs = []
	f1s = []

	for i in range(batch_count * epochs):
		begin = (i % batch_count) * batch_size
		end = (i % batch_count + 1) * batch_size
		print begin, end
		batch_data = np.array(train_data[begin: end])
		batch_labels = np.array(train_labels[begin: end])
		_, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_data, y_: batch_labels})

		test_acc = sess.run(accuracy, feed_dict={x: test_data, y_: test_labels})
		train_acc = sess.run(accuracy, feed_dict={x: batch_data, y_: batch_labels})

		prediction = tf.argmax(y, 1)
		y_pred = prediction.eval(feed_dict={x: test_data})
		gold = []
		for l in test_labels:
			label = list(l).index(1)
			gold.append(label)

		from sklearn import metrics
		f1 = metrics.f1_score(gold, list(y_pred), average="weighted")
		# print loss, train_acc, test_acc, f1
		losses.append(loss)
		train_accs.append(train_acc)
		test_accs.append(test_acc)
		f1s.append(f1)
		if i % batch_count == 0:
			save_path = saver.save(sess, "/media/cluster/data1/lambert/models/model.ckpt", global_step=i)
	# Save the variables to disk.


			print "Model saved in file: ", save_path
	sess.close()


def load_model():


	sess = tf.InteractiveSession()
	n_classes = len(hashtags)
	dim = 70
	x = tf.placeholder(tf.float32, shape=[None, dim], name="Input")
	y_ = tf.placeholder(tf.float32, shape=[None, n_classes], name="Output")
	W = tf.Variable(tf.zeros([dim, n_classes]))
	b = tf.Variable(tf.zeros([n_classes]))

	y = tf.matmul(x, W) + b
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
	train_step = tf.train.AdagradOptimizer(1.0).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	sess.run(tf.initialize_all_variables())
	saver = tf.train.Saver()


	saver.restore(sess,"/media/cluster/data1/lambert/models/model.ckpt-1645")
	print "Model restored"
	import balance_categories
	dataset = balance_categories.load_balanced_data()
	print "dataset"

	trainset = dataset.sample(frac=0.8, random_state=200)
	testset = dataset.drop(trainset.index)
	# testset = testset[testset["labels"] == 0]
	print "testset"

	test_data = np.array(testset[range(70)].values.tolist())
	test_labels = np.array(testset["labels"].apply(lambda x: onehot(x)).values.tolist())
	test_acc = sess.run(accuracy, feed_dict={x: test_data, y_: test_labels})
	print test_acc

	prediction = tf.argmax(y, 1)
	y_pred = prediction.eval(feed_dict={x: test_data})
	gold = []
	for l in test_labels:
		label = list(l).index(1)
		gold.append(label)
	from sklearn import metrics
	f1 = metrics.f1_score(gold, list(y_pred), average="weighted")
	print f1

	for t in test_labels:
		if sum(t) != 1.0:
			print "broken"

	def normalize(probs):
		norms = [min(sublist) for sublist in probs]
		norm = [prob - norm for prob, norm in zip(probs, norms)]
		norm = [prob / sum(prob) for prob in norm]
		return norm
	probs = y.eval(feed_dict={x: test_data})
	probs =  normalize(probs)
	voetbal_probs = [prob[0] for prob in probs]
	print voetbal_probs[0]
	tmp = np.argmax(probs, axis=1)
	print "tmp?"
	print tmp
	sorted_ids =  [i for i,e in enumerate(tmp) if e == 0]
	l = testset.id.values.tolist()
	tweet_ids = [l[i] for i in sorted_ids]
	probs_ids = [voetbal_probs[i] for i in sorted_ids]
	df = pd.DataFrame({"id": tweet_ids, "probs": probs_ids})
	print tmp
	tweets = pd.read_csv("/media/cluster/data1/lambert/lambert_w2v_data_jan_tweet_id.csv", names=["id", "text"])
	merged = pd.merge(df, tweets, on="id")
	print merged
	meta_subject =  merged[~merged.text.str.contains("voetbal")].sort(columns=["probs"])
	meta_subject = meta_subject.drop_duplicates("id")
	meta_subject.to_csv("/media/cluster/data1/lambert/meta_subject.csv")
	print meta_subject
	from plotly.offline import init_notebook_mode, plot
	init_notebook_mode()
	from plotly.graph_objs import Scatter, Figure, Layout, Pie
	trace = Scatter(x=range(len(meta_subject.probs)), y=meta_subject.probs)
	layout = Layout(title="Sorted probabilities of voetbal versus other subjects", xaxis=dict(title="indices"),
					yaxis=dict(title="Probabilities"))
	fig = Figure(data=[trace], layout=layout)
	plot(fig)

	import sys
	sys.exit(0)
	sorted_ids = sorted(range(len(tmp)), key=tmp.__getitem__)
	l = testset.id.values.tolist()
	tweet_ids = [l[i] for i in sorted_ids]
	df = pd.DataFrame({"probs": tmp, "id": tweet_ids})
	print tmp
	tweets = pd.read_csv("/media/cluster/data1/lambert/lambert_w2v_data_jan_tweet_id.csv", names=["id", "text"])
	merged = pd.merge(df, tweets, on="id")
	print merged.sort(columns=["probs"], ascending=False)


	#
	# norm_probs = normalize(probs)
	# voetbal = [x[0] for x in norm_probs]
	# voetbal.sort(reverse=True)
	# sorted_ids = sorted(range(len(voetbal)), key=voetbal.__getitem__)

	# tweet_ids = [testset.id.iloc[i] for i in sorted_ids]
	# df = pd.DataFrame({"probs": voetbal, "id": tweet_ids})
	# print df
	tweets = pd.read_csv("/media/cluster/data1/lambert/lambert_w2v_data_jan_tweet_id.csv", names=["id", "text"])
	merged = pd.merge(df, tweets, on="id")
	print merged

	def plot_voetbal_probs(voetbal):


		from plotly.offline import init_notebook_mode, plot
		init_notebook_mode()
		from plotly.graph_objs import Scatter, Figure, Layout, Pie
		traces = []
		traces.append(Scatter(x=range(len(voetbal)), y=voetbal))
		for i in range(1, len(hashtags),1):
			sub_data = [x[i] for x in norm_probs]
			traces.append(Scatter(x=range(len(sub_data)), y=sub_data))
		layout = Layout(title="Sorted probabilities of voetbal versus other subjects", xaxis=dict(title="indices"),
						yaxis=dict(title="Probabilities"))
		fig = Figure(data=traces, layout=layout)
		plot(fig)
		print voetbal


	sess.close()

# save_nn()
# load_model()


# nn()

# Add ops to save and restore all the variables.
# saver = tf.train.Saver()

# # Later, when launching the model
# with tf.Session() as sess:
# # Run the init operation.
#     sess.run(init_op)
#     ...
# # Use the model
#     ...
# # Save the variables to disk.
#     save_path = saver.save(sess, "/tmp/model.ckpt")
#     print "Model saved in file: ", save_path