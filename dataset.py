# creating and manipulating datasets

import pandas as pd
import numpy as np
hashtags = ["voetbal", "moslim", "werk", "economie", "jihad", "seks", "politiek"]
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

def onehot(x):
	tmp = np.zeros(len(hashtags))
	tmp[x] = 1
	return tmp

# create_subject_sets()

# dataset = load_all_data()
import balance_categories
dataset = balance_categories.load_balanced_data()
trainset = dataset.sample(frac=0.8)

train_data = np.array(trainset[range(70)].values.tolist())
train_labels = np.array(trainset["labels"].apply(lambda x: onehot(x)).values.tolist())
print train_data.shape
testset = dataset.drop(trainset.index)
test_data = np.array(testset[range(70)].values.tolist())
test_labels = np.array(testset["labels"].apply(lambda x: onehot(x)).values.tolist())
print test_data.shape
print test_data[0]
print test_labels[0]



def test2():
	import tensorflow as tf


	x = tf.placeholder(tf.float32, shape=[None, 70], name="Input")
	y_ = tf.placeholder(tf.float32, shape=[None, 7], name="Output")
	t = tf.placeholder(tf.float32, shape=[None, 70], name="TestInput")
	# y_t = y_ = tf.placeholder(tf.float32, shape=[None, 7], name="TestOutput")

	W = tf.Variable(tf.zeros([70, 7]))
	b = tf.Variable(tf.zeros([7]))


	y = tf.matmul(x, W) + b
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))



	l = tf.placeholder(tf.float32, shape=[], name="learning_rate")
	train_step = tf.train.GradientDescentOptimizer(l).minimize(cross_entropy)
	prediction = tf.argmax(y, 1)
	# t_pred = tf.argmax(y, 1)
	sess = tf.Session()
	init = tf.initialize_all_variables()
	sess.run(init)

	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	for _learning_rate in np.arange(0, 1, 0.1):
		# acc = accuracy.eval(session=sess, feed_dict={x: test_data, y_: test_labels})


		acc_list = []
		iter = 100
		size = len(train_data) / iter
		print "size", size
		for i in range(iter):
			print i
			sess.run(train_step, feed_dict={x: np.array(train_data[i * size: (i + 1) * size]),
									  y_: np.array(train_labels[i * size: (i + 1) * size]), l: _learning_rate})
			acc = accuracy.eval(session=sess, feed_dict={x: test_data, y_: test_labels})
			print acc

		# n_iters = 1000
		# batch_fraction = len(train_data) / n_iters
		# for i in range(n_iters):
		# 	sess.run(train_step, feed_dict={x: np.array(train_data[i * batch_fraction: (i + 1) * batch_fraction]), y_: np.array(train_labels[i * batch_fraction: (i + 1) * batch_fraction]), l: _learning_rate})
		#

		# pred = prediction.eval(session=sess, feed_dict={x: test_data})
		#
		# print pred

		# y_p = tf.argmax(y, 1)
		# val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x:test_data, y_ :test_labels})
		#
		#
		acc = accuracy.eval(session=sess, feed_dict={x: test_data, y_: test_labels})
		print acc

		# acc = sess.run(accuracy, feed_dict={x: test_data, y_: test_labels})
		# print acc
		# 	# acc, pred = sess.run([accuracy, prediction], feed_dict={x: test_data,  y_: test_labels})
		# print val_accuracy





def test1():
	import tensorflow as tf
	sess = tf.InteractiveSession()

	n_classes = len(hashtags)
	dim = 70
	x = tf.placeholder(tf.float32, shape=[None, dim], name="Input")
	y_ = tf.placeholder(tf.float32, shape=[None, n_classes], name="Output")
	W = tf.Variable(tf.zeros([dim, n_classes]))
	b = tf.Variable(tf.zeros([n_classes]))


	y = tf.matmul(x, W) + b
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

	train_step = tf.train.AdagradOptimizer(0.5).minimize(cross_entropy)
	sess.run(tf.initialize_all_variables())
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	import numpy as np

	acc_list = []
	iter = 1000
	size = len(train_data) / iter
	print "size", size
	for i in range(iter):
		print i
		train_step.run(feed_dict={x: np.array(train_data[i * size: (i + 1) * size]),
									  y_: np.array(train_labels[i * size: (i + 1) * size])})
		acc = accuracy.eval(feed_dict={x: test_data, y_: test_labels})
		if i %10 == 0:
			print(acc)

		from sklearn import metrics
		if i == 30:
			break
		print "validation accuracy:", acc
		prediction = tf.argmax(y, 1)

			# y_true = prediction.eval(feed_dict={x: test_data})
			# # print y_true

			# print y_true
			# # print correct_prediction.eval(feed_dict={x: train_data, y_: train_labels})
			# prediction = tf.argmax(test_data, 1)
		y_pred = prediction.eval(feed_dict={x: test_data})
		print y_pred
		gold = list([list(x).index(1) for x in list(test_labels)])
			# # print "Precision", metrics.precision_score(y_true, y_pred, average="weighted")
			# # print "Recall", metrics.recall_score(y_true, y_pred, average="weighted")
		print "f1_score", metrics.f1_score(gold, y_pred, average="weighted")
			# # print "confusion_matrix"
			# # print metrics.confusion_matrix(y_true, y_pred)
			# # fpr, tpr, tresholds = metrics.roc_curve(y_true, y_pred)




def test3():
	import numpy as np
	file = open("/media/cluster/data1/lambert/results/result", "w")
	for lr in [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4,5]:
		file = open("/media/cluster/data1/lambert/results/result_acc_adagrad", "a")
		import tensorflow as tf
		sess = tf.InteractiveSession()

		n_classes = len(hashtags)
		dim = 70
		x = tf.placeholder(tf.float32, shape=[None, dim], name="Input")
		y_ = tf.placeholder(tf.float32, shape=[None, n_classes], name="Output")
		W = tf.Variable(tf.zeros([dim, n_classes]))
		b = tf.Variable(tf.zeros([n_classes]))


		y = tf.matmul(x, W) + b
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

		train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
		sess.run(tf.initialize_all_variables())
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		import numpy as np

		iter = 1000
		size = len(train_data) / iter
		print "size", size
		max = (0,0, 0, 0, 0)
		for i in range(iter):
			train_step.run(feed_dict={x: np.array(train_data[i * size: (i + 1) * size]),
										  y_: np.array(train_labels[i * size: (i + 1) * size])})
			acc = accuracy.eval(feed_dict={x: test_data, y_: test_labels})

		# 	if acc > max[0]:
		# 		max = (acc, i, lr)
		#
		# 	# print "validation accuracy:", acc
		#
		# print max

			prediction = tf.argmax(y, 1)
			y_pred = prediction.eval(feed_dict={x: test_data})
			gold = []
			for l in test_labels:
				label = list(l).index(1)
				gold.append(label)
			# print len(gold)
			# print len(y_pred)
			# print gold
			# print y_pred
			# import sys
			# sys.exit(0)

			# gold = list([list(x).index(1) for x in list(test_labels)])
			from sklearn import metrics
			f1 = metrics.f1_score(gold, list(y_pred), average="weighted")
			# print "f1_score", f1
			print acc
			if acc > max[0]:
				max = (acc, i, lr, gold, list(y_pred))
			# print metrics.confusion_matrix(gold, y_pred)
		print max
		file.write(str(max) + "\n")
		file.close()




test3()

# test2()

# test1()