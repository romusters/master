
def normalize(probs):
	norms = [min(sublist) for sublist in probs]
	norm = [prob - norm for prob, norm in zip(probs, norms)]
	norm = [prob / sum(prob) for prob in norm]
	return norm





def train_nn_iter(it, train_data, train_labels, test_data, test_labels, W_old, b_old):
	print "train nn iter"
	import numpy as np
	W = np.hstack([W_old, np.zeros((70, 1), dtype=np.float32)])
	b = np.append(b_old, np.float32(0.0))

	import tensorflow as tf
	sess = tf.InteractiveSession()
	n_classes = it
	dim = 70
	x = tf.placeholder(tf.float32, shape=[None, dim], name="Input")
	y_ = tf.placeholder(tf.float32, shape=[None, n_classes], name="Output")
	W = tf.Variable(W)
	b = tf.Variable(b)
	y = tf.matmul(x, W) + b
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
	train_step = tf.train.AdagradOptimizer(1.0).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()


	batch_size = 1000
	batch_count = len(train_data) / batch_size
	epochs = 8
	losses = []
	train_accs = []
	test_accs = []
	f1s = []

	max = 0
	for i in range(batch_count * epochs):
		begin = (i % batch_count) * batch_size
		end = (i % batch_count + 1) * batch_size
		batch_data = np.array(train_data[begin: end])
		batch_labels = np.array(train_labels[begin: end])
		_, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_data, y_: batch_labels})

		test_acc = sess.run(accuracy, feed_dict={x: test_data, y_: test_labels})
		train_acc = sess.run(accuracy, feed_dict={x: batch_data, y_: batch_labels})
		train_accs.append(train_acc)
		test_accs.append(test_acc)
		prediction = tf.argmax(y, 1)
		y_pred = prediction.eval(feed_dict={x: test_data})
		gold = []
		for l in test_labels:
			label = list(l).index(1)
			gold.append(label)

		from sklearn import metrics
		f1 = metrics.f1_score(gold, list(y_pred), average="weighted")

		if f1 > max:
			max = f1
		# if i % (batch_count * epochs) == 0:
		# 	import os
		# 	os.mkdir("/media/cluster/data1/lambert/models/iter/" + str(it))
		# 	saver.save(sess, "/media/cluster/data1/lambert/models/iter/" + str(it) + "/model_iter.ckpt-" + str(it))
	print max
	print train_accs
	print test_accs
	W = W.eval()
	b = b.eval()
	sess.close()
	return W, b

# import balance_categories
# hashtags = ["voetbal", "moslim", "werk"]
# train_data, train_labels, test_data, test_labels = balance_categories.balance_iteratively(hashtags)
# a,b =  restore_nn(3)
# print a,b
