def restore_nn(it):
	print "restore nn"
	import tensorflow as tf
	sess = tf.InteractiveSession()
	n_classes = it-1
	print n_classes
	x = tf.placeholder(tf.float32, shape=[None, 70], name="Input")
	y_ = tf.placeholder(tf.float32, shape=[None, n_classes], name="Output")
	W = tf.Variable(tf.zeros([70, n_classes]))
	b = tf.Variable(tf.zeros([n_classes]))
	y = tf.matmul(x, W) + b
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
	train_step = tf.train.AdagradOptimizer(1.0).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	saver = tf.train.Saver()
	path = "/media/cluster/data1/lambert/models/iter/"+ str(it-1)+"/model_iter.ckpt-" + str(it-1)
	saver.restore(sess, path)
	W_old = W.eval()
	b_old = b.eval()
	sess.close()
	print W_old, b_old
	return W_old, b_old
# import nn_utils
# a,b = restore_nn(3)
# print a,b
# import balance_categories
# import nn_utils
# train_data, train_labels, test_data, test_labels = balance_categories.balance_iteratively(hashtags = ["voetbal", "moslim", "werk"])
# nn_utils.train_nn_iter(3,train_data, train_labels, test_data, test_labels,a,b )
