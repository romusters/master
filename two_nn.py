all_hashtags = ["voetbal", "moslim", "werk", "economie", "jihad", "seks", "politiek"]
import nn_utils
import nn_init
import nn_restore
import balance_categories

W_prev, b_prev = None, None

for i in range(2, len(all_hashtags), 1):
	print "network %i" % i
	hashtags = all_hashtags[0:i]

	# balance n hashtag datasets
	train_data, train_labels, test_data, test_labels = balance_categories.balance_iteratively(hashtags)
	# load model iteratively

	# use previous models network settings

	if i == 2:
		print "init"
		# it's important to train the network 1v1 subject and not 1vnoise because in the noise there can still be tweets about the subject.
		W_prev, b_prev = nn_init.init_nn(i, hashtags, train_data, train_labels, test_data, test_labels)

		# now we need the probabilities to assess if the topic has been learned.
		import al
		al.threshold_subject(hashtags[0])
		al.threshold_subject(hashtags[1])

		# use the meta subjects versus noise as initialization.
		nn_utils.train_meta_init(hashtags[0])
		# next use the previous subject(s) as noise and use the threshold for the current subject to make the subject class
		nn_utils.train_meta_iter(hashtags[1])

		import pickle as p
		path_w = "/media/cluster/data1/lambert/W"
		path_b = "/media/cluster/data1/lambert/b"
		p.dump(W_prev, open(path_w, "wb"))
		p.dump(b_prev, open(path_b, "wb"))
		import sys
		sys.exit(0)
	else:
		print hashtags
		# a, b = nn_restore.restore_nn(i)
		# print a, b
		W_prev, b_prev = nn_utils.train_nn_iter(i, train_data, train_labels, test_data, test_labels, W_prev, b_prev)
		nn_utils.train_meta(hashtags[-1])