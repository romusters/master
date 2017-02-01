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
		W_prev, b_prev = nn_init.init_nn(i, hashtags, train_data, train_labels, test_data, test_labels)
		import sys
		sys.exit(0)
	else:
		print hashtags
		# a, b = nn_restore.restore_nn(i)
		# print a, b
		W_prev, b_prev = nn_utils.train_nn_iter(i, train_data, train_labels, test_data, test_labels, W_prev, b_prev)