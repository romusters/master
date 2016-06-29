# used to test how to use multiple cores in Python
from functools import partial
from multiprocessing import Pool
import numpy as np
import pickle

def edit_distance(base_tweet, tweet):
	base_tweet = eval(base_tweet)
	tweet = eval(tweet)
	base_tweet = np.asarray(base_tweet)
	tweet = np.asarray(tweet)
	return abs(np.sum(np.subtract(base_tweet, tweet)))

def dist_mp(base_tweet, tweet):
	return tweet, edit_distance(base_tweet, tweet)

def main():
	sequence_list = []
	for line in open("/data/s1774395/vectors.txt","r"):
		sequence_list.append(line)

	results = []

	base_tweet = sequence_list[0]
	pool = Pool()  # use all CPUs
	for tweet, d in pool.imap_unordered(partial(dist_mp, base_tweet), sequence_list):
		results.append(d)
	pool.close()
	pool.join()
	pickle.dump(open("/data/s1774395/distances", "rb"), results)


if __name__ == "__main__":
	main()