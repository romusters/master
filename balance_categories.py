# load the data and balance the categories
import pandas as pd
import numpy as np
hashtags = ["voetbal", "moslim", "werk", "economie", "jihad", "seks", "politiek"]

def load_all_data():
	dict = {}
	all = pd.HDFStore("/media/cluster/data1/lambert/datasets/"+ hashtags[0] + ".h5")["data"]
	for hashtag in hashtags[1:]:
		print hashtag
		dict[hashtag] = pd.HDFStore("/media/cluster/data1/lambert/datasets/"+ hashtag + ".h5")["data"]
		all = all.append(dict[hashtag])
		print len(all.index)
	all = all.sample(frac=1)
	return all

def balance_data():
	data = load_all_data()

	# largest_label = data.groupby("labels").count().id.idxmax()#42869
	size_largest_label = data.groupby("labels").count().id.max()#42869
	print size_largest_label
	# print largest_label, size_largest_label


	balanced_data = pd.DataFrame()
	for label in set(data["labels"]):
		category_data = data[data["labels"] == label]
		new_data = category_data.sample(n=size_largest_label, replace=True)
		balanced_data = balanced_data.append(new_data)
	balanced_data = balanced_data.sample(frac=1)
	print len(balanced_data.index)
	# print balanced_data.groupby("labels").count()
	balanced_data.to_hdf("/media/cluster/data1/lambert/datasets/balanced.h5", "data")

def load_balanced_data():
	return pd.HDFStore("/media/cluster/data1/lambert/datasets/balanced.h5")["data"]