# load the data and balance the categories
import pandas as pd
# hashtags = ["voetbal", "moslim", "werk", "economie", "jihad", "seks", "politiek"]


def balance_iteratively(hashtags):
	all = pd.DataFrame()
	for hashtag in hashtags:
		tmp = pd.HDFStore("/media/cluster/data1/lambert/datasets/" + hashtag + ".h5")["data"]
		tmp.index = tmp.id
		all = all.append(tmp)
	all_tweets = pd.read_csv("/media/cluster/data1/lambert/lambert_w2v_data_jan_tweet_id.csv", names=["id", "text"])
	all_ids = all_tweets["id"]
	all = all.drop_duplicates("id", keep=False)
	size_largest_label = all.groupby("labels").count().id.max()  # 42869
	print size_largest_label

	balanced_data = pd.DataFrame()
	for label in set(all["labels"]):
		category_data = all[all["labels"] == label]
		new_data = category_data.sample(n=size_largest_label, replace=True)
		balanced_data = balanced_data.append(new_data)
	balanced_data = balanced_data.sample(frac=1)
	print len(balanced_data.index)

	dataset = all.dropna()
	trainset = dataset.sample(frac=0.8, random_state=200)
	testset = dataset.drop(trainset.index)
	print "trainset"
	import numpy as np
	import feature
	train_data = np.array(trainset[range(70)].values.tolist())
	train_labels = np.array(trainset["labels"].apply(lambda x: feature.onehot(x, hashtags)).values.tolist())
	print "testset"
	test_data = np.array(testset[range(70)].values.tolist())
	test_labels = np.array(testset["labels"].apply(lambda x: feature.onehot(x, hashtags)).values.tolist())
	return train_data, train_labels, test_data, test_labels

def load_all_data():
	all = pd.DataFrame()
	for hashtag in hashtags:
		tmp = pd.HDFStore("/media/cluster/data1/lambert/datasets/"+ hashtag + ".h5")["data"]
		tmp.index = tmp.id
		all = all.append(tmp)
	all_tweets = pd.read_csv("/media/cluster/data1/lambert/lambert_w2v_data_jan_tweet_id.csv", names=["id", "text"])
	all_ids = all_tweets["id"]
	print len(all.index)
	print len(all_ids)
	return all.sample(frac=1)


def balance_data():
	data = load_all_data()
	data = data.drop_duplicates("id", keep=False)
	size_largest_label = data.groupby("labels").count().id.max()#42869
	print size_largest_label

	balanced_data = pd.DataFrame()
	for label in set(data["labels"]):
		category_data = data[data["labels"] == label]
		new_data = category_data.sample(n=size_largest_label, replace=True)
		balanced_data = balanced_data.append(new_data)
	balanced_data = balanced_data.sample(frac=1)
	print len(balanced_data.index)
	balanced_data.to_hdf("/media/cluster/data1/lambert/datasets/balanced.h5", "data")

def load_balanced_data():
	data = pd.HDFStore("/media/cluster/data1/lambert/datasets/balanced.h5")["data"]
	data.index = range(len(data.index))
	return data.dropna()

def train_testset():
	data = load_balanced_data()
	data.index = range(len(data.index))
	print len(data.index)

	train = data.sample(frac=0.8, random_state=200)
	print train
	test = data[~data.index.isin(train.index)]
	print test


def add_category(category):
	all_tweets = pd.read_csv("/media/cluster/data1/lambert/lambert_w2v_data_jan_tweet_id.csv", names=["id", "text"])
	category_data = all_tweets[all_tweets.text.str.contains(r"\b%s\b" % category, case=False)]
	balanced_data = load_balanced_data()
	size_largest_label = balanced_data.groupby("labels").count().id.max()
	print size_largest_label
	new_data = category_data.sample(n=size_largest_label, replace=True)
	new_data.to_csv("/media/cluster/data1/lambert/" + category + ".csv")

	store = pd.HDFStore("/media/cluster/data1/lambert/data_sample_vector_id.clean.h5")
	ids = new_data.id
	vectors = store.select("data", where=store["data"].id.isin(ids))
	vectors = vectors.sample(n=size_largest_label, replace=True)
	vectors["labels"] = [len(hashtags)]*len(vectors.index)
	vectors.to_hdf("/media/cluster/data1/lambert/datasets/" + category + ".h5", "data")
	hashtags.append(category)
	print hashtags

category = "trump"
# add_category(category)
# cat_data = pd.read_hdf("/media/cluster/data1/lambert/datasets/" + category+ ".h5", "data")
# all_data = pd.read_hdf("/media/cluster/data1/lambert/datasets/balanced.h5", "data")
# print cat_data
# print all_data
# all_data =  all_data.append(cat_data).sample(frac=1)
# all_data.to_hdf("/media/cluster/data1/lambert/datasets/balanced.h5", "data")



# load_all_data()
# balance_data()
# train_testset()
# data = load_balanced_data()

# def show_tweet():
# 	import pandas as pd
# 	all_tweets = pd.read_csv("/media/cluster/data1/lambert/lambert_w2v_data_jan_tweet_id.csv", names=["id", "text"])
# 	while True:
# 		id = int(input())
# 		print all_tweets[all_tweets.id == id].text.values