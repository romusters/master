def no_use():
	import pandas as pd
	d_name = "/media/cluster/data1/model3971/w2v_data_jan_large.csv"
	id_name = "/media/cluster/data1/model3971/w2v_data_jan.csv.clean.sims.h5"
	store = pd.HDFStore(id_name)
	ids = store["data"]["id"].values.tolist()

	data = pd.read_csv(d_name, header=None, iterator=True, chunksize=1000)
	chunk = data.get_chunk()

	# iterate over the ids we need
	for id in ids:
		while chunk is not None:
			df = chunk[chunk[3].isin([id])]
			if not df.empty:
				print df

			else:
				continue
			# found_ids = []
			# found_tweets = []
			# for row in chunk.iterrows():
			# 	found_id = row[1][3]
			# 	if id in ids:
			# 		found_ids.append(id)
			# 		print row[1][0]
			# 		found_tweets.append(row[1][0])
			chunk = data.get_chunk()



def data_sample_tweet_id(vector_name, tweet_name):
	import pandas as pd
	tweets = pd.read_csv(vector_name, header=None, iterator=True, chunksize=1000, usecols=[0, 1])
	chunk = tweets.get_chunk()
	while chunk is not None:
		chunk.index = chunk[0]
		chunk[1].to_csv(tweet_name, mode="a")
		chunk = tweets.get_chunk()



# when collecting the lookup table from hadoop through the terminal, we get a dataframe which we convert to dict: dict(df)
def dict_to_csv(dict, fname):
	import csv
	with open(fname, 'wb') as f:
		w = csv.writer(f)
		w.writerows(dict.items())


def convert_lookup_to_hdf(fname):
	import pandas as pd
	data = pd.read_csv(fname, names=["words", "vectors"])
	data["vectors"] = data["vectors"].apply(lambda x: eval(x))
	data.to_hdf(fname + ".h5", "data")



def data_sample_vector_id(data_name, vector_name):
	import pandas as pd
	chunksize = 100000
	tweets = pd.read_csv(data_name, header=None, iterator=True, index_col=False, chunksize=chunksize, usecols=[2,3], names=["vector", "id"])
	store = pd.HDFStore(vector_name + ".clean.h5")
	chunk = tweets.get_chunk()
	idx = 0
	while chunk is not None:
		print idx
		chunk["vector"] = chunk["vector"].apply(lambda x: eval(x.replace("WrappedArray(", "[").replace(")", "]").replace("null", "None")))
		print "Amount of None is: ", chunksize-chunk["vector"].dropna().count()
		vector = chunk["vector"].apply(pd.Series, 1)
		id = chunk["id"].to_frame()
		vector["id"] = id
		store.append("data", vector)
		chunk = tweets.get_chunk()
		idx+=1
	store.close()


def data_sample_tokens_id(data_name):
	import pandas as pd


	tweets = pd.read_csv(data_name, header=None, iterator=True, index_col=False, chunksize=1000, usecols=[0,2],
						 names=["id", "token"])
	chunk = tweets.get_chunk()
	while chunk is not None:
		chunk.index = chunk["id"]
		chunk["token"].to_csv("/media/cluster/data1/lambert/data_sample_tokens_id.csv", mode="a")
		chunk = tweets.get_chunk()


if __name__ == "__main__":
	data_name = "/media/cluster/data1/lambert/lambert_w2v_data_jan_all_columns.csv"
	data_name = "/media/cluster/data1/lambert/lambert_w2v_data_jan_all_columns3.csv"
	vector_name = "/media/cluster/data1/lambert/data_sample_vector_id"
	data_sample_vector_id(data_name, vector_name)
	tweet_name = "/media/cluster/data1/lambert/lambert_w2v_data_jan_tweet_id.csv"
	# data_sample_tweet_id(data_name, tweet_name)
	# data_sample_tokens_id(data_name)
	model_name = "/media/cluster/data1/lambert/lambert_model.csv"
	# convert_lookup_to_hdf(model_name)
