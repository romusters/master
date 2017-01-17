# try to do multiprocessing for cosine similarity on pandas dataframe

import pandas as pd
# import multiprocessing as mp
store = pd.HDFStore("/media/cluster/data1/lambert/data_sample_vector_id.clean.h5")
vectors = store["data"][range(70)]
df = store["data"]
id = 644245097431

seed_vector = store.select("data",  where=store["data"].id.isin([id]).drop("id", axis=1).values.tolist())

from sklearn.metrics.pairwise import pairwise_distances
distances = pairwise_distances(seed_vector, vectors, n_jobs=-1)
# def worker():
#
# if __name__ == '__main__':
#     pool = mp.Pool(processes = (mp.cpu_count() - 1))
#     pool.map(calc_dist, ['lat','lon'])
#     pool.close()
#     pool.join()