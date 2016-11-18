import pandas as pd
import numpy as np

filename = 	'/media/cluster/data1/data_sample.csv.clean.h5'


def remove_all():
    store = pd.HDFStore(filename)
    keys = store.keys()
    for key in keys:
        store.remove(key)
    print store

def convert_data_sample():
    data = pd.read_csv('/media/cluster/data1/data_sample.csv.clean', header=None, chunksize=1000)
    store = pd.HDFStore(filename)
    for i in range(258):
        print i
        try:
            d = data.get_chunk()
            inds = d[3].to_frame()
            vectors = d[2].to_frame()
            vectors = vectors[2].apply(lambda x: eval(x))
            vectors.index = inds[3].values.tolist()
            vectors = vectors.to_frame()
            vectors = vectors[2].apply(pd.Series, 1)
            # df.index = d[3].values.tolist()


            store.append('data', vectors)
        except:
            break
    store.close()

# remove_all()
# convert_data_sample()
# store = pd.HDFStore(filename)
# print store

def get_data_by_indices(filename, indices):
    store = pd.HDFStore(filename)
    store.select("data", where=store["data"].index.isin(indices))


from w2v import  cosine_similarity
def calc_sim_fast():
    store = pd.HDFStore(filename)
    nrows = store.get_storer('data').nrows
    chunksize = 1000

    seed = store.select('data', start=0, stop=1).values.tolist()[0]
    print seed
    sims = []
    for i in range(257):
        print i
        chunk = store.select('data', start=i*chunksize, stop=(i+1)*chunksize)
        iter = chunk.iterrows()

        for row in iter:
            sims.append([row[0], cosine_similarity(row[1].values.tolist(), seed)])
    print sims

    result = pd.DataFrame(sims)
    print result.keys()
    result = result.sort_values(by=1, ascending=False)
    result.to_csv('/media/cluster/data1/data_sample.csv.clean.sims', index=False)

calc_sim_fast()