def onehot(x, hashtags):
	import numpy as np
	tmp = np.zeros(len(hashtags))
	tmp[x] = 1
	return tmp
