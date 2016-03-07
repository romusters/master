from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk.corpus import stopwords
from time import time
import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def LDA(data, tf, n_samples, n_features, bow):
	logger.info("Start LDA")
	n_topics = 5
	n_top_words = 309

	logger.info("Fitting LDA models with tf features, n_samples=%d and n_features=%d..."
		  % (n_samples, n_features))
	lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
									learning_method='online', learning_offset=50.,
									random_state=0)
	t0 = time()
	lda.fit(tf)
	logger.info("done in %0.3fs." % (time() - t0))

	logger.info("\nTopics in LDA model:")
	print_top_words(lda, bow, n_top_words)

def print_top_words(model, feature_names, n_top_words):
	for topic_idx, topic in enumerate(model.components_):
		print("Topic #%d:" % topic_idx)
		print(" ".join([feature_names[i]
						for i in topic.argsort()[:-n_top_words - 1:-1]]))
	print()
