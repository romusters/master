import logging
import sys
import gensim

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def train_model():
	dname = '/data/s1774395/201201.tar.gz'
	sentences = gensim.models.word2vec.LineSentence(dname, max_sentence_length=150)
	model = gensim.models.word2vec.Word2Vec(sentences, size=100, window=5, min_count=5, workers=24)
	fname = "/data/s1774395/gensimPeregrineModel.bin"
	model.save(fname)
	print model.most_similar(positive=['word'], negative=['ik'])

def test_model():
	mname = '/data/s1774395/8GgensimPeregrineModel.bin'
	model = gensim.models.Word2Vec.load(mname)
	tweets = ['ik heb geen zin meer in Twitter', '@celine_maas ja joh mn zus kijkt het :(', '@RiannePathuis ik ook niet :P', 'Ik hoop egt dat morge de juf beter is anders zitte we weer een hele klote dag bij juf suzan !', 'Een burger eten als een amerikaan: zoveel mogelijk met je voortanden kauwen en zo minmogelijk je tong gebruiken om te proeven.', '@Ilona1970 sorry, laat het volgende keer eerder weten... kun je de mijne ook doen ;)', '@gabvirtualworld die wolowitz is echt een smerig mannetje.']

	for tweet in tweets:
		print model.most_similar(positive=[tweet])

if __name__ == "__main__":
	#train_model()
	test_model()

