
import logging
import sys
import gensim

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class DirOfPlainTextCorpus(object):
	def __init__(self, fname):
		self.fname = fname

	def __iter__(self):
		text = open(self.fname).read()
		for sentence in text:
			yield sentence.split()


def train_model():
		dname = '/data/s1774395/text/input.txt'
		logger.info("Loading data")

		#Method 1
		#sentences = gensim.models.word2vec.LineSentence(dname)

		#Method 2
		#sentences = [d.split() for d in open(dname).readlines()]

		#Method 3

		logger.info("Training model")
		try:
			#model = gensim.models.word2vec.Word2Vec(sentences, size=100000, window=5, min_count=5, workers=24)
			model = gensim.models.word2vec.Word2Vec(DirOfPlainTextCorpus(dname), size=100000, window=5, min_count=5, workers=24)

		except UnicodeDecodeError:
			import traceback
			traceback.print_exc()

		logger.info("Saving model")
		fname = "/data/s1774395/fullGensimPeregrineModel.bin"
		model.save(fname)


def test_model():
	mname = '/data/s1774395/8GgensimPeregrineModel.bin'
	model = gensim.models.Word2Vec.load(mname)
	tweets = ['ik heb geen zin meer in Twitter', '@celine_maas ja joh mn zus kijkt het :(', '@RiannePathuis ik ook niet :P']
	for tweet in tweets:
		#using topn you can acquire the feature vector
		print model.most_similar(tweet.split(), topn=False)


max_int_size = 268435455
vocab_size = 34179832
vector_size = max_int_size / vocab_size



stopwords = ['de', 'en', 'van', 'ik', 'te', 'dat', 'die', 'in', 'een', 'hij', 'het', 'niet', 'zijn', 'is', 'was', 'op', 'aan', 'met', 'als', 'voor', 'had', 'er', 'maar', 'om', 'hem', 'dan', 'zo', 'of', 'wat', 'mijn', 'men', 'dit', 'zo', 'door', 'over', 'ze', 'zich', 'bij', 'ook', 'tot', 'je', 'mij', 'it', 'der', 'daar', 'haar', 'naar', 'heb', 'hoe', 'heeft', 'hebben', 'deze', '', 'want', 'nog', 'zal', 'me', 'zij', 'n', 'ge', 'geen', 'omdat', 'iets', 'worden', 'toch', 'al', 'waren', 'veel', 'meer', 'doen', 'toen', 'moet', 'ben', 'zonder', 'kan', 'hn', 'ds', 'alles', 'onder', 'ja', 'eens', 'hier', 'wie', 'werd', 'altijd', 'doch', 'wordt', 'wezen', 'knnen', 'ons', 'zelf', 'tegen', 'na', 'reeds', 'wil', 'kon', 'niets', 'w', 'iemand', 'geweest', 'andere']

def mv_tags(tweet):
	words = tweet.split(" ")
	for i, word in enumerate(words):
		if "http" in word:
			words[i] = "<URL>"
		if "@" in word:
			words[i] = "<MENTION>"
		if word in stopwords:
			words[i] = "<STOPWORD>"
	return " ".join(words)

#/data/s1774395/text/2015/01/20150101:00.out.gz.txt

def rm_encoding(s):
	#former ascii
	return s.decode('utf-8').encode('utf-8','ignore')


def filter(s):
	s = rm_encoding(s)
	s = mv_tags(s)
	return s


def iterative_training():

	mdir = '/data/s1774395/models/'
	ddir = '/data/s1774395/text/'

	import os
	mname = "gensimIterModel.bin"

	topdir = os.listdir(ddir)
	topdir.sort()
	for middir in topdir:
		bottomdir = os.listdir(ddir + middir)
		bottomdir.sort()
		for subdir in bottomdir:
			files = os.listdir(ddir + middir + "/" + subdir)
			files.sort()
			print files

			for file in files:
				try:
					fname = ddir +  middir + "/" + subdir + "/" + file

					logger.info(fname)

					done_files = open(mdir + "good_files", 'r')
					done_files = done_files.readlines()
					cp_done_files = []
					for file in done_files:
						cp_done_files.append(file.strip())
					done_files = cp_done_files

					#done_files = [d.strip() for d in open(mdir + "good_files", 'r').readlines()]

					if fname in done_files:
						logger.info("Already trained on file: " + fname)
						continue

					#sentences = [line.split() for line in open(fname).readlines()]
					sentences = []
					for s in open(fname):
						sentences.append(filter(s).split(" "))
					print sentences[0]
					if os.path.isfile(mdir + mname):
						logger.info("loading model")
						model = gensim.models.word2vec.Word2Vec.load(mdir + mname)
						try:
							model.train(sentences)
							gensim.models.word2vec.Word2Vec.save(model, mdir + mname)
							g = open(mdir + "good_files", 'a')
							g.write(fname + '\n')
							g.close()
						except:
							import traceback
							traceback.print_exc()
							b = open(mdir + "bad_files", 'a')
							b.write(fname + '\n')
							b.close()
							continue
					else:
						try:
							logger.info("Training model")
							#size was previously 100000
							model = gensim.models.word2vec.Word2Vec(sentences, size=vector_size, window=5, min_count=5, workers=24)
							gensim.models.word2vec.Word2Vec.save(model, mdir + mname)
							g = open(mdir + "good_files", 'a')
							g.write(fname + "\n")
							g.close()
						except:
							import traceback
							traceback.print_exc()
							continue

				except Exception:
					import traceback
					traceback.print_exc()
					continue
				try:
					os.system("cp " + mdir + mname + " " + mdir + "copy_" + mname)
				except:
					logger.info("model copy error")



def get_dates_freq(fname, year):

	logger.info("Starting collecting dates")
	import dateutil.parser
	import sys
	from datetime import date
	last_day = date(year, 12, 31)
	day_freqs = {}
	days = range(1,366,1)
	for i in days:
		day_freqs[str(i)] = 0

	old_day_number = -1

	logger.info("opening file")
	f = open(fname, 'r')
	logger.info("reading lines")
	lines = f.readlines()
	f.close()
	logger.info("length lines%i", len(lines))
	for line in lines:
		#print line
		date = dateutil.parser.parse(line)
		if date.year == year:
			day_number = (365 + (date.date() - last_day).days) + 1 #+1 is correct?
			if day_number == old_day_number:
				day_freqs[str(day_number)] += 1
				pass
			else:
				#print day_number
				old_day_number = day_number
				day_freqs[str(day_number)] += 1
		if date.year == year +1:
			#print day_freqs
			sys.exit(0)

	import json
	json.dump(day_freqs, open("/data/s1774395/dates2015", "w"))

if __name__ == "__main__":
	#train_model()
	iterative_training()
	year = 2015
	fname = "/data/s1774395/dates.txt"
	#get_dates_freq(fname, year)

