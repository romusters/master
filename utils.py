import tweepy
from langdetect.lang_detect_exception import LangDetectException
import nltk
from nltk.corpus import stopwords
from langdetect import detect_langs
stop = stopwords.words('english')
from nltk.tokenize import RegexpTokenizer
toker = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
import sklearn

import sys
import utils
import logging

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)
punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~'

class Utils:
	locationCount = {}
	api = None
	user = None
	auth = None
	consumer_key = None
	consumer_secret = None
	access_token = None
	access_token_secret = None

	def __init__(self, username):
		import ConfigParser
		config = ConfigParser.ConfigParser()
		config.read('config.cfg')
		self.consumer_key = config.get('key', 'consumer_key')
		self.consumer_secret = config.get('key', 'consumer_secret')
		self.access_token = config.get('token', 'access_token')
		self.access_token_secret = config.get('token', 'access_token_secret')
		self.auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
		self.auth.set_access_token(self.access_token, self.access_token_secret)
		self.api = tweepy.API(self.auth)
		self.user = self.api.get_user(username)
		print "Username: ", self.user.screen_name
		print "Location of user: ", self.user.location
		print "Follower count of user: ", self.user.followers_count

	#infer location of a tweet which does not have geo-location data by looking at befriended users
	def inferLocation(self, user):
		for friend in user.friends():
			try:
				location = toker.tokenize(self.api.get_user(friend.screen_name).location)[0]
			except IndexError:
				continue
			if location not in stop:
				if location in self.locationCount.keys():
					self.locationCount[location] = self.locationCount[location] + 1
				else:
					self.locationCount[location] = 1
		import operator
		print max(self.locationCount.iteritems(), key=operator.itemgetter(1))[0]

def kullback_leibner_divergence(labels_true, labels_pred, contingency=None):
	return sklearn.metrics.mutual_info_score(labels_true, labels_pred)

def loadAbbreviations():
	#http://www.webopedia.com/quick_ref/textmessageabbreviations.asp
	f = open("../resources/abbreviations.txt")
	lines = f.readlines()
#        print lines[5].lower()
	idx = 1
	length = len(lines)
	print length
	abb = {}
	while idx < length-1:
		abb[lines[idx].lower().rstrip()] = lines[idx+1].lower().rstrip()
		idx = idx + 2
	return abb

def convertAbbreviation(abbreviations, word):
	try:
		abbreviation = abbreviations[word]
		logger.info("Converted abbreviation is: %s", abbreviation)

		return abbreviation
	except Exception:
		return None

def detectLang(word):
	try:
		lang = detect_langs(word.decode('utf-8'))
		logger.info("Token: %s with language %s", word, lang)
		return lang
	except LangDetectException:
		return None


def convertNumber(word):
	for c in word:
		if c.isdigit():
			pass

def testData(file):
	import json
	for line in open(file, 'r'):
		try:
			tweet = json.loads(line)['text']
			logger.info(tweet)
		except KeyError:
			logger.info("No text key in JSON")
			continue
		except ValueError:
			logger.info("No JSON object could be decoded")
			continue

#load a file with tweets getting sentences in return with or without geolocation
def loadData(fname, geo=False):
	#Don't forget to use the geolocation infer method from the article.
	logger.info("Loading data with geo: %s", geo)
	import json
	tweets = []
	sentence = []
	geolocations = []
	for line in open(fname, 'r'):
		jsonData = json.loads(line)
		if geo:
			if jsonData['geo'] != None:
				logger.info("Geolocation is: %s", jsonData['geo']['coordinates'])
				geolocations.append(jsonData['geo']['coordinates'])
				tweet = jsonData['text']
				for t in tweet.split():
					sentence.append(t)
				tweets.append(sentence)
				sentence = []
			else:
				continue
		else:
			tweet = jsonData['text']
			for t in tweet.split():
				sentence.append(t)
			tweets.append(sentence)
			sentence = []

	return tweets, geolocations

def extractAllData(dir, value):
	import os
	import gzip
	tweets = []
	for file in os.listdir(dir):
		if file.endswith(".gz"):
			with gzip.open(dir + '/' + file, 'rb') as f:
				file_content = f.read()
				outF = open(dir + '/' + file[0:-3], 'wb')
				outF.write(file_content)
				outF.close()

	return tweets

def getSpecificData(dir, value):
	import json
	import os
	tweets = []
	for file in os.listdir(dir):
		if file.endswith(".out"):
			logger.info('Topic being extracted from: %s', dir + '/' + file)
			for line in open(dir + '/' + file, 'r'):
				try:
					tweet = json.loads(line)['text']
				except KeyError:
					logger.info("No text key in JSON for tweet: %s", tweet)
					continue
				except ValueError:
					logger.info("No JSON object could be decoded for tweet: %s", tweet)
					continue
				if value in tweet:
					tweets.append(tweet.split())

	return tweets

def getSpecificTextData(dir, value):
	tweets = []
	import os
	for file in os.listdir(dir):
		if file.endswith(".out"):
			logger.info('Topic being extracted from: %s', dir + '/' + file)
			for line in open(dir + '/' + file, 'r'):
				if value in line:
					tweets.append(line)
			return tweets
	return tweets


def loadAllData(fdir, value=None, filter=False, save=False):
	import os
	import pickle
	import json
	#dirs = os.listdir(fdir)
	#for dir in dirs:
	files = os.listdir(fdir)

	try:
		sentences = pickle.load( open(fdir + "filtered_sentences.p", 'rb') )
		return sentences
	except Exception, err:
		tweets = []
		filtered_tweets = []
		for fname in files:
			logger.info("Reading file: %s", fname)
			if fname.endswith(".out"):
				for line in open(fdir + fname, 'r'):
					tweet = json.loads(line)['text']
					if value is None:
						if filter is True:
							tweet = tweet.split()
							filtered_tweet = []
							for word in tweet:
								w = utils.filter(word)
								logger.debug("w has the value %s: ", w)
								if w is True:
									continue
								else:
									filtered_tweet.append(w)
							filtered_tweets.append(filtered_tweet)
							logger.debug("Tweet read and filtered: %s", filtered_tweet)
						else:
							tweets.append(tweet)
							logger.debug("Tweet read and not filtered: %s", tweet)
					elif value in tweet:
						tweets.append(tweet)
	if save:
		with open(fdir + "filtered_sentences.p", 'wb') as f:
			pickle.dump(filtered_tweets, f)
	if filter:
		return filtered_tweets
	else:
		return tweets


def checkWords(sentence, model):
	tweet = []
	for word in sentence.split():
		try:
			model.most_similar(positive=word)
			tweet.append(word)
		except KeyError:
			continue

	return tweet

def getTags(data):
	result = []
	for tweet in data:
		for token in tweet:
			if '#' in token:
				#logger.info("tag: %s", token[0])
				result.append(token)
	logger.info("Number of tags is: %f", len(result))
	return result

def getFreqTags(hashtags):
	logger.info("Starting determining frequency of tags, number of tweets is: %f", len(hashtags))
	freq = {}
	for tag in hashtags:
			try:
				if tag in freq:
					freq[tag] = freq[tag] + 1
				else:
					freq[tag] = 1
			except KeyError:
				continue
	import operator
	sortedList = sorted(freq.items(), key=operator.itemgetter(1), reverse=True)
	logger.info("The frequency of first 10 tags is: %s", sortedList[0:10])

def similar(abbrevation, word):
	from difflib import SequenceMatcher
	if abbrevation == None:
		logger.info("No abbrevation for word: %s", word)
		return -1.0
	else:
		prob = SequenceMatcher(None, abbrevation, word).ratio()
	logger.info("Similarity probability for token is: %s is %f", word, prob)
	return prob

#Under construction
def metaphone(word):
	#import metaphone
	#met = metaphone.doublemetaphone(word)
	if word == None:
		return " "
	import fuzzy
	met = fuzzy.nysiis(word)
	logger.info("Metaphone for token %s is %s", word, met)
	return met

def checkDigit(word):
	punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~'
	word = word.encode('utf-8')
	word = word.translate(None, punctuation)
	if word.isdigit():
		return True
	return False

def filter(w):

	stemmer =  nltk.stem.snowball.DutchStemmer()
	logger.debug("Start to check word: %s", w)
	#filter stopwords
	if w in stopwords.words("dutch"):
		logger.debug("Filter %s as stopword", w)
		return True
	#filter url
	if 'http' in w or 'www' in w:
		logger.debug("Filter %s as url", w)
		return True
	#if word contains a hashtag, do not stem
	if '#' in w:
		logger.debug("Word %s is a hashtag", w)
		return w
	if '@' in w:
		logger.debug("Word %s is a mention", w)
		return w
	if checkDigit(w):
		logger.debug("Filter %s as a number", w)
		return True
	else:
		logger.debug("Stem and add word: %s", w)
		w = stemmer.stem(w)
		return w
	#perhaps use metaphones in the future? Using speech sound: "inb4" to "inbefore"
	#perhaps normalize using linux dictionary Aspell with edit distance ftp:#ftp.gnu.org/gnu/aspell/dict/0index.html

def filterTweet(tweet):
	tweet = tweet.split()
	filtered_tweet = []
	for word in tweet:
		w = utils.filter(word)
		if w is True:
			continue
		else:
			filtered_tweet.append(w)
	return filtered_tweet


def getBagOfwords(tweets):
	bow = set()
	for tweet in tweets:
		if len(tweet) == 0:
			logger.info("Empty tweet: %s", tweet)
			continue

		for w in tweet:
			if filter(w) is not True:
				bow.add(w)

	bow2 = set()

	for b in bow:
		b = b.encode("ascii", "ignore")
		b = b.translate(None, punctuation)
		bow2.add(b)

	logger.info("The bag of words are: %s", bow2)
	return bow


def main():
	#username = 'romusters'
	#utils = Utils(username)
	#utils.inferLocation(utils.user)
	#abbreviations = utils.loadAbbreviations()
	fname = '/home/robert/data/2015123101/'
	tweets = loadAllData(fname, filter=True, save=True)
	getBagOfwords(tweets)

	#print abbreviations

if __name__ == "__main__":
	main()

