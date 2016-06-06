import stemmer

stemming = stemmer.Stemmer()

def filter(s):
	s = s.strip()
	s = rm_encoding(s)
	s = rm_punctuation(s)
	s = mv_tags(s)
	return s

def rm_encoding(s):
#	return s.decode('utf-8').encode('ascii', 'ignore')

	return s.encode('ascii','ignore')


def rm_punctuation(s):
	s = s.encode('utf-8')
	s = s.translate(None, stemming.punctuation)
	return s

def mv_tags(tweet):
	words = tweet.split(" ")
	for i, word in enumerate(words):
		if "http" in word:
			words[i] = "<URL>"
		elif "@" in word:
			words[i] = "<MENTION>"
		elif word in stemming.stopwords:
			words[i] = "<STOPWORD>"
		elif "haha" in word:
			words[i] = "haha"
		elif "#" in word:
			words[i] = word.replace("#", "")
		else:
			words[i] = stemming.stem(word)
	return " ".join(words)


def test(i):
	if i>1:
		print 1
	elif i>2:
		print 2
	else:
		print 3