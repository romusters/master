import stemmer

stemming = stemmer.Stemmer()

def filter(s):
	s = s.strip()
	s = rm_encoding(s)
	s = rm_punctuation(s)
	s = mv_tags(s)
	return s

def rm_encoding(s):
# Peregrine
#	return s.decode('utf-8').encode('ascii', 'ignore')

# Hadoop
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
