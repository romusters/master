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
		word = word.lower()
		if "http" in word:
			words[i] = "<URL>"
		elif word[0:2] == "06" and len(word) == 10:
			words[i] = "<MOBIEL>"
		elif "@" in word:
			words[i] = "<MENTION>"
		elif word in stemming.stopwords:
			words[i] = "<STOPWORD>"
		elif "haha" in word:
			words[i] = "haha"
		elif "#" in word:
			words[i] = word.replace("#", "")
		else:
			# split strings containing numbers and strings in to constituents.
			import re
			parts = re.split('(\d+)', words[i])
			tmp = []
			for part in parts:
				if len(part) is not 0:
					tmp.append(stemming.stem(part))
			words[i] = tmp


			# words[i] = stemming.stem(word)
	return " ".join([item for sublist in words for item in sublist]) # unravel string into flat list
	# return " ".join(words)
