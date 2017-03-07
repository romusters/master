import stemmer

stemming = stemmer.Stemmer()

def filter(s):
	s = s.lower()
	s = s.strip()
	s = rm_encoding(s)
	s = rm_punctuation(s)
	s = mv_tags(s)
	return s

def rm_encoding(s):
# Peregrine
# 	return s.decode('utf-8').encode('ascii', 'ignore')

# Hadoop
	return s.encode('ascii','ignore')


def rm_punctuation(s):
	s = s.encode('utf-8')
	s = s.translate(None, stemming.punctuation)
	return s

def mv_tags(tweet):
	import re
	_digits = re.compile('\d')
	words = tweet.split()
	for i, word in enumerate(words):
		# word = word.strip()
		if "http" in word:
			words[i] = "<url>"
		elif word[0:2] == "06" and len(word) == 10 and word.isdigit():
			words[i] = "<mobiel>"
		elif word.isdigit():
				if int(word) == 0:
					words[i] = "0"
				else:
					words[i] = word.strip().lstrip("0").strip() # remove leading zeros
				print words[i]
		elif "@" in word:
			words[i] = "<mention>"
		elif word in stemming.stopwords:
			words[i] = "<stopword>"
		elif "haha" in word:
			words[i] = "haha"
		elif "#" in word:
			words[i] = word.replace("#", "")
		elif bool(_digits.search(word)):
			# split strings containing numbers and strings in to constituents.
			import re
			parts = re.split('(\d+)', words[i])
			tmp = []
			for part in parts:
				if len(part) is not 0:
					tmp.append(stemming.stem(part).strip().lstrip("0").strip())
			words[i] = " ".join(tmp).strip() #changed
		else:
			words[i] = stemming.stem(word).strip().lstrip("0")
			# return " ".join([item for sublist in words for item in sublist]) # unravel string into flat list
	return " ".join(words).strip()
