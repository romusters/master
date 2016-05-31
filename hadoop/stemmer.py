class Stemmer:
	vowels = "aeiouy\xE8"
	step1_suffixes = ("heden", "ene", "en", "se", "s")
	step3b_suffixes = ("baar", "lijk", "bar", "end", "ing", "ig")
	stopwords = ['de', 'en', 'van', 'ik', 'te', 'dat', 'die', 'in', 'een', 'hij', 'het', 'niet', 'zijn', 'is', 'was', 'op', 'aan', 'met', 'als', 'voor', 'had', 'er', 'maar', 'om', 'hem', 'dan', 'zo', 'of', 'wat', 'mijn', 'men', 'dit', 'zo', 'door', 'over', 'ze', 'zich', 'bij', 'ook', 'tot', 'je', 'mij', 'it', 'der', 'daar', 'haar', 'naar', 'heb', 'hoe', 'heeft', 'hebben', 'deze', '', 'want', 'nog', 'zal', 'me', 'zij', 'n', 'ge', 'geen', 'omdat', 'iets', 'worden', 'toch', 'al', 'waren', 'veel', 'meer', 'doen', 'toen', 'moet', 'ben', 'zonder', 'kan', 'hn', 'ds', 'alles', 'onder', 'ja', 'eens', 'hier', 'wie', 'werd', 'altijd', 'doch', 'wordt', 'wezen', 'knnen', 'ons', 'zelf', 'tegen', 'na', 'reeds', 'wil', 'kon', 'niets', 'w', 'iemand', 'geweest', 'andere']
	punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~'

	def stem(self, word):
			"""
			Stem a Dutch word and return the stemmed form.
			:param word: The word that is stemmed.
			:type word: str or unicode
			:return: The stemmed form.
			:rtype: unicode
			"""
			word = word.lower()

			if word in self.stopwords:
				return word

			step2_success = False

			# Vowel accents are removed.
			word = (word.replace("\xE4", "a").replace("\xE1", "a")
						.replace("\xEB", "e").replace("\xE9", "e")
						.replace("\xED", "i").replace("\xEF", "i")
						.replace("\xF6", "o").replace("\xF3", "o")
						.replace("\xFC", "u").replace("\xFA", "u"))

			# An initial 'y', a 'y' after a vowel,
			# and an 'i' between vowels is put into upper case.
			# As from now these are treated as consonants.
			if word.startswith("y"):
				word = "".join(("Y", word[1:]))

			for i in range(1, len(word)):
				if word[i-1] in self.vowels and word[i] == "y":
					word = "".join((word[:i], "Y", word[i+1:]))

			for i in range(1, len(word)-1):
				if (word[i-1] in self.vowels and word[i] == "i" and
				   word[i+1] in self.vowels):
					word = "".join((word[:i], "I", word[i+1:]))

			r1, r2 = self.r1r2_standard(word, self.vowels)

			# R1 is adjusted so that the region before it
			# contains at least 3 letters.
			for i in range(1, len(word)):
				if word[i] not in self.vowels and word[i-1] in self.vowels:
					if len(word[:i+1]) < 3 and len(word[:i+1]) > 0:
						r1 = word[3:]
					elif len(word[:i+1]) == 0:
						return word
					break

			# STEP 1
			for suffix in self.step1_suffixes:
				if r1.endswith(suffix):
					if suffix == "heden":
						word = self.suffix_replace(word, suffix, "heid")
						r1 = self.suffix_replace(r1, suffix, "heid")
						if r2.endswith("heden"):
							r2 = self.suffix_replace(r2, suffix, "heid")

					elif (suffix in ("ene", "en") and
						  not word.endswith("heden") and
						  word[-len(suffix)-1] not in self.vowels and
						  word[-len(suffix)-3:-len(suffix)] != "gem"):
						word = word[:-len(suffix)]
						r1 = r1[:-len(suffix)]
						r2 = r2[:-len(suffix)]
						if word.endswith(("kk", "dd", "tt")):
							word = word[:-1]
							r1 = r1[:-1]
							r2 = r2[:-1]

					elif (suffix in ("se", "s") and
						  word[-len(suffix)-1] not in self.vowels and
						  word[-len(suffix)-1] != "j"):
						word = word[:-len(suffix)]
						r1 = r1[:-len(suffix)]
						r2 = r2[:-len(suffix)]
					break

			# STEP 2
			if r1.endswith("e") and word[-2] not in self.vowels:
				step2_success = True
				word = word[:-1]
				r1 = r1[:-1]
				r2 = r2[:-1]

				if word.endswith(("kk", "dd", "tt")):
					word = word[:-1]
					r1 = r1[:-1]
					r2 = r2[:-1]

			# STEP 3a
			if r2.endswith("heid") and word[-5] != "c":
				word = word[:-4]
				r1 = r1[:-4]
				r2 = r2[:-4]

				if (r1.endswith("en") and word[-3] not in self.vowels and
					word[-5:-2] != "gem"):
					word = word[:-2]
					r1 = r1[:-2]
					r2 = r2[:-2]

					if word.endswith(("kk", "dd", "tt")):
						word = word[:-1]
						r1 = r1[:-1]
						r2 = r2[:-1]

			# STEP 3b: Derivational suffixes
			for suffix in self.step3b_suffixes:
				if r2.endswith(suffix):
					if suffix in ("end", "ing"):
						word = word[:-3]
						r2 = r2[:-3]

						if r2.endswith("ig") and word[-3] != "e":
							word = word[:-2]
						else:
							if word.endswith(("kk", "dd", "tt")):
								word = word[:-1]

					elif suffix == "ig" and word[-3] != "e":
						word = word[:-2]

					elif suffix == "lijk":
						word = word[:-4]
						r1 = r1[:-4]

						if r1.endswith("e") and word[-2] not in self.vowels:
							word = word[:-1]
							if word.endswith(("kk", "dd", "tt")):
								word = word[:-1]

					elif suffix == "baar":
						word = word[:-4]

					elif suffix == "bar" and step2_success:
						word = word[:-3]
					break

			# STEP 4: Undouble vowel
			if len(word) >= 4:
				if word[-1] not in self.vowels and word[-1] != "I":
					if word[-3:-1] in ("aa", "ee", "oo", "uu"):
						if word[-4] not in self.vowels:
							word = "".join((word[:-3], word[-3], word[-1]))

			# All occurrences of 'I' and 'Y' are put back into lower case.
			word = word.replace("I", "i").replace("Y", "y")


			return word

	def suffix_replace(self, original, old, new):
		"""
		Replaces the old suffix of the original string by a new suffix
		"""
		return original[:-len(old)] + new

	def r1r2_standard(self, word, vowels):
		"""
		Return the standard interpretations of the string regions R1 and R2.
		R1 is the region after the first non-vowel following a vowel,
		or is the null region at the end of the word if there is no
		such non-vowel.
		R2 is the region after the first non-vowel following a vowel
		in R1, or is the null region at the end of the word if there
		is no such non-vowel.
		:param word: The word whose regions R1 and R2 are determined.
		:type word: str or unicode
		:param vowels: The vowels of the respective language that are
					   used to determine the regions R1 and R2.
		:type vowels: unicode
		:return: (r1,r2), the regions R1 and R2 for the respective word.
		:rtype: tuple
		:note: This helper method is invoked by the respective stem method of
			   the subclasses DutchStemmer, FinnishStemmer,
			   FrenchStemmer, GermanStemmer, ItalianStemmer,
			   PortugueseStemmer, RomanianStemmer, and SpanishStemmer.
			   It is not to be invoked directly!
		:note: A detailed description of how to define R1 and R2
			   can be found at http://snowball.tartarus.org/texts/r1r2.html
		"""
		r1 = ""
		r2 = ""
		for i in range(1, len(word)):
			if word[i] not in self.vowels and word[i-1] in self.vowels:
				r1 = word[i+1:]
				break

		for i in range(1, len(r1)):
			if r1[i] not in self.vowels and r1[i-1] in self.vowels:
				r2 = r1[i+1:]
				break

		return (r1, r2)

# if __name__ == "__main__":
# 	s = Stemmer()
# 	print s.stem("lopen")