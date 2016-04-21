#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import gensim
import utils
import os
import sys
import logging
import word2vec
import word2vecReader

#sys.setdefaultencoding()

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

dir = '/media/robert/dataThesis/tekst/'




# utils.encodeTextToUTF8(dname)

sentences = gensim.models.word2vec.LineSentence(dname, max_sentence_length=150, limit=1000)

w2v = word2vec.Word2vec()
model = w2v.trainModel(sentences, False)

mname = "/home/robert/data/gensimModel.bin"
w2v.saveModel(model, mname)
