
import sys
from pymongo import MongoClient


connection = MongoClient("zardoz.service.rug.nl", 27017)
db = connection["twitter-2014-01"]
db.authenticate("guest", "guest")

for tweet in db.tweets.find(spec={}, fields=["text", "user.screen_name"]):

    print tweet["text"]
