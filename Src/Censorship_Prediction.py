from tensorflow import keras
import numpy as np
import tensorflow as tf
import os
import json
import gzip
import googleSentimentAnalysis
import pandas as pd

fileDir = os.path.dirname(__file__)
mypath = os.path.join(fileDir, '../Data/20210823')
onlyfiles = [ f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) ]
tweets = np.empty(len(onlyfiles), dtype=object)
#for n in range(0, len(onlyfiles)):
with gzip.open(os.path.join(mypath,onlyfiles[0]), mode="rb") as f:
    tweets = [json.loads(line) for line in f]
    
for n in range(0, len(tweets)):
    if "created_at" in tweets[n]:    
        if "withheld_in_countries" in tweets[n]:
            print("test")
            if tweets[n]['withheld_in_countries'] != [] :
                print(tweets[n]['text'])




censored_tweets = pd.read_csv('tweet_text.csv')
print(censored_tweets.head()) # print first 5 rows

for tweet in censored_tweets:
    tweet_entities_sentiment = googleSentimentAnalysis.sample_analyze_entity_sentiment(tweet)
    #Add to censored tweets dataframe
