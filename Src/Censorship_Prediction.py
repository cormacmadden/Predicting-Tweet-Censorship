from tensorflow import keras
import numpy as np
import tensorflow as tf
import os
import json
import gzip
import googleSentimentAnalysis
import pandas as pd

fileDir = os.path.dirname(__file__)

censored_tweets = pd.read_csv(os.path.join(fileDir, 'tweet_text.csv'))
print(censored_tweets.head()) # print first 5 rows

for tweet in censored_tweets.text:
    print(tweet)
    #Add to censored tweets dataframe

