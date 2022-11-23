import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import json
import gzip

fileDir = os.path.dirname(__file__)
mypath = os.path.join(fileDir, '../Data/20210823')
onlyfiles = [ f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) ]
tweets = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    with gzip.open(os.path.join(mypath,onlyfiles[n]), mode="rb") as f:
        tweets = [json.loads(line) for line in f]
    if "withheld_in_countries" in tweets[n]:
        print("test")
        if tweets[n]['withheld_in_countries'] != [] :
            print(tweets[n]['text'])



'''
tf.keras.preprocessing.text.text_to_word_sequence(
    input_text,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True,
    split=' '
)
'''