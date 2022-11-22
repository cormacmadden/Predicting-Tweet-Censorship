import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import json
import gzip

fileDir = os.path.dirname(__file__)
mypath = os.path.join(fileDir, '../Data/20210823')
onlyfiles = [ f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) ]
files = np.empty(len(onlyfiles), dtype=object)
#for n in range(0, len(onlyfiles)):
#    with gzip.open(os.path.join(mypath,onlyfiles[n]), mode="rt") as f:
#        files[n] = f.read()
for n in range(0, 3):
    with gzip.open(os.path.join(mypath,onlyfiles[n]), mode="rt") as f:
        files[n] = f.read()

print(files[0])

text_file = open("test.json", "w")
n = text_file.write(json.dumps(files[0]))
text_file.close()

'''
tf.keras.preprocessing.text.text_to_word_sequence(
    input_text,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True,
    split=' '
)
'''