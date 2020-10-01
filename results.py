import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import re
import tensorflow as tf
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


def preprocess(text, stem=False):
  text = re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", ' ', str(text).lower()).strip()
  stop_words = stopwords.words('english')
  stemmer = SnowballStemmer('english')
  tokens = []
  for token in text.split():
    if token not in stop_words:
      if stem:
        tokens.append(stemmer.stem(token))
      else:
        tokens.append(token)
  return " ".join(tokens)

inp = input()
inp = preprocess(inp)
vect = pickle.load(open("vectorizer.pickle",'rb'))
inp = vect.transform([inp]).toarray().reshape(1,1,2500)
mod = tf.keras.models.load_model('model.h5')
out = mod.predict(inp)
print(out[0][0])
