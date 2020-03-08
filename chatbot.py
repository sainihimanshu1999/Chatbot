import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import tensorflow as tf
import numpy as np
import tflearn
import random
import json

from google.colab import files
files.upload()


#import chat bot intents file
with open('intents.json') as json_data:
  intents = json.load(json_data)

#running intent file

words = []
classes = []
documents = []
ignore = ['?']

#looping through each sentence in the json file's pattern
for intent in intents['intents']:
  for pattern in intent['patterns']:
    #tokeninzing each word in the sentence
    w = nltk.word_tokenize(pattern)
    #add word to the word list
    words.extend(w)
    #adding words to the document
    documents.append((w , intent['tag']))
    #adding tags to the class list
    if intent['tag'] not in classes:
      classes.append(intent['tag'])


  
#lowering the cases of the words, stemming them, and simultaneously removing the duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore]
words = sorted(list(set(words)))

#removing duplicate classes
classes = sorted(list(set(classes)))

print(len(documents) , "documents")
print(len(classes) , "classes" , classes)
print(len(words) , "unique stem words" , words)
