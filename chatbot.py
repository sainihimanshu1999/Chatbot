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

# create training data
training = []
output = []
# create an empty array for output
output_empty = [0] * len(classes)

# create training set, bag of words for each sentence
for doc in documents:
    # initialize bag of words
    bag = []
    #tokenized words for the pattern
    pattern_words = doc[0]
    # stemming
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is '1' for current tag and '0' for rest of other tags
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffling features and turning it into np.array
random.shuffle(training)
training = np.array(training)

# creating training lists
train_x = list(training[:,0])
train_y = list(training[:,1])


# resetting underlying graph data
tf.reset_default_graph()

# Building neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Defining model and setting up tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# Start training
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')


#Importing pickle
import pickle
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )


#restoring all the datastructures
data = pickle.load(open("training_data", "rb"))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

with open('intents.json') as json_data:
  intents = json.load(json_data)

  #loading the saved model
model.load('./model.tflearn')



#training the model
def clean_up_sentence(sentence):
  sentence_words = nltk.word_tokenize(sentence)
  sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
  return sentence_words


def bow(sentence, words, show_details = False):
  sentence_words = clean_up_sentence(sentence)
  bag = [0]*len(words)
  for s in sentence_words:
    for i,w in enumerate(words):
      if w == s:
        bag[i] = 1
        if show_details:
          print('found in bag: %s' %w)
    return(np.array(bag))




    ERROR_THRESHOLD = 0.30
def classify(sentence):
   
    results = model.predict([bow(sentence, words)])[0]
   
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
   
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    
    if results:
        
        while results:
            for i in intents['intents']:
                
                if i['tag'] == results[0][0]:
                    
                    return print(random.choice(i['responses']))

            results.pop(0)