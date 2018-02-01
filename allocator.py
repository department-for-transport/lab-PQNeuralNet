import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import string
import unicodedata
import sys

stemmer = LancasterStemmer()

#create table (a dictionary) to hold remove_punctuation
#unicodedata.category for a chactare is like 'punctuation, connector'
tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

#punctuation remover (cos keys in table point to nothing it just deletes the punctuation)
def remove_punctuation(sentence):
    return sentence.translate(tbl)

#load json data into var called json
with open('json2.txt', 'r') as json_data:
    data = json.load(json_data)

#list of categories ( needs to the same as a category list created when the model is trained!!! TODO!)
categories = sorted(list(data.keys()))
#list of words that holds all unique stemmed words
words = []

for each_category in data.keys():
    for each_sentence in data[each_category]:
        #remove punctuation
        each_sentence = remove_punctuation(each_sentence)
        #extract words from sentence and add to list 'words'
        w = nltk.word_tokenize(each_sentence)
        words.extend(w)

#stem and lower each word
words = [stemmer.stem(w.lower()) for w in words]
#remove duplicates (the set does this)
words = sorted(list(set(words)))

#net is expecting a shape the size of the bag of words
net = tflearn.input_data(shape=[None,len(words)])
#layer?
net = tflearn.fully_connected(net,8)
#layer?
net = tflearn.fully_connected(net,8)
#output layer
net = tflearn.fully_connected(net, len(categories),activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_dir = 'tflearn_logs')
model.load('model.tflearn')

def get_tf_record(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    BoW = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w==s:
                BoW[i]=1
    return(np.array(BoW))

while True:
    sent1 = input("Enter question: ")
    print("{}: {}".format(sent1, categories[np.argmax(model.predict([get_tf_record(sent1)]))]))
