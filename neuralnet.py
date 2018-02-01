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

#SORT OUT DATA#

#create table (a dictionary) that holds all punctuation characters for removal
#unicodedata.category for a character is like 'punctuation, connector'
tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

#punctuation remover (cos keys in tbl point to nothing it just deletes the punctuation)
def remove_punctuation(sentence):
    return sentence.translate(tbl)


#initialise LancasterStemmer (stems words quite harshly)
stemmer = LancasterStemmer()

#initialise pointer to data file
data = None

#load json data into var called json
with open('json2.txt', 'r') as json_data:
    data = json.load(json_data)


#get categories from json (ie business unit)
categories = sorted(list(data.keys()))
print(categories)

#list of words that holds all unique stemmed words
words = []

#list of tuples of words and units
docs = []

for each_category in data.keys():
    for each_sentence in data[each_category]:
        #remove punctuation
        each_sentence = remove_punctuation(each_sentence)
        #extract words from sentence and add to list 'words'
        w = nltk.word_tokenize(each_sentence)
        words.extend(w)
        #add list of words from stenence and category as a tuple to docs
        docs.append((w, each_category))

#stem and lower each word
words = [stemmer.stem(w.lower()) for w in words]
#remove duplicates (the set does this)
words = sorted(list(set(words)))

#list to hold training data
training = []
# create an array of 0s for our output
output_empty = [0] * len(categories)



for doc in docs:
    # initialize our bag of words(bow) for each document in the list
    bow = []
    # list of tokenized words for the pattern
    token_words = doc[0]
    # stem each word
    token_words = [stemmer.stem(word.lower()) for word in token_words]
    # create our bag of words array
    for w in words:
        bow.append(1) if w in token_words else bow.append(0)

    output_row = list(output_empty)
    output_row[categories.index(doc[1])] = 1

    # our training set will contain a the bag of words model and the output row that tells which catefory that bow belongs to.
    training.append([bow, output_row])


# shuffle our features and turn into np.array as tensorflow  takes in numpy array
random.shuffle(training)
training = np.array(training)

# train_x contains the Bag of words and train_y contains the label/ category
train_x = list(training[:,0])
train_y = list(training[:,1])

#DESIGN MODEL#

#reset underlying graph data
tf.reset_default_graph()
#Build neural network
#input layer - size is the same as the input matrix (ie length of the BoW)
net = tflearn.input_data(shape=[None,len(train_x[0])])
#Hidden layer1 of size 8
net = tflearn.fully_connected(net,8)
#Hidden layer2 of size 8
net = tflearn.fully_connected(net,8)
#output layer - size is dictated by the number of units we're allocating cases to
net = tflearn.fully_connected(net, len(train_y[0]),activation='softmax')
#regression layer that does the gradient descent (i think)
net = tflearn.regression(net)

#RUN MODEL#

#define model and set up tensorboard
model = tflearn.DNN(net, tensorboard_dir = 'tflearn_logs')
#start training
model.fit(train_x, train_y, n_epoch = 1000, batch_size=10, show_metric = True)
#save model once training is complete
model.save('model.tflearn')

#TESTING#

#testing questions
sent1 = "How much money is spent on cycling"
sent2 = "How many accidents where there last year"
sent3 = "cycle city grant is how much?"
sent4 = "drug driving is illegal"
sent5 = "penalty points for mobile phone"
sent6 = "active travel cycling and walking"

#convert testing questions into bag of words numpy array
def get_tf_record(sentence):
    global words
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    BoW = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w==s:
                BoW[i]=1
    return(np.array(BoW))

#print results of testing
print(categories[np.argmax(model.predict([get_tf_record(sent1)]))])
print(categories[np.argmax(model.predict([get_tf_record(sent2)]))])
print(categories[np.argmax(model.predict([get_tf_record(sent3)]))])
print(categories[np.argmax(model.predict([get_tf_record(sent4)]))])
print(categories[np.argmax(model.predict([get_tf_record(sent5)]))])
print(categories[np.argmax(model.predict([get_tf_record(sent6)]))])
