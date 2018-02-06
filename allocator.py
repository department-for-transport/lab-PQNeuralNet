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
from nltk.corpus import stopwords

#punctuation remover (keys in table point to nothing so it just deletes the punctuation in the sentence)
def remove_punctuation(sentence, tbl):
    return sentence.translate(tbl)


def prepare_input_and_output_data(pq_data):
    """takes json data as per json2.txt, processes and returns a list of
    all the words used in each question and list of categories"""


    #create table (a dictionary) to hold remove_punctuation
    #unicodedata.category for punctuation always starts with a 'P'
    table = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

    #list of categories ( needs to the same as a category list created when the model is trained!!! TODO!)
    categories = sorted(list(pq_data.keys()))
    #list of words that holds all unique stemmed words
    words = []

    for each_category in pq_data.keys():
        for each_sentence in pq_data[each_category]:
            #remove punctuation
            each_sentence = remove_punctuation(each_sentence, table)
            #extract words from sentence and add to list 'words'
            w = nltk.word_tokenize(each_sentence)
            words.extend(w)

    #stem and lower each word
    words = [stemmer.stem(w.lower()) for w in words]
    #remove duplicates (the set does this)
    words = sorted(list(set(words)))

    return words, categories

def rebuild_neural_net(word_list, category_list, model_location):
    """Takes expected input and output data lists plus location of saved model
    returns a complete model for use"""
    #net is expecting a shape the size of the bag of words
    net = tflearn.input_data(shape=[None,len(word_list)])
    #layer?
    net = tflearn.fully_connected(net,8)
    #layer?
    net = tflearn.fully_connected(net,8)
    #output layer
    net = tflearn.fully_connected(net, len(category_list),activation='softmax')
    net = tflearn.regression(net)

    model = tflearn.DNN(net, tensorboard_dir = '../tflearn_logs')
    model.load('{}'.format(model_location))
    return model

def create_bagofwords(sentence, word_list):
    """Takes input string and list of all words and returns as a BoW NP array ready for insertion into the mode"""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    BoW = [0]*len(word_list)
    for s in sentence_words:
        for i, w in enumerate(word_list):
            if w==s:
                BoW[i]=1
    return(np.array(BoW))


if __name__ == '__main__':

    stemmer = LancasterStemmer()

    #load json data into var called json
    with open('json2.txt', 'r') as json_data:
        data = json.load(json_data)


    wordlist, categories = prepare_input_and_output_data(data)

    net = rebuild_neural_net(wordlist, categories,'model/model.tflearn')

    while True:
        sent1 = input("Enter question: ")
        print("{}: {}".format(sent1, categories[np.argmax(net.predict([create_bagofwords(sent1, wordlist)]))]))
