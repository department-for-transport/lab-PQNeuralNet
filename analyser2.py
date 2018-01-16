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

#list of categories (annoyingly set when the model is trained..TODO!)
categories = ['Maritime Strategy & Programmes', 'Rail Intercity Express Programme', 'Aviation Strategy and Consumers', 'PO P&C', 'Rail Network Services-North', 'Rail Network Services-Intercity Team', 'Private Offices', 'Rail Operations and Performance', 'Regional Strategies London and South', 'Energy, Technology and International', 'Economics of Regional and Local Transport Division', 'Business Partner MPL', 'Air Accidents Investigation Branch', 'Maritime Security & Resilience', 'Strategy Unit', 'Traffic and Technology', 'Office for Low Emission Vehicles', 'Corporate Governance', 'HR Policy', 'Joint Air Quality Unit PQs', 'Rail Analysis', 'Motoring Agency Sponsorship', 'Rail North and Wales Market', 'Digital Services Project Delivery', 'DfT Legal Advisers', 'Rail South, West and London Market', 'Rail Funding and Fares', 'Government Car Service', 'Rail Projects - Legal', 'High Speed Rail', 'Communications Media Team', 'Shared Services Implementation Programme', 'Rail Network Upgrades', 'Environmental Strategy', 'International Cooperation', 'Maritime International & Trade', 'Centre for Connected and Autonomous Vehicles', 'Regional Transport Strategies', 'Roads Economics', 'Active Accessible Travel', 'Freight, Operator Licensing and Roadworthiness', 'Aviation Policy Delivery', 'Group Human Resources', 'Road Investment Strategy Client', 'Rail Industry Competitiveness', 'Road User Licensing, Insurance and Safety', 'Digital Services Live Services', 'HR Operations', 'Communications Strategy and External Affairs', 'Rail Corporate Services and Portfolio Office', 'International and Regulatory Reform', 'Science and Research', 'Communications Marketing and Content', 'Aviation, Maritime, Environment Statistics', 'Rail Network Rail Sponsorship', 'Rail Major Projects and Growth', 'Rail Passenger Services', 'International Vehicle Standards', 'Corporate Finance Directorate', 'Maritime Infrastructure, People & Services', 'DfTc HR Team', 'Road Investment Strategy Futures', 'Maritime And Coastguard Agency', 'Digital and Open Data', 'Rail Strategy and Freight', 'Rail Investment Strategy', 'Transport Appraisal and Strategic Modelling', 'Group Finance', 'Financial Control and Governance', 'Rail Lead and Intercity Market', 'Rail South, East and London Market', 'EU Exit Team', 'Group Procurement and Estates', 'Infrastructure Skills', 'Low Carbon Fuels', 'Northern Transport Programme and Delivery', 'Rail Crossrail and Associated Services', 'Airport Capacity Policy', 'Rail Markets Strategy', 'Rail Digital Services', 'International Aviation, Safety and Environment', 'Road Safety, Standards and Services', 'Rail Franchising Policy Development', 'Rail Passenger Service Excellence', 'Driver and Vehicle Licensing Agency', 'Aviation Security Policy', 'Better Regulation and EU Transposition Policy', 'Cities, Places and Devolution', 'Rail In-Franchise Change', 'Local Infrastructure', 'Digital Services Information & Security', 'Transport Security Strategy', 'Rail Network Services-London & South East', 'Rail Land Transport Security', 'Rail Crossrail 2', 'Driver and Vehicle Standards Agency PQs', 'Rail Thameslink Programme', 'Statistics, Road and Freight', 'Airport Capacity Delivery', 'Statistics Travel and Safety', 'Strategic Finance and Planning', 'Property', 'Maritime Environment, Technology & Innovation', 'Rail Cross London Network Market', 'Dangerous Goods', 'Rail Strategy Projects', 'Marine Accidents Investigation Branch', 'Rail Transforming Ticketing, Payments and Mobility', 'Rail Network Services-West', 'Buses and Taxis', 'Aviation Security Operations', 'Rail Passenger Services Directorate', 'Environment and International Transport Analysis']

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


net = tflearn.input_data(shape=[None,5494])
#layer?
net = tflearn.fully_connected(net,8)
#layer?
net = tflearn.fully_connected(net,8)
#output layer
net = tflearn.fully_connected(net, 113,activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_dir = 'tflearn_logs/DANWYH')
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
    if sent1 == 'exit':
        break
