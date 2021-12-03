# -*- coding: UTF-8 -*-

import socket
import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
import json
import numpy
import random
import tflearn
from tensorflow.python.framework import ops
import pickle

# load all data in json file

with open('intents.json') as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:  # read by bytes
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])  # for training the model

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]
    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        # generate output
        output_row = out_empty[:]  # make copy
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)
    with open("data.pickle", "wb") as f:  # write by bytes
        pickle.dump((words, labels, training, output), f)

# use tflearn
ops.reset_default_graph()

# tensorflow.reset_default_gragh()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)  # add 8 fully connected neural with 2 hidden layers
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")  # softmax can trasfer outcome to probability
net = tflearn.regression(net)

model = tflearn.DNN(net)


try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
'''
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")
'''
def bag_of_word(s, words):
    bags = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bags[i] = 1

    return numpy.array(bags)



server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


server.bind(('localhost', 6688))

server.listen(5)
#print(server.getsockname())
print(u'waiting for connect...')

connect, (host, port) = server.accept()

peer_name = connect.getpeername()
sock_name = connect.getsockname()
print(u'the client %s:%s has connected.' % (host, port))
#print('The peer name is %s and sock name is %s' % (peer_name, sock_name))

welcome_message = 'This is a chat bot! Chat with me or type \'quit\' at any time! \n'
welcome_message = welcome_message.encode('utf-8')
connect.sendall(welcome_message)

greeting = "Hi how are you?\n"
greeting = greeting.encode('utf-8')
connect.sendall(greeting)


while True:
    inp = connect.recv(1024)
    inp = inp.decode('utf-8')
    if inp.lower() == "quit":
            break
    
    result = model.predict([bag_of_word(inp, words)])
    result_index = numpy.argmax(result)  # give the max index
    tag = labels[result_index]
    for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
    send = random.choice(responses)
    send = send.encode('utf-8')
    connect.sendall(send)
    


server.close()