import sys
#import pandas
import collections
import copy
import os
import math
#import sklearn
import re



# Storing class to store messages
class Storing:
    msg_txt= ""
    count_dic = {}

    #we use spam or ham classes
    init_class= ""
    percp_class = ""

    # Constructor
    def __init__(self,msg_txt,ct,init_class):
        self.msg_txt= msg_txt
        self.count_dic = ct
        self.init_class= init_class

    def MessageText(self):
        return self.text

    def Word_count(self):
        return self.count_dic

    def OriginData(self):
        return self.init_class

    def TrainedData(self):
        return self.percp_class

    def PercpClass(self,prediction):
        self.percp_class =prediction



#we used this function in last hw too
def Counts(words):
    WordCount = collections.Counter(re.findall(r'\w+',words))
    return dict(WordCount)



def Messages(str_dic, directory, init_class):
    for message in os.listdir(directory):
        message_path = os.path.join(directory, message)
        if os.path.isfile(message_path):
            with open(message_path, 'r') as msg_file:
                msg_txt= msg_file.read() #init
                str_dic.update({message_path: Storing(msg_txt,Counts(msg_txt),init_class)})


# Set the stop words
def Stp_wrds(stop_file):
    stop_Wrd_list = []
    with open(stop_file, 'r') as stp_txt:
        stop_Wrd_list = (stp_txt.read().splitlines())
    return stop_Wrd_list


# Remove stop words from data set and store in dictionary
def FilterStopWords(stopl,data):
    filtered=copy.deepcopy(data)
    for i in stopl:
        for j in filtered:
            if i in filtered[j].Word_count():
                del filtered[j].Word_count()[i]
    return filtered


# Extracts the vocabulary of all the messages text  a data set
def extract(dt):
    my_list= []
    for i in dt:
        for j in dt[i].Word_count():
            if j not in my_list:
                my_list.append(j)
    return my_list


#we are going to use perceptron training rule
def PercpWeights(weights,learning_rate,trainset,iterations,classes):
    for i in iterations:
        for t in trainset:
            weightedSum = weights['weight_zero']
            for f in trainset[t].Word_count():
                if f not in weights:
                    weights[f] = 0.0
                weightedSum+=weights[f]*trainset[t].Word_count()[f]
            tempOut=0.0
            if weightedSum > 0:
                tempOut = 1.0
            target = 0.0
            if trainset[t].OriginData() == classes[1]:
                target = 1.0
            for w in trainset[t].Word_count(): #updating all weights
                weights[w] += float(learning_rate) * float((target - tempOut)) * \
                              float(trainset[t].Word_count()[w])


#applying perceptron 
def apply(weights,classes,helper):
    weightedSum=weights['weight_zero']
    for i in helper.Word_count():
        if i not in weights:
            weights[i] = 0.0
        weightedSum += weights[i]*helper.Word_count()[i]
    if weightedSum > 0:
        # return is spam
        return 1
    else:
        # return is ham
        return 0



def main(train_dir, test_dir, iterations, learning_rate):
    trainset = {}
    test_set = {}
    filtered_trainset = {}
    filtered_test_set = {}

    # Stop words to filter out.We use the same file as last homework
    stop_words = Stp_wrds('stopWords.txt')

    #same with last homework
    classes = ["ham", "spam"]

    #we are going to choose a learning rate
    iterations = iterations
    learning_rate = learning_rate

    #same as last home work.
    Messages(trainset, train_dir + "/spam", classes[1])
    Messages(trainset, train_dir + "/ham", classes[0])
    Messages(test_set, test_dir + "/spam", classes[1])
    Messages(test_set, test_dir + "/ham", classes[0])

    #same as last  home work. 
    filtered_trainset = FilterStopWords(stop_words,trainset)
    filtered_test_set = FilterStopWords(stop_words,test_set)

    # Extract training set 
    extract_vocab =extract(trainset)
    filtered_extract_vocab =extract(filtered_trainset)

    # store weights as dictionary. w0 initiall 1.0, others initially 1.0. token : weight value
    weights = {'weight_zero': 1.0}
    filtered_weights = {'weight_zero': 1.0}
    for i in extract_vocab:
        weights[i] = 0.0
    for i in filtered_extract_vocab:
        filtered_weights[i] = 0.0

    #We are going to use training set and the filtered training set
    PercpWeights(weights, learning_rate, trainset,iterations,classes)#iterations number of iterations
    PercpWeights(filtered_weights, learning_rate, filtered_trainset,iterations,classes)

    #with stopwords
    temp_true = 0
    for i in test_set:
        predict= apply(weights,classes,test_set[i])
        if predict== 1:
            test_set[i].PercpClass(classes[1])
            if test_set[i].OriginData() == test_set[i].TrainedData():
                temp_true += 1
        if predict == 0:
            test_set[i].PercpClass(classes[0])
            if test_set[i].OriginData() == test_set[i].TrainedData():
                temp_true += 1

    #Without Stop words
    filt_temp_true = 0
    for i in filtered_test_set:
        predict= apply(filtered_weights, classes, filtered_test_set[i])
        if predict== 1:
            filtered_test_set[i].PercpClass(classes[1])
            if filtered_test_set[i].OriginData() == filtered_test_set[i].TrainedData():
                filt_temp_true+=1
        if predict== 0:
            filtered_test_set[i].PercpClass(classes[0])
            if filtered_test_set[i].OriginData() == filtered_test_set[i].TrainedData():
                filt_temp_true+=1 #increase temp value

    #Output
    print "True predictions without filtering stop words: %d/%d" % (temp_true, len(test_set))
    print "Accuracy: %.4f%%" % (float(temp_true) / float(len(test_set)) * 100.0)
    print "True Predictions with filtering stop words: %d/%d" % (filt_temp_true, len(filtered_test_set))
    print "Filtered accuracy: %.4f%%" % (float(filt_temp_true) / float(len(filtered_test_set)) * 100.0)



main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]) #command call
