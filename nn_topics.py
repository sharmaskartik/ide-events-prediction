import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from random import randint
import sys
import mlutils as ml
import pickle
from NNClassifier import NNClassifier
import matplotlib.pyplot as plt

def getAccuracy(net, X):
    totalSamples = 0
    totalCorrect = 0

    inputs = Variable(X[:,:-1], requires_grad=True)
    labels = Variable(X[:,-1].type(torch.cuda.LongTensor))
    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)
    totalCorrect = (predicted == labels).sum().data
    totalSamples = labels.data.size()[0]

    accuracy = totalCorrect / totalSamples
    return accuracy



def rollingWindows(X, windowSize):
    nSamples, nAttributes = X.shape
    nWindows = nSamples - windowSize + 1
    # Shape of resulting matrix
    newShape = (nWindows, nAttributes * windowSize)
    itemSize = X.itemsize  # number of bytes
    # Number of bytes to increment to starting element in each dimension
    strides = (nAttributes * itemSize, itemSize)
    return np.lib.stride_tricks.as_strided(X, shape=newShape, strides=strides)



def main():

    #files = [r"topics_50.pickle", r"topics_100.pickle", r"topics_150.pickle", r"topics_200.pickle"]
    files = ["50", "100", "150", "200"]
    for file in files:

        window_sizes = [1 , 3, 5]

        for window_size in window_sizes:
            print("Working on: "+file+", at window_size: "+ str(window_size))
            with open(r"topics_"+file+".pickle", "rb") as input_file:
                topics = pickle.load(input_file)

            X = rollingWindows(topics, window_size)

            print(X.shape)

            # MAIN CODE STARTS HERE
            # MAKE SURE THAT T CONTAINS TARGETS AS LABELS BETWEEN 0  AND num_topics-1
            classes = list(np.unique(T))
            temp = np.array([ classes.index(T[i]) for i in range(0, T.shape[0])]).reshape(-1,1)
            num_topics = len(classes)
            T = temp
            classes = np.array(classes)
            batch_size = 10

            Xtrain, Ttrain, Xtest, Ttest = ml.partition(X,T,[0.8, 0.2],shuffle=True, classification=True)

            Xtrain = np.hstack((Xtrain, Ttrain))
            Xtrain = torch.from_numpy(Xtrain).type(torch.cuda.FloatTensor)
            trainloader = torch.utils.data.DataLoader(Xtrain, batch_size=batch_size, shuffle=True, num_workers=0)


            Xtest = np.hstack((Xtest, Ttest))
            Xtest = torch.from_numpy(Xtest).type(torch.cuda.FloatTensor)
            testloader = torch.utils.data.DataLoader(Xtest, batch_size=batch_size, shuffle=True, num_workers=0)


            net = NNClassifier(window_size, num_topics)
            net = net.cuda()
            num_epochs = 5
            learning_rate = 0.001


            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

            accuracyTrace =[]
            logStep = 10
            #TRAINING THE NETWORK
            for epoch in range(2000):  # loop over the dataset multiple times

                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):

                    # get the inputs
                    inputs =  Variable(data[:,:-1], requires_grad=True)
                    labels = Variable(data[:,-1].type(torch.cuda.LongTensor))
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = net(inputs)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.data
                if epoch % logStep == logStep - 1:    # print every 2000 mini-batches


                    #for data in testloader:
                    trainAccuracy = getAccuracy(net, Xtrain)
                    testAccuracy = getAccuracy(net, Xtest)
                    accuracyTrace.append([trainAccuracy,testAccuracy])

                    print('[%d, %5d] loss: %.3f , trainAccuracy: %.3f ,  testAccuracy: %.3f' %
                          (epoch + 1, i + 1, running_loss / logStep, trainAccuracy, testAccuracy))
                    running_loss = 0.0


            accuracyTrace = np.array(accuracyTrace).reshape(-1, 2)

            with open("accuracy_"+file+"_"+str(window_size)+".pickle", "wb") as output_file:
                pickle.dump(accuracyTrace, output_file)
if __name__ == "__main__": main()
