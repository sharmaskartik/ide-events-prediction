import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
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

    inputs = X[:,:-1]
    labels = X[:,-1].type(torch.LongTensor)
    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)
    totalCorrect = (predicted == labels).sum()
    totalSamples = labels.size()[0]

    accuracy = 100 * totalCorrect / totalSamples
    return accuracy

def main():

    #files = [r"topics_50.pickle", r"topics_100.pickle", r"topics_150.pickle", r"topics_200.pickle"]
    files = [r"topics_50.pickle", r"topics_100.pickle", r"topics_150.pickle", r"topics_200.pickle"]
    for file in files:

        window_sizes = [1 , 3, 5]

        for window_size in window_sizes:
            with open(r"topics_50.pickle", "rb") as input_file:
                topics = pickle.load(input_file)

            topics = topics.flatten()
            rolled_data = np.zeros([1, window_size])
            start = 0
            while start + window_size <= topics.shape[0]:
                rolled_data = np.vstack((rolled_data, topics[start: start + window_size]))
                start  += window_size
            X = rolled_data[1:, :] # remove the first rows of zeros

            T = X[1:,0].reshape(-1,1)
            X = X[:-1,:]

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
            trainloader = torch.utils.data.DataLoader(Xtrain, batch_size=batch_size, shuffle=True, num_workers=1)


            Xtest = np.hstack((Xtest, Ttest))
            Xtest = torch.from_numpy(Xtest).type(torch.ls
            cuda.FloatTensor)
            testloader = torch.utils.data.DataLoader(Xtest, batch_size=batch_size, shuffle=True, num_workers=1)


            net = NNClassifier(window_size, num_topics)

            num_epochs = 5
            learning_rate = 0.001


            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

            accuracyTrace =[]

            #TRAINING THE NETWORK
            for epoch in range(2000):  # loop over the dataset multiple times

                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    # get the inputs
                    inputs =  data[:,:-1]
                    labels = data[:,-1].type(torch.cuda.LongTensor)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = net(inputs).squeeze()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                if epoch % 20 == 19:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0



                    with torch.no_grad():
                        #for data in testloader:
                        trainAccuracy = getAccuracy(net, Xtrain)
                        testAccuracy = getAccuracy(net, Xtest)
                        accuracyTrace.append([trainAccuracy,testAccuracy])

            accuracyTrace = np.array(accuracyTrace).reshape(-1, 2)

            with open(filename+".accuracy"+"."+str(window_size), "wb") as output_file:
                pickle.dump(topics, output_file)
if __name__ == "__main__": main()
