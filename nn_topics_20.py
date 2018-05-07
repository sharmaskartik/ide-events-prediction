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
from NNFeedforward import NNFeedForward
from NNConvolution import NNModelConv
import matplotlib.pyplot as plt

def getAccuracy(net, X, xCols, tCols, type):
    totalSamples = 0
    totalCorrect = 0

    inputs =  Variable(X[:,0: xCols], requires_grad=True)
    if type == 1:
        rows, cols = inputs.size()
        inputs = inputs.contiguous().view(rows, 1, cols)
    labels = Variable(X[:,xCols: xCols+tCols])
    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)
    _, labels = torch.max(labels, 1)

    totalCorrect = (predicted == labels).sum().data
    totalSamples = labels.data.size()[0]

    accuracy = float(totalCorrect) / float(totalSamples)
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

    if(len(sys.argv) != 3):
        print("Usage: python nn_topics <file> <mode> (1: Convolution Network 2: FeedForward)")
        exit(0)
    file = sys.argv[1]
    type = int(sys.argv[2])

    #files = [r"topics_50.pickle", r"topics_100.pickle", r"topics_150.pickle", r"topics_200.pickle"]
    #dataDir = "../dataset/topics/"
    dataDir = "../dataset/topics_20/"
    #targetDir = "../dataset/accuracy/"
    targetDir = "../dataset/accuracy_20/"
    files = ["50", "100", "150", "200"]

    if type == 1:
        print("Running in Convolution Mode")
        window_sizes = [10]
        netClass = NNModelConv
    else:
        print("Running in FeedForward Mode")
        window_sizes = [1 , 3, 5]
        netClass = NNFeedForward

    for window_size in window_sizes:
        print("Working on: "+file+", at window_size: "+ str(window_size))
        with open(dataDir+r"topics_"+file+".pickle", "rb") as input_file:
            topics = pickle.load(input_file)

        X = rollingWindows(topics, window_size)
        tCols = topics.shape[1]
        xCols = X.shape[1]

        T = X[1:,0:tCols]
        X = X[1:,:]

        print(X.shape, T.shape)
        # MAIN CODE STARTS HERE

#            continue
        batch_size = 50

        Xtrain, Ttrain, Xtest, Ttest = ml.partition(X,T,[0.8, 0.2],shuffle=True)

        Xtrain = np.hstack((Xtrain, Ttrain))
        Xtrain = torch.from_numpy(Xtrain).type(torch.cuda.FloatTensor)
        trainloader = torch.utils.data.DataLoader(Xtrain, batch_size=batch_size, shuffle=True, num_workers=0)


        Xtest = np.hstack((Xtest, Ttest))
        Xtest = torch.from_numpy(Xtest).type(torch.cuda.FloatTensor)
        testloader = torch.utils.data.DataLoader(Xtest, batch_size=batch_size, shuffle=True, num_workers=0)


        net = netClass(xCols, tCols)
        net = net.cuda()
        num_epochs = 2000
        learning_rate = 0.001

        criterion = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

        accuracyTrace =[]
        logStep = 100
        #TRAINING THE NETWORK
        print("before training")
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            print("Epoch:", epoch)
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):

                # get the inputs
                inputs =  Variable(data[:,0: xCols], requires_grad=True)
                if type == 1:
                    rows, cols = inputs.size()
                    inputs = inputs.contiguous().view((rows, 1, cols))
                labels = Variable(data[:,xCols: xCols+tCols])
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.data
                if i % logStep == logStep - 1:    # print every logStep mini-batches

                    #for data in testloader:
                    trainAccuracy = getAccuracy(net, Xtrain, xCols, tCols, type)
                    testAccuracy = getAccuracy(net, Xtest, xCols, tCols, type)
                    accuracyTrace.append([trainAccuracy,testAccuracy])

                    print('[%d, %d] loss: %.3f , trainAccuracy: %.3f ,  testAccuracy: %.3f' %
                          (epoch + 1, i + 1,running_loss / 100, trainAccuracy, testAccuracy))
                    running_loss = 0.0


        accuracyTrace = np.array(accuracyTrace).reshape(-1, 2)

        with open(targetDir+"type_"+str(type)+"_accuracy_"+file+"_"+str(window_size)+".pickle", "wb") as output_file:
            pickle.dump(accuracyTrace, output_file)
if __name__ == "__main__": main()
