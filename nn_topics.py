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


def main():

    if len(sys.argv) < 2 :
        print("usage python nn_topics.py WindowSize")
        exit(0)

    window_size = int(sys.argv[1])

    # creating sample data_
    with open(r"topics_50.pickle", "rb") as input_file:
        topics = pickle.load(input_file)

    rolled_data = np.zeros([1, window_size])
    start = 0
    while start + window_size <= topics.shape[0]:
        rolled_data = np.vstack((rolled_data, topics[start: start + window_size]))
        start  += window_size
    X = rolled_data[1:, :] # remove the first rows of zeros

    T = X[1:,0].reshape(-1,1)
    X = X[:-1,:]

    # n = 500
    # x1 = np.linspace(5,20,n) + np.random.uniform(-2,2,n)
    # y1 = ((20-12.5)**2-(x1-12.5)**2) / (20-12.5)**2 * 10 + 14 + np.random.uniform(-2,2,n)
    # x2 = np.linspace(10,25,n) + np.random.uniform(-2,2,n)
    # y2 = ((x2-17.5)**2) / (25-17.5)**2 * 10 + 5.5 + np.random.uniform(-2,2,n)
    # angles = np.linspace(0,2*np.pi,n)
    # x3 = np.cos(angles) * 15 + 15 + np.random.uniform(-2,2,n)
    # y3 = np.sin(angles) * 15 + 15 + np.random.uniform(-2,2,n)
    # X =  np.vstack((np.hstack((x1,x2,x3)), np.hstack((y1,y2,y3)))).T
    # T = np.repeat(range(0,3),n).reshape((-1,1))

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
    Xtrain = torch.from_numpy(Xtrain).type(torch.FloatTensor)
    trainloader = torch.utils.data.DataLoader(Xtrain, batch_size=batch_size, shuffle=True, num_workers=2)


    Xtest = np.hstack((Xtest, Ttest))
    Xtest = torch.from_numpy(Xtest).type(torch.FloatTensor)
    testloader = torch.utils.data.DataLoader(Xtest, batch_size=batch_size, shuffle=True, num_workers=2)


    net = NNClassifier(window_size, num_topics)

    num_epochs = 5
    learning_rate = 0.001


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)



    #TRAINING THE NETWORK
    for epoch in range(50):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs =  data[:,:-1]
            labels = data[:,-1].type(torch.LongTensor)
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

    print('Finished Training')


    #EVALUATING THE NETWORK
    class_correct = list(0. for i in range(num_topics))
    class_total = list(0. for i in range(num_topics))
    with torch.no_grad():
        for data in testloader:
            inputs =  data[:,:-1]
            labels = data[:,-1].type(torch.LongTensor)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            print("batch size ", labels.size()[0])
            for i in range(labels.size()[0]):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(num_topics):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

if __name__ == "__main__": main()
