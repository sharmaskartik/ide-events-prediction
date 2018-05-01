import json
import numpy as np
from os import listdir


dirPath = "../dataset/main"
types = []
events = []
for dateDir in listdir(dirPath):
    print("In "+dateDir)
    for userDir in listdir(dirPath+"/"+dateDir):

        numFiles = len(listdir(dirPath+"/"+dateDir+"/"+userDir))
        print("In "+dirPath+"/"+dateDir+"/"+userDir," ----- contains "+str(numFiles)+" number of files")
        for i in range(numFiles):
            data = json.load(open(dirPath+"/"+dateDir+"/"+userDir+"/"+str(i)+".json"))
            try:
                types.index(data['$type'])
            except ValueError:
                types.append(data['$type'])
                events.append(data)

types = np.array(types)
print(events)
