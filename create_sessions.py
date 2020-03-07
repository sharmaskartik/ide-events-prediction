import pickle
import parser
import numpy as np
import os

targetPath = "/s/neptune/a/nobackup/kartikay/extracted_events_data"
with open(os.path.join(targetPath, 'extracted_events.pickle'), 'rb') as f:
    events = pickle.load(f)


sessions = []
current_session = []

last_time = int(parser.parse(events[0][2]).strftime("%s"))
last_user = events[0][1]

import pdb; pdb.set_trace()
#get unique users
#extra data for each user
#sort the data on time
#use parts of code from below to create sessions
for i, event in enumerate(events):

    current_time = int(parser.parse(event[2]).strftime("%s"))
    current_user = events[0][1]

    if(current_time - last_time > 300) or current_user != last_user:
        sessions.append(current_session)
        current_session = []
        #print("difference greater than 5 minutes")
    current_session.append(event[0])

    last_time = current_time
    last_user = current_user



# NOW CREATE WINDOWS FROM SESSIONS
#windowSizes = [50, 100, 150, 200]
windowSizes = [50]

for windowSize in windowSizes:
    windows = np.zeros([1, windowSize])
    for session in sessions:
        session = np.array(session)

        start = 0
        while start + windowSize <= session.shape[0]:
            windows = np.vstack((windows, session[start: start + windowSize]))
            start  += windowSize
    windows = windows[1:, :] # remove the first rows of zeros
    with open(targetPath+r"data_"+str(windowSize)+".pickle", "wb") as output_file:
        pickle.dump(windows, output_file)
    print(windows.shape)
