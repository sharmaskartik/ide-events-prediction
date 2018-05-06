import json
import numpy as np
from os import listdir
from datetime import datetime
import time
from dateutil import parser
import pickle

ACTIVITY_EVENT = "KaVE.Commons.Model.Events.ActivityEvent, KaVE.Commons"
COMMAND_EVENT = "KaVE.Commons.Model.Events.CommandEvent, KaVE.Commons"
COMPLETION_EVENT = "KaVE.Commons.Model.Events.CompletionEvents.CompletionEvent, KaVE.Commons"
completion_termination_state ={
    "0": "Applied",
    "1": "Cancelled",
    "2": "Filtered",
    "3": "Unknown"
}

BUILD_EVENT = "KaVE.Commons.Model.Events.VisualStudio.BuildEvent, KaVE.Commons"
DEBUGGER_EVENT = "KaVE.Commons.Model.Events.VisualStudio.DebuggerEvent, KaVE.Commons"
debugger_mode = {
    "0": "Design",
    "1": "Run",
    "2": "Break",
    "3": "ExceptionThrown",
    "4": "IgnorExceptionNotHandleded"
}


DOCUMENT_EVENT = "KaVE.Commons.Model.Events.VisualStudio.DocumentEvent, KaVE.Commons"
document_action = {
    "0": "Opened",
    "1": "Saved",
    "2": "Closing"
}


EDIT_EVENT = "KaVE.Commons.Model.Events.VisualStudio.EditEvent, KaVE.Commons"


FIND_EVENT = "KaVE.Commons.Model.Events.VisualStudio.FindEvent, KaVE.Commons"



IDE_STATE_EVENT = "KaVE.Commons.Model.Events.VisualStudio.IDEStateEvent, KaVE.Commons"
ide_life_cycle_phase = {
    "0": "Startup",
    "1": "Shutdown",
    "2": "Runtime"
}

SOLUTION_EVENT = "KaVE.Commons.Model.Events.VisualStudio.SolutionEvent, KaVE.Commons"
solution_action = {
    "0": "OpenSolution",
    "1": "RenameSolution",
    "2": "CloseSolution",
    "3": "AddSolutionItem",
    "4": "RenameSolutionItem",
    "5": "RemoveSolutionItem",
    "6": "AddProject",
    "7": "RenameProject",
    "8": "RemoveProject",
    "9": "AddProjectItem",
    "10": "RenameProjectItem",
    "11": "RemoveProjectItem"
    }



WINDOW_EVENT = "KaVE.Commons.Model.Events.VisualStudio.WindowEvent, KaVE.Commons"
window_action = {
    "0": "Create",
    "1": "Activate",
    "2": "Move",
    "3": "Close",
    "4": "Deactivate"
}


VERSION_CONTROL_EVENT = "KaVE.Commons.Model.Events.VersionControlEvents.VersionControlEvent, KaVE.Commons"
version_control_action_type = {
    "0": "Unknown",
    "1": "Branch",
    "2": "Checkout",
    "3": "Clone",
    "4": "Commit",
    "5": "CommitAmend",
    "6": "CommitInitial",
    "7": "Merge",
    "8": "Pull",
    "9": "Rebase",
    "10": "RebaseFinished",
    "11": "Reset"
}


USER_PROFILE_EVENT = "KaVE.Commons.Model.Events.UserProfiles.UserProfileEvent, KaVE.Commons"
NAVIGATION_EVENT = "KaVE.Commons.Model.Events.NavigationEvent, KaVE.Commons"
navigation_type = {
    "0": "Unknown",
    "1": "CtrlClick",
    "2": "Click",
    "3": "Keyboard"
}

SYSTEM_EVENT = "KaVE.Commons.Model.Events.SystemEvent, KaVE.Commons"
system_event_type = {
    "0": "Unknown",
    "1": "Suspend",
    "2": "Resume",
    "3": "Lock",
    "4": "Unlock",
    "5": "RemoteConnect",
    "6": "RemoteDisconnect"
}

TEST_RUN_EVENT = "KaVE.Commons.Model.Events.TestRunEvents.TestRunEvent, KaVE.Commons"
test_result = {
    "0": "Unknown",
    "1": "Success",
    "2": "Failed",
    "3": "Error",
    "4": "Ignored"
}

INFO_EVENT = ""
ERROR_EVENT = ""


# EVENT DICTIONARIES
triggered_by = {
    "0": "Unknown",
    "1": "Click",
    "2": "Shortcut",
    "3": "Typing",
    "4": "Automatic"
}


dirPath = "../dataset/main"
events = []

for dateDir in listdir(dirPath):
    print("In "+dateDir)
    for userDir in listdir(dirPath+"/"+dateDir):

        numFiles = len(listdir(dirPath+"/"+dateDir+"/"+userDir))
        print("In "+dirPath+"/"+dateDir+"/"+userDir," ----- contains "+str(numFiles)+" number of files")
        for i in range(numFiles):
            data = json.load(open(dirPath+"/"+dateDir+"/"+userDir+"/"+str(i)+".json"))
            event_type = data['$type']

            if "ErrorEvent" in event_type or "InfoEvent" in event_type:
                print(event_type)

            if event_type == INFO_EVENT or event_type == USER_PROFILE_EVENT:
                continue

            word = event_type.split(",")[0].split(".")[-1] + "_"

            if  event_type == COMMAND_EVENT:
                word += str(data['CommandId'].replace(" ","_"))

            elif event_type == IDE_STATE_EVENT:
                word += ide_life_cycle_phase[str(data['IDELifecyclePhase'])]

            elif event_type == WINDOW_EVENT:
                word += window_action[str(data['Action'])]

            elif event_type == DOCUMENT_EVENT:
                word += document_action[str(data['Action'])]

            elif event_type == SOLUTION_EVENT:
                word += solution_action[str(data['Action'])]


            elif event_type == NAVIGATION_EVENT:
                word += navigation_type[str(data['TypeOfNavigation'])]

            elif event_type == COMPLETION_EVENT:
                word += completion_termination_state[str(data['TerminatedState'])]

            elif event_type == BUILD_EVENT:
                word += str(data['Scope']) + "_" + str(data['Action'])

            elif event_type == DEBUGGER_EVENT:
                try:
                    word += debugger_mode[str(data['Mode'])] + "_" + str(data['Reason']) + str(data['Action'])
                except:
                    word += debugger_mode[str(data['Mode'])] + "_" + str(data['Reason'])
            elif event_type == FIND_EVENT:
                word += "Cancelled_" + str(data['Cancelled'])

            elif event_type == SYSTEM_EVENT:
                word += system_event_type[str(data['Type'])]

            elif event_type == TEST_RUN_EVENT:
                tests = data["Tests"]
                allPass = True
                for test in tests:
                    if test['Result'] == 0:
                        allPass = False
                        break
                if allPass:
                    result = "Pass"
                else:
                    result = "Fail"
                word += "Aborted_" + str(data['WasAborted']) + "_" + result

            elif event_type == VERSION_CONTROL_EVENT:
                actions = data["Actions"]
                for action in actions:
                    vcWord  = word + version_control_action_type[str(action['ActionType'])]
                    time = action["ExecutedAt"]
                    events.append([vcWord, time])
                continue


            #word += "_" + triggered_by[str(data["TriggeredBy"])]

            time = data["TriggeredAt"]

            events.append([word, time])

events = np.array(events)

## WE HAVE THE DATA HERE IN ONE ARRAY
## NOW WE HAVE TO CREATE SESSIONS

print(events.shape)

sessions = []
tempSession = []
lastTime = int(parser.parse(events[0][1]).strftime("%s"))

for event in events:

    currentTime = int(parser.parse(event[1]).strftime("%s"))
    if(currentTime - lastTime > 300):
        sessions.append(tempSession)
        tempSession = []
        #print("difference greater than 5 minutes")
    tempSession.append(event[0])

    lastTime = currentTime


# NOW CREATE WINDOWS FROM SESSIONS
windowSizes = [50, 100, 150, 200]

for windowSize in windowSizes:
    windows = np.zeros([1, windowSize])
    for session in sessions:
        session = np.array(session)

        start = 0
        while start + windowSize <= session.shape[0]:
            windows = np.vstack((windows, session[start: start + windowSize]))
            start  += windowSize
    windows = windows[1:, :] # remove the first rows of zeros
    with open(r"data_"+str(windowSize)+".pickle", "wb") as output_file:
        pickle.dump(windows, output_file)
    print(windows.shape)
