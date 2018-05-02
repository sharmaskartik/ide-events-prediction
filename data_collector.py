import json
import numpy as np
from os import listdir


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
    "2": "Clone",
    "3": "Commit",
    "4": "CommitAmend",
    "5": "CommitInitial",
    "6": "Merge",
    "7": "Pull",
    "8": "Rebase",
    "9": "RebaseFinished",
    "10": "Reset"
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

#INFO_EVENT =
#ERROR_EVENT =


# EVENT DICTIONARIES
triggered_by = {
    "0": "Unknown",
    "1": "Click",
    "2": "Shortcut",
    "3": "Typing",
    "4": "Automatic"
}




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
            # try:
            #     types.index(data['$type'])
            # except ValueError:
            #     types.append(data['$type'])
            #     events.append(data)
            if data['$type'] == EDIT_EVENT:
                types.append(data["NumberOfChanges"])

types = np.array(types)
print(np.unique(types, return_counts=True))
