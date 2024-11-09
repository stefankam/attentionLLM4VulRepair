import sys
import os
import contextlib

def IO_redirection(cmd):
    """
    Redirect input/output based characters < > >> in the command.
    < will be redirecting input, > will be output and >> is output append
    """
    #check for input redirection first
    if("<" in cmd):
        <vul/>SetUpInputRedirect(cmd)</vul>

    if(">" in cmd):
        <vul/>SetUpOutputRedirect(cmd)</vul>

    if(">>" in cmd):
        <vul/>SetUpAppendRedirect(cmd)</vul>

def SetUpInputRedirect(cmd):
    <vul/><vul/>import pdb; pdb.set_trace()
    pass</vul></vul>

def SetUpOutputRedirect(cmd):
    <vul/>i = cmd.index(">")
    sys.stdout = open(cmd[i + 1], "w")
    del cmd[i + 1]
    del cmd[i]
    print("this is a test")</vul>

def SetUpAppendRedirect(cmd):
    import pdb; pdb.set_trace()
    pass
