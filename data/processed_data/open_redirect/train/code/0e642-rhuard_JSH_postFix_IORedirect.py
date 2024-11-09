import sys
import os
import contextlib

def IO_redirection(cmd):
    """
    Redirect input/output based characters < > >> in the command.
    < will be redirecting input, > will be output and >> is output append
    """
    infile = None
    inputred = False
    outfile = None
    outred = False
    appfile = None
    appred = False
    #check for input redirection first
    if("<" in cmd):
        <fix/>infile, inputred = SetUpInputRedirect(cmd)</fix>

    if(">" in cmd):
        <fix/>outfile, outred = SetUpOutputRedirect(cmd)</fix>

    if(">>" in cmd):
        <fix/>appfile, appred = SetUpAppendRedirect(cmd)

    return (inputred, outred, appred, infile, outfile, appfile)</fix>

def SetUpInputRedirect(cmd):
    <fix/>return _find_redirect("<", cmd)</fix>

def SetUpOutputRedirect(cmd):
    <fix/>return _find_redirect(">", cmd)</fix>

def SetUpAppendRedirect(cmd):
    <fix/>return _find_redirect(">>", cmd)

def _find_redirect(symbol, cmd):
    i = cmd.index(symbol)
    rfile = cmd[i + 1]
    del cmd[i + 1]
    del cmd[i]
    return rfile, True</fix>
