#! /usr/bin/env python3
# Python distribution modules
import sys

# Community modules
from pandas import read_csv, read_feather

# Local modules
from dimx import ReverseMDE
from dimx.ReverseCLI_Parser import ParseReverseCmdLine

#----------------------------------------------------------------------------
def ReverseManifoldDimExpand():
    '''CLI wrapper for ReverseManifoldDimExpand.'''
    args = ParseReverseCmdLine( argv = sys.argv[1:] )

    # -mo is a runtime data path : load it to a DataFrame ( as MDE
    # resolves dataFile / slopeMatrixFile ), leaving the parser to
    # capture only the string.
    MDEOut = None
    if args.MDEOut :
        if '.feather' in args.MDEOut[-8:] :
            MDEOut = read_feather( args.MDEOut )
        else :
            MDEOut = read_csv( args.MDEOut )

    rmde = ReverseMDE( args = args, MDEOut = MDEOut )
    rmde.Run()
    rmde.Output()

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading
if __name__ == "__main__":
    ReverseManifoldDimExpand()
