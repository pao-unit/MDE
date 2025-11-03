#! /usr/bin/env python3

# Python distribution modules
import sys
# Community modules
# Local modules
from dimx import MDE
from dimx.CLI_Parser import ParseCmdLine

#----------------------------------------------------------------------------
def ManifoldDimExpand():
    '''CLI wrapper for ManifoldDimExpand.'''
    args = ParseCmdLine( argv = sys.argv[1:] )
    mde  = MDE( args = args )
    mde.Run()
    mde.Output()

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading
if __name__ == "__main__":
    ManifoldDimExpand()
