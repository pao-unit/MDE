#! /usr/bin/env python3

# Python distribution modules
# Community modules
# Local modules
from MDE        import MDE as MDE
from CLI_Parser import ParseCmdLine

#----------------------------------------------------------------------------
def ManifoldDimExpand():
    '''CLI wrapper for ManifoldDimExpand.'''

    args = ParseCmdLine()
    mde  = MDE( args = args )
    mde.Run()
    mde.Output()

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading
if __name__ == "__main__":
    ManifoldDimExpand()
