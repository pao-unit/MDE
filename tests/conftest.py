"""
conftest.py — shared resources for MDE pytest suite.

pytest loads this file automatically before any test collection.

Contents:
  - GetMP_ContextName()  multiprocessing context helper
  - ValidData()          load a validation CSV by filename
  - *Args dicts          default keyword arguments for each EDM API function
"""

import os
from importlib import resources
from multiprocessing import get_context, get_start_method

from pandas import read_csv

# ---------------------------------------------------------------------------
# Multiprocessing context helper  (remove when > Python 3.13)
# ---------------------------------------------------------------------------

def GetMP_ContextName():
    '''Until > Python 3.14, disallow "fork" multiprocessing context.'''
    allowedContext = ("forkserver", "spawn")
    current = get_start_method( allow_none = True )
    if current in allowedContext:
        return get_context( current )._name
    for method in allowedContext:
        try:
            return get_context( method )._name
        except ValueError:
            continue

# ---------------------------------------------------------------------------
# Validation file helper
# ---------------------------------------------------------------------------

VALID_DIR = os.path.join( os.path.dirname(os.path.abspath(__file__)),
                          "ValidOutput" )

def ValidData( filename ):
    '''Return DataFrame of validation CSV from the validation/ directory.'''
    return read_csv( os.path.join( VALID_DIR, filename ) )

# ---------------------------------------------------------------------------
# MDE data file helper
# ---------------------------------------------------------------------------

def MDE_FlyData():
    '''Get dimx Fly data'''
    import dimx as dx
    dx_path = dx.__file__
    data_path = dx_path.replace('__init__.py', 'data/Fly80XY_norm_1061.csv')

    return read_csv( data_path )

# ---------------------------------------------------------------------------
# Default argument dictionaries — one per API function.
#
# Every parameter is listed. Parameters not actively tested carry a comment.
# Tests copy the relevant dict and update only the parameters under test
# ---------------------------------------------------------------------------

MDEArgs = dict( dataFile        = None,  # file name for DataFrame
                dataName        = None,  # dataName in npz archive
                removeTime      = False, # remove dataFrame first column
                noTime          = False, # first dataFrame column is data
                columnNames     = [],    # partial match columnNames
                initDataColumns = [],    # .npy .npz : see ReadData()
                removeColumns   = [],    # columns to remove from dataFrame
                D               = 3,     # MDE max dimension
                target          = None,  # target variable to predict
                lib             = [],    # EDM library start,stop 1-offset
                pred            = [],    # EDM prediction start,stop 1-offset
                Tp              = 1,     # prediction interval
                tau             = -1,    # CCM embedding delay
                exclusionRadius = 0,     # exclusion radius: CCM, CrossMap
                sample          = 20,    # CCM random sample
                pLibSizes       = [10, 15, 85, 90], # CCM libSizes percentiles
                noCCM           = False, # Do not validate with CCM
                ccmSlope        = 0.01,  # CCM convergence criteria
                ccmSeed         = None,  # CCM random seed
                E               = 0,     # Static E for all CCM
                crossMapRhoMin  = 0.5,   # threshold for L_rhoD in Run()
                embedDimRhoMin  = 0.5,   # maxRhoEDim threshold in Run()
                maxE            = 15,    # maximum embedding dim for CCM
                firstEMax       = False, # use first local peak for E-dim
                timeDelay       = 0,     # Number of time delays to add
                cores           = 5,     # Number of cores for CrossMapColumns
                mpMethod        = GetMP_ContextName(), # multiprocessing context
                chunksize       = 1,     # multiprocessing chunksize
                outDir          = './',  # use pathlib for windog
                outFile         = None,
                outCSV          = None,
                logFile         = None,
                consoleOut      = True,  # LogMsg() print() to console
                verbose         = False,
                debug           = False,
                plot            = False,
                title           = None,
                args            = None )

EvalArgs = dict( dataFile        = None,
                 outFile         = None,
                 mde_columns     = [],
                 columns_range   = [],
                 i_columns       = [],
                 columnMatch     = [],
                 removeColumns   = [],
                 removeTime      = False,
                 initDataColumns = [],
                 predictVar      = None,
                 library         = [],
                 prediction      = [],
                 E               = 0,
                 tau             = -1,
                 Tp              = 0,
                 components      = 3,
                 dmap_k          = 5,
                 dmap_epsilon    = 'bgh',
                 dmap_alpha      = 0.5,
                 plot            = False,
                 plotRho         = False,
                 minMax          = False,
                 maxN            = 7,
                 figsize         = (8,8),
                 xlim            = None,
                 verbose         = False,
                 args            = None )
