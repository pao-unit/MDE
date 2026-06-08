#! /usr/bin/env python3
'''Cross map a list of candidate column combinations to a target.

CrossMapColumns() is the standalone / command-line entry point: it prepares
a numeric-only frame, stands up a (single-use) CrossMapPool, runs one
cross-map sweep, and tears the pool down.  The MDE Run() method does NOT use
this function; it owns a persistent CrossMapPool reused across all dimensions
and calls pool.CrossMap() directly.  Both share the worker contract and data
delivery defined in Parallel.py.

Returns dict of 'columns:target' : (rho, [columns]) pairs.
'''

# Distribution modules
import argparse

# Community modules
from pandas import read_csv

# Local
from .Parallel import CrossMapPool, PrepareNumericFrame

#----------------------------------------------------------------------------
def CrossMapColumns( data, columns = [], target = None, E = 0,
                     Tp = 1, tau = -1, exclusionRadius = 0,
                     lib = None, pred = None, embedded = False,
                     crossMapCores = None, mpMethod = None, sharedMem = 0.1,
                     noTime = False, logPct = 0, kdWorkers = 1,
                     LogMsg = None, verbose = False, debug = False ):
    '''Cross map target (-t) to columns (-c)
       columns is a list of column lists (each a candidate combination).
       A run-scoped pool is created, used once, and closed.
       Return dict of 'columns:target' : (rho, column) pairs.'''

    if not len( columns ) :
        raise RuntimeError( 'columns list required' )
    if not target :
        raise RuntimeError( 'target required' )

    # If no lib or pred, create from full data span
    if not lib :
        lib = [ 1, data.shape[0] ]
    if not pred :
        pred = [ 1, data.shape[0] ]

    # Numeric-only frame; workers force noTime = True against it.
    numericDF, _ = PrepareNumericFrame( data, noTime, logMsg = LogMsg )

    # Dictionary of arguments for CrossMapPool initializer = InitWorker()
    argsD = { 'target' : target, 'lib' : lib, 'pred' : pred, 'E' : E,
              'embedded' : embedded, 'exclusionRadius' : exclusionRadius,
              'Tp' : Tp, 'tau' : tau, 'kdWorkers' : kdWorkers }

    with CrossMapPool( numericDF, argsD,
                       crossMapCores = crossMapCores,
                       mpMethod = mpMethod,
                       sharedMem = sharedMem,
                       maxTasks = len( columns ),
                       logMsg   = LogMsg ) as pool :

        CrossMap_D = pool.CrossMap( columns, dimension = 1,
                                    logPct = logPct, verbose = verbose )

    if debug :
        print( 'CrossMapColumns() CrossMap_D:' )
        print( CrossMap_D )

    return CrossMap_D

#----------------------------------------------------------------------------
def CrossMapColumns_CmdLine():
    '''Wrapper for CrossMapColumns with command line parsing.'''
    args = ParseCmdLine()

    if args.inputFile :
        dataFrame = read_csv( args.inputFile )
    else :
        raise RuntimeError( 'inputFile .csv required' )

    CrossMapColumns( data = dataFrame,
                     columns = [ args.columns ], target = args.target,
                     E = args.E, Tp = args.Tp, tau = args.tau,
                     exclusionRadius = args.exclusionRadius,
                     lib = args.lib, pred = args.pred,
                     crossMapCores = args.crossMapCores,
                     mpMethod = args.mpMethod,
                     sharedMem = args.sharedMem, noTime = args.noTime,
                     logPct = args.logPct,
                     verbose = args.verbose, debug = args.debug )

#----------------------------------------------------------------------------
def ParseCmdLine():
    parser = argparse.ArgumentParser( description = 'CrossMap Column List' )

    parser.add_argument( '-i', '--inputFile', dest = 'inputFile',
                         type = str, default = None,
                         help = 'Input data file.' )
    parser.add_argument( '-nT', '--noTime', dest = 'noTime',
                         action = 'store_true', default = False,
                         help = 'noTime.' )
    parser.add_argument( '-E', '--E', dest = 'E', type = int, default = 0 )
    parser.add_argument( '-xr', '--exclusionRadius', dest = 'exclusionRadius',
                         type = int, default = 0 )
    parser.add_argument( '-T', '--Tp', dest = 'Tp', type = int, default = 1 )
    parser.add_argument( '-tau', '--tau', dest = 'tau', type = int,
                         default = -1 )
    parser.add_argument( '-c', '--columns', nargs = '*', dest = 'columns',
                         type = str, default = [], help = 'List of columns.' )
    parser.add_argument( '-t', '--target', dest = 'target', type = str,
                         default = '', help = 'Data target name.' )
    parser.add_argument( '-l', '--lib', nargs = '*', dest = 'lib',
                         type = int, default = [] )
    parser.add_argument( '-p', '--pred', nargs = '*', dest = 'pred',
                         type = int, default = [] )
    parser.add_argument( '-C', '--crossMapCores', dest = 'crossMapCores',
                         type = int, default = None,
                         help = 'Cross-map sweep core cap; all cores if unset.' )
    parser.add_argument( '-mp', '--mpMethod', dest = 'mpMethod', type = str,
                         default = None, help = 'Start method (never fork).' )
    parser.add_argument( '-sM', '--sharedMem', dest = 'sharedMem',
                         type = float, default = 0.1,
                         help = 'Shared-memory threshold (decimal MB, 1e6 '
                                'bytes); 0 forces initargs.' )
    parser.add_argument( '-lp', '--logPct', dest = 'logPct', type = float,
                         default = 0,
                         help = 'Progress band width (percent); needs verbose.' )
    parser.add_argument( '-v', '--verbose', dest = 'verbose',
                         action = 'store_true', default = False )
    parser.add_argument( '-g', '--debug', dest = 'debug',
                         action = 'store_true', default = False )

    return parser.parse_args()

#----------------------------------------------------------------------------
if __name__ == '__main__' :
    CrossMapColumns_CmdLine()
