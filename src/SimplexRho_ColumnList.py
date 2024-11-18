#! /usr/bin/env python3

# Distribution modules
import time, argparse
from   datetime        import datetime
from   multiprocessing import Pool
from   itertools       import repeat

# Community modules
from pandas import DataFrame, read_csv, concat
from pyEDM  import Simplex, ComputeError

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def SimplexRho_ColumnList( data, columns = [], target = None, E = 0,
                           Tp = 1, tau = -1, exclusionRadius = 0,
                           lib = None, pred = None, embedded = False,
                           cores = 5, outputFile = None, noTime = False,
                           verbose = False, debug = False, LogMsg = None ):

    '''Use multiprocessing Pool to process parallelize Simplex. 
       columns is a list of columns to be cross mapped against
       the target (-t). columns can be a list of single columns,
       or list of multiple columns. 
       Return dict of 'columns:target' : (rho, column) pairs.
    '''

    if verbose and not LogMsg is None :
        LogMsg( f'\tSimplexRho_ColumnList() start {datetime.now()}' )

    startTime = time.time()

    if not len( columns ) :
        raise( RuntimeError( 'columns list required' ) )

    if not target :
        raise( RuntimeError( 'target required' ) )

    # If no lib and pred, create from full data span
    if not lib :
        lib = [ 1, data.shape[0] ]
    if not pred :
        pred = [ 1, data.shape[0] ]

    # Dictionary of arguments for Pool : SimplexFunc
    argsD = { 'target' : target, 'lib' : lib, 'pred' : pred, 'E' : E,
              'embedded' : embedded, 'exclusionRadius' : exclusionRadius,
              'Tp' : Tp, 'tau' : tau, 'noTime' : noTime }

    # Create iterable for Pool.starmap, use repeated copies of argsD, data
    poolArgs = zip( columns, repeat( argsD ), repeat( data ) )

    # Use pool.starmap to distribute among cores
    with Pool( processes = cores ) as pool :
        CMList = pool.starmap( SimplexFunc, poolArgs )

    # Load CMList results into dictionary
    SimplexRho_D = {}
    for i in range( len( columns ) ) :
        key = ','.join( columns[i] )
        SimplexRho_D[ f'{key}:{target}' ] = CMList[ i ] # ( rho, [cols] )

    if verbose and not LogMsg is None :
        LogMsg( f'\tSimplexRho_ColumnList() finished {datetime.now()}' )

    if debug :
        print( "SimplexRho_ColumnList() SimplexRho_D:" )
        print( SimplexRho_D )

    return SimplexRho_D

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def SimplexFunc( columns, argsD, data ):
    '''Call pyEDM Simplex using the column, args, and data'''

    cols = list( columns )

    df = Simplex( dataFrame       = data,
                  columns         = cols,
                  target          = argsD['target'],
                  lib             = argsD['lib'],
                  pred            = argsD['pred'],
                  E               = argsD['E'],
                  embedded        = argsD['embedded'],
                  exclusionRadius = argsD['exclusionRadius'],
                  Tp              = argsD['Tp'],
                  tau             = argsD['tau'],
                  noTime          = argsD['noTime'] )

    err = ComputeError( df['Observations'], df['Predictions'] )
    rho = err['rho']
    return rho, cols

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def SimplexRho_ColumnList_CmdLine():
    '''Wrapper for SimplexRho_ColumnList with command line parsing'''

    args = ParseCmdLine()

    # Read data
    if args.inputFile:
        dataFrame = read_csv( args.inputFile )
    else:
        raise RuntimeError( "inputFile .csv required" )

    # Call SimplexRho_ColumnList()
    df = SimplexRho_ColumnList( data = dataFrame,
                                columns = args.columns, target = args.target,
                                E = args.E, Tp = args.Tp, tau = args.tau,
                                exclusionRadius = args.exclusionRadius,
                                lib = args.lib, pred = args.pred,
                                cores = args.cores, noTime = args.noTime,
                                outputFile = args.outputFile,
                                verbose = args.verbose )

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def ParseCmdLine():
    
    parser = argparse.ArgumentParser( description = 'CrossMap Column List' )

    parser.add_argument('-o', '--outputFile',
                        dest    = 'outputFile', type = str, 
                        action  = 'store',
                        default = None,
                        help    = 'Output file.')

    parser.add_argument('-i', '--inputFile',
                        dest    = 'inputFile', type = str, 
                        action  = 'store',
                        default = None,
                        help    = 'Input data file.')

    parser.add_argument('-nT', '--noTime',
                        dest    = 'noTime',
                        action  = 'store_true',
                        default = False,
                        help    = 'noTime.')

    parser.add_argument('-E', '--E',
                        dest    = 'E', type = int, 
                        action  = 'store',
                        default = 0,
                        help    = 'E.')

    parser.add_argument('-xr', '--exclusionRadius',
                        dest    = 'exclusionRadius', type = int, 
                        action  = 'store',
                        default = 0,
                        help    = 'Exclusion Radius.')

    parser.add_argument('-T', '--Tp',
                        dest    = 'Tp', type = int, 
                        action  = 'store',
                        default = 1,
                        help    = 'Tp.')

    parser.add_argument('-tau', '--tau',
                        dest    = 'tau', type = int, 
                        action  = 'store',
                        default = -1,
                        help    = 'tau.')

    parser.add_argument('-cols', '--columns', nargs = '*',
                        dest    = 'columns', type = str, 
                        action  = 'store',
                        default = [],
                        help    = 'List of columns.')
    
    parser.add_argument('-t', '--target',
                        dest    = 'target', type = str, 
                        action  = 'store',
                        default = '',
                        help    = 'Data target name.')

    parser.add_argument('-l', '--lib', nargs = '*',
                        dest    = 'lib', type = int, 
                        action  = 'store',
                        default = [],
                        help    = 'library indices.')

    parser.add_argument('-p', '--pred', nargs = '*',
                        dest    = 'pred', type = int, 
                        action  = 'store',
                        default = [],
                        help    = 'prediction indices.')

    parser.add_argument('-C', '--cores',
                        dest    = 'cores', type = int, 
                        action  = 'store',
                        default = 5,
                        help    = 'Multiprocessing cores.')

    parser.add_argument('-v', '--verbose',
                        dest    = 'verbose',
                        action  = 'store_true',
                        default = False,
                        help    = 'verbose.')

    args = parser.parse_args()

    return args

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading
if __name__ == "__main__":
    SimplexRho_ColumnList_CmdLine()
