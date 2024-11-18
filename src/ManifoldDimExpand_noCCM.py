#! /usr/bin/env python3

# Distribution modules
import time, argparse
from   datetime import datetime

# Community modules
from pandas import DataFrame, read_csv
from numpy  import array, argmax, load, zeros
from matplotlib import pyplot as plt

# Local module from CausalCompression/src
from SimplexRho_ColumnList import SimplexRho_ColumnList

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def ManifoldDimExpand( df, target = None, removeColumns = [],
                       D = 3, Tp = 1, tau = -1, exclusionRadius = 0,
                       lib = None, pred = None, cores = 5, noTime = False,
                       verbose = False, debug = False, plot = False,
                       title = None ):

    '''1-D Cross map all columns against target, select column with highest
       rho. Use that column and all others as 2-D model of target. Select
       2-D column with highest rho. Continue until D columns processed.
    '''

    if verbose :
        print( f'ManifoldDimExpand() start {datetime.now()}' )

    startTime = time.time()

    if not target :
        raise( RuntimeError( 'target required' ) )

    # If no lib and pred, create from full data span
    if not lib :
        lib = [ 1, df.shape[0] ]
    if not pred :
        pred = [ 1, df.shape[0] ]

    # Set initial set of columns and remove any removeColumns
    # Order doesn't matter, use set for efficiency
    allColumns  = df.columns
    dataColumns = list( set( allColumns ) - set( removeColumns ) )

    # Vector of columns selected based on max rho
    columnsVec = []
    # Vector of rho for each column collection
    rhoVec = zeros( D )

    for d in range( 1, D+1 ) :
        # Additional d-D columns : SimplexRho_ColumnList columns nested list
        # Order doesn't matter, use set for efficiency
        columns = list( set( dataColumns ) - set( columnsVec ) )
        columns = [ [c]+columnsVec for c in columns ]

        if debug:
            print( f'{d}-D Cross Map target {target} columns.. {columns[:5]}' )

        # D is dict of 'columns:target' : (rho, column) pairs.
        rhoD = SimplexRho_ColumnList( df, columns = columns,
                                      target = target, # D = args.D,
                                      Tp = Tp, tau = tau,
                                      exclusionRadius = exclusionRadius,
                                      lib = lib, pred = pred,
                                      embedded = True,
                                      cores = cores,
                                      noTime = noTime,
                                      verbose = verbose )

        evalRho  = [ c[0] for c in rhoD.values() ]
        evalCols = [ c[1] for c in rhoD.values() ]
        max_i    = argmax( array( evalRho ) )
        maxRho   = evalRho [max_i]
        maxCols  = evalCols[max_i]

        # Add the first column to columnsVec
        columnsVec.append( maxCols[0] )

        rhoVec[d-1] = maxRho

        if verbose :
            print( f'{d}-D {columnsVec} rho {maxRho}' )

    DF = DataFrame( { 'variables' : columnsVec, 'rho' : rhoVec } )

    if verbose :
        print( f'ManifoldDimExpand() finished {datetime.now()} ' +\
               f'ET {round(time.time()-startTime,3)}')
        print( DF )

    if plot :
        ax = DF.plot( 'variables', 'rho', lw = 4, title = title )
        ax.annotate( '\n'.join( columnsVec ),
                     xy = (0.65, 0.85), xycoords = 'axes fraction',
                     annotation_clip = False, fontsize = 11,
                     verticalalignment = 'top', wrap = True )
        plt.show()

    return DF

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def ManifoldDimExpand_CmdLine():
    '''Wrapper for ManifoldDimExpand with command line parsing'''

    args = ParseCmdLine()

    if args.verbose :
        print( f'ManifoldDimExpand_CmdLine() read data {datetime.now()}' )

    # Read data
    if args.dataFile:
        if '.csv' in args.dataFile :
            dataFrame = read_csv( args.dataFile )
        elif '.npy' in args.dataFile :
            data = load( args.dataFile )
            # Specific for Zebrafish time_epoch_visu_pulse_rswim_lswim_dFF.npy
            cells    = [ f'c{col}' for col in range( data.shape[1] ) ]
            initCols = ['time','epoch','visu','pulse','rswim','lswim']
            initCols.reverse()
            for initCol in initCols :
                cells.insert( 0, initCol )
                removed = cells.pop()

            dataFrame = DataFrame( data, columns = cells )
    else:
        raise RuntimeError( "dataFile .csv pr .npy required" )

    if args.verbose :
        print( f'DataFrame {dataFrame.shape} {dataFrame.columns[:5].tolist()}' )

    # Call ManifoldDimExpand()
    df = ManifoldDimExpand( df = dataFrame,
                            target = args.target,
                            removeColumns = args.removeColumns, D = args.D, 
                            Tp = args.Tp, tau = args.tau,
                            exclusionRadius = args.exclusionRadius,
                            lib = args.lib, pred = args.pred,
                            cores = args.cores, noTime = args.noTime,
                            verbose = args.verbose, debug = args.debug,
                            plot = args.plot, title = args.title )

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def ParseCmdLine():
    
    parser = argparse.ArgumentParser(
        description = 'Manifold Dimensional Expansion no CCM' )

    parser.add_argument('-d', '--dataFile',
                        dest    = 'dataFile', type = str, 
                        action  = 'store',
                        default = None,
                        help    = 'Input data file.')

    parser.add_argument('-rc', '--removeColumns', nargs = '*',
                        dest    = 'removeColumns', type = str, 
                        action  = 'store',
                        default = [],
                        help    = 'data columns to remove.')

    parser.add_argument('-nT', '--noTime',
                        dest    = 'noTime',
                        action  = 'store_true',
                        default = False,
                        help    = 'noTime.')

    parser.add_argument('-D', '--D',
                        dest    = 'D', type = int, 
                        action  = 'store',
                        default = 3,
                        help    = 'D.')

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

    parser.add_argument('-g', '--debug',
                        dest    = 'debug',
                        action  = 'store_true',
                        default = False,
                        help    = 'debug.')

    parser.add_argument('-P', '--plot',
                        dest    = 'plot',
                        action  = 'store_true',
                        default = False,
                        help    = 'plot.')

    parser.add_argument('-title', '--title',
                        dest    = 'title', type = str, 
                        action  = 'store',
                        default = None,
                        help    = 'Plot title.')

    args = parser.parse_args()

    return args

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading
if __name__ == "__main__":
    ManifoldDimExpand_CmdLine()
