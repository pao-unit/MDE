#! /usr/bin/env python3

# Distribution modules
import time, argparse
from   datetime import datetime

# Community modules
from pandas import DataFrame, read_csv
from numpy  import array, argmax, greater, load, nan_to_num, zeros
from pyEDM  import EmbedDimension, CCM
from scipy.signal import argrelextrema
from matplotlib   import pyplot as plt
from sklearn.linear_model import LinearRegression

# Local module from CausalCompression/src
from SimplexRho_ColumnList import SimplexRho_ColumnList

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def ManifoldDimExpand( df, outFile = None,
                       target = None, removeColumns = [],
                       D = 3, Tp = 1, tau = -1, exclusionRadius = 0,
                       lib = None, pred = None, firstEMax = False,
                       embedDimRhoMin = 0.5,
                       sample = 20, pLibSizes = [10,20,80,90],
                       ccmSlope = 0.002, cores = 5, 
                       noTime = False, verbose = False,
                       debug = False, plot = False, title = None ):

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

    # libSizes from percentiles in pLibSizes
    libSizes = [ int( df.shape[0] * (p/100) ) for p in pLibSizes ]
    # libSizes ndarray for CCM convergence slope esimate
    libSizesVec = array( libSizes, dtype = float ).reshape( -1, 1 )
    # normalize [0,1]
    libSizesVec = libSizesVec / libSizesVec[ -1 ]
    if debug :
        print( f'libSizes {libSizes}  libSizesVec {libSizesVec}')

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
        # Order doesn't matter, use set for efficiency, create nested lists
        columns = list( set( dataColumns ) - set( columnsVec ) )
        columns = [ [c] + columnsVec for c in columns ]

        if debug:
            print( f'\n{d}-D Cross Map target {target} columns:5 {columns[:5]}' )
            print( '---------------------------------------------------------' )

        # rhoD is dict of 'columns:target' : (rho, [columns]) pairs.
        # JP Note E = 0, embedded = True : Change for adding lags
        rhoD = SimplexRho_ColumnList( df,
                                      columns         = columns,
                                      target          = target, # E = 0,
                                      Tp              = Tp,
                                      tau             = tau,
                                      exclusionRadius = exclusionRadius,
                                      lib             = lib,
                                      pred            = pred,
                                      embedded        = True,
                                      cores           = cores,
                                      noTime          = noTime,
                                      verbose         = verbose )

        # Sort rhoD values by decreasing rho
        # L_rhoD is ranked list of tuples : (rho, [columns])
        L_rhoD = sorted( rhoD.values(), key = lambda x:x[0], reverse = True )

        if debug:
            print( f'   L_rhoD[:5] {L_rhoD[:5]}' )
        
        # Validate addition of maxCols with CCM
        # First need an embedding dimension
        maxCol_i  = None
        newColumn = None
        for col_i in range( len( L_rhoD )  ) :

            columns_i   = L_rhoD[col_i][1] # List of columns
            newColumn = columns_i[ 0 ]

            if debug :
                print( f'   col_i {col_i}   L_rhoD[col_i] {columns_i}' )

            # Validate column(s) with CCM, first need E
            EDimDF = EmbedDimension( dataFrame       = df,
                                     columns         = newColumn,
                                     target          = target,
                                     maxE            = 15,
                                     lib             = lib,
                                     pred            = pred,
                                     Tp              = Tp,
                                     tau             = tau,
                                     exclusionRadius = exclusionRadius,
                                     validLib        = [],
                                     noTime          = noTime,
                                     verbose         = verbose,
                                     numProcess      = cores,
                                     showPlot        = False )

            # If firstEMax is True, return the first (lowest E) local maximum.
            # Find max E(rho)
            if firstEMax :
                iMax = argrelextrema( EDimDF['rho'].to_numpy(), greater )[0]

                if len( iMax ) :
                    iMax = iMax[0] # first element of array
                else :
                    iMax = len( EDimDF['E'] ) - 1 # no local maxima, last is max
            else : 
                iMax = EDimDF['rho'].round(4).argmax() # global maximum

            maxRhoEDim = EDimDF['rho'].iloc[ iMax ].round(4)
            maxEDim    = EDimDF['E'].iloc[ iMax ]

            if debug :
                print( f'   EDim {columns_i[ 0 ]} E {maxEDim} rho {maxRhoEDim}' )

            if maxRhoEDim < embedDimRhoMin :
                continue # Keep looking

            # CCM
            ccmDF = CCM( dataFrame        = df,
                         columns          = newColumn,
                         target           = target,
                         libSizes         = libSizes,
                         sample           = sample,
                         E                = maxEDim,
                         Tp               = Tp,
                         tau              = tau,
                         exclusionRadius  = exclusionRadius,
                         seed             = None,
                         noTime           = noTime )

            if debug:
                print( ccmDF )

            ccmVals = ccmDF[ f'{target}:{newColumn}' ].to_numpy()

            # Slope of linear fit to rho(libSizes)
            lm = LinearRegression().fit( libSizesVec, nan_to_num( ccmVals ) )
            slope = lm.coef_[0]
            
            if debug :
                print( f'   {target}:{newColumn} slope {slope}' )

            if slope > ccmSlope :
                maxCol_i = col_i
                break # This vector is good
            
            else :
                maxCol_i = None

        # Add the column to columnsVec
        if maxCol_i is not None :
            columnsVec.append( newColumn )

            rhoVec[d-1] = L_rhoD[ maxCol_i ][ 0 ]

            if verbose :
                print( f'{d}-D {columnsVec} rho {L_rhoD[ maxCol_i ][ 0 ]}' )

        else :
            raise RuntimeError( f'{d}-D CCM failed to find columns.' )
            

    DF = DataFrame( { 'variables' : columnsVec, 'rho' : rhoVec } )

    if verbose :
        print( f'ManifoldDimExpand() finished {datetime.now()} ' +\
               f'ET {round(time.time()-startTime,3)}')
        print( DF )

    if outFile :
        DF.to_csv( outFile, index = False )

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
    df = ManifoldDimExpand( df = dataFrame, outFile = args.outFile,
                            target = args.target,
                            removeColumns = args.removeColumns, D = args.D, 
                            Tp = args.Tp, tau = args.tau,
                            exclusionRadius = args.exclusionRadius,
                            lib = args.lib, pred = args.pred,
                            firstEMax = args.firstEMax,
                            embedDimRhoMin = args.embedDimRhoMin,
                            sample = args.sample, pLibSizes = args.pLibSizes,
                            ccmSlope = args.ccmSlope, cores = args.cores,
                            noTime = args.noTime,
                            verbose = args.verbose, debug = args.debug,
                            plot = args.plot, title = args.title )

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def ParseCmdLine():
    
    parser = argparse.ArgumentParser(
        description = 'Manifold Dimensional Expansion' )

    parser.add_argument('-d', '--dataFile',
                        dest    = 'dataFile', type = str, 
                        action  = 'store',
                        default = None,
                        help    = 'Input data file.')

    parser.add_argument('-o', '--outFile',
                        dest    = 'outFile', type = str, 
                        action  = 'store',
                        default = None,
                        help    = 'Output csv file.')

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

    parser.add_argument('-s', '--sample',
                        dest    = 'sample', type = int, 
                        action  = 'store',
                        default = 20,
                        help    = 'CCM sample.')

    parser.add_argument('-L', '--pLibSizes', nargs = '*',
                        dest    = 'pLibSizes', type = int, 
                        action  = 'store',
                        default = [10,15,85,90],
                        help    = 'CCM pLibSizes.')

    parser.add_argument('-ccs', '--ccmSlope',
                        dest    = 'ccmSlope', type = float, 
                        action  = 'store',
                        default = 0.002,
                        help    = 'CCM slope threshold.')

    parser.add_argument('-emin', '--embedDimRhoMin',
                        dest    = 'embedDimRhoMin', type = float, 
                        action  = 'store',
                        default = 0.5,
                        help    = 'embedDimRhoMin threshold.')

    parser.add_argument('-fE', '--firstEMax',
                        dest    = 'firstEMax',
                        action  = 'store_true',
                        default = False,
                        help    = 'firstEMax.')

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
