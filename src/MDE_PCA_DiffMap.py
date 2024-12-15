#! /usr/bin/env python3

# Python distribution modules
import argparse

# Community modules
from sklearn.decomposition   import PCA
from sklearn.linear_model    import LinearRegression
from sklearn.metrics         import root_mean_squared_error
from pydiffmap               import diffusion_map as dmap
from pydiffmap.visualization import embedding_plot, data_plot
from pyEDM                   import Simplex, ComputeError

from numpy  import abs, arange, corrcoef, full, sum
from pandas import read_csv, DataFrame

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.ticker import NullFormatter

#----------------------------------------------------------------------------
# Main module
#----------------------------------------------------------------------------
def main():
    '''Compute MDE, Diffusion Map, PCA decompositions for n = args.components
       Training   set: args.library of data columns. 
       Prediction set: args.prediction
       Project (transform) the modes onto the args.prediction test set columns.
       Use the projected (transformed) modes to predict args.predictVar
       over the test set span args.prediction by linear least squares.

       Arguments
       -c  --columns       : list of columns in dataFile for data block
       -i  --i_columns     : list of dataFile column indices for data block
       -cr --columns_range : (start,stop) indices for data columns

       -r --predictVar  : target variable
       -m --mde_columns : MDE columns

       -n --components  : number of PCA, Diffusion Map components
    '''

    args = ParseCmdLine()
    df   = read_csv( args.dataFile )

    if args.verbose :
        print( args )

    # Variable to predict : Presumed not in args.i_columns
    predictVar = df[ args.predictVar ].values

    # Get list of column names. If not specified use all except first.
    if args.columns :
        columns = args.columns
    elif args.columns_range : # 0-offset column indices
        col_i   = arange( args.columns_range[0], args.columns_range[1] )
        columns = df.columns[ col_i ].to_list()
    elif args.i_columns :     # 0-offset column indices
        columns = df.columns[ args.i_columns ].to_list()
    else :
        columns = df.columns[1:].to_list() # Skip first column

    data = df[ columns ] # select columns

    if args.verbose :
        print( "Read", args.dataFile, "shape:", data.shape,
               "\ncolumns:", columns, "\ntarget:", args.predictVar )

    # Subset data into training (library) and test (prediction) sets
    lib_i  = [ x-1 for x in range(args.library[0],    args.library[1]    + 1) ]
    pred_i = [ x-1 for x in range(args.prediction[0], args.prediction[1] + 1) ]

    data_lib  = data.iloc[ lib_i,  : ]
    data_pred = data.iloc[ pred_i, : ]

    # Variable to predict : Presumed not in args.i_columns
    predictVar      = df[ args.predictVar ].values
    predictVar_lib  = predictVar[ lib_i  ]
    predictVar_pred = predictVar[ pred_i ]

    if args.verbose:
        print( "data", data.shape, "data_lib", data_lib.shape,
               "data_pred", data_pred.shape )
        print( "predictVar", len( predictVar_lib ), len( predictVar_pred ) )

    # MDE --------------------------------------------------------------
    mde = Simplex( dataFrame = df,
                   target = args.predictVar, columns = args.mde_columns,
                   lib = args.library, pred = args.prediction,
                   Tp = args.Tp, tau = args.tau, embedded = True,
                   showPlot = False )

    mdeErr       = ComputeError( mde['Observations'], mde['Predictions'] )
    mdeCAE       = round( CAE( mde['Observations'], mde['Predictions'] ), 2 )
    mdeRMSE      = round( mdeErr['RMSE'],  3 )
    mdeCorrCoeff = round( mdeErr['rho'],   3 )
    mdeRsqr      = round( mdeCorrCoeff**2, 3 )

    print( '\tMDE:  CAE', mdeCAE, ' RMSE', mdeRMSE,
           ' R', mdeCorrCoeff, ' R^2',  mdeRsqr )

    # PCA -------------------------------------------------------------
    # pca.components_ :
    #     Principal axes in feature space, representing the
    #     directions of maximum variance in the data.
    # transform(X)
    #     Apply dimensionality reduction to X.
    pca = PCA( n_components = args.components )
    pca.fit( data_lib ) # Fit data_lib to the PCA model

    if args.verbose :
        print( '\tPCA: Explained_variance ratio' )
        print( '\t', pca.explained_variance_ratio_ )

    # Project the pca.fit modes onto the data library and predictions
    pca_lib  = pca.transform( data_lib  )
    pca_pred = pca.transform( data_pred )

    # Linear least squares regression to predict predictVar over the
    # args.prediction set using the pca.fit library projection
    pcaLinMod     = LinearRegression().fit( pca_lib, predictVar_lib )
    pcaLinPred    = pcaLinMod.predict( pca_pred )
    pcaLinModRsqr = round( pcaLinMod.score( pca_pred, predictVar_pred ), 3 )
    pcaLinCAE     = round( CAE( predictVar_pred, pcaLinPred ), 2 )
    pcaRMSE       = round(root_mean_squared_error(predictVar_pred,pcaLinPred),3)
    pcaCorrCoeff  = round( corrcoef( predictVar_pred, pcaLinPred,
                                     rowvar = False )[0,1], 3 )

    print( '\tPCA:  CAE', pcaLinCAE, ' RMSE', pcaRMSE,
           ' R', pcaCorrCoeff, ' R^2', pcaLinModRsqr )

    # Diffusion Map ------------------------------------------------------
    # Initialize Diffusion map object.
    DMap = dmap.DiffusionMap.from_sklearn( n_evecs = args.components,
                                           k = args.dmap_k, alpha = 0.5 )

    # Fit to data and return the diffusion map.
    DMap.fit( data_lib )
    dMap_lib  = DMap.transform( data_lib )
    dMap_pred = DMap.transform( data_pred )

    # Linear least squares regression to predict predictVar over the
    # args.prediction set using the DMap.fit library projection
    dmapLinModel   = LinearRegression().fit( dMap_lib, predictVar_lib )
    dmapLinPred    = dmapLinModel.predict( dMap_pred )
    dmapLinModRsqr = round( dmapLinModel.score( dMap_pred, predictVar_pred ), 3 )
    dmapRMSE       = round( root_mean_squared_error(predictVar_pred,
                                                    dmapLinPred), 3 )
    dmapLinCAE     = round( CAE( predictVar_pred, dmapLinPred ), 2 )
    dmapCorrCoeff  = round( corrcoef( predictVar_pred, dmapLinPred,
                                      rowvar = False )[0,1], 3 )

    print( '\tdMap: CAE', dmapLinCAE, ' RMSE', dmapRMSE,
           ' R', dmapCorrCoeff, ' R^2', dmapLinModRsqr )

    #---------------------------------------------------------------------
    if args.plot:
        fig, axs = plt.subplots( 4, 1, sharex = True, figsize = args.figsize )

        title = f'{args.predictVar} : train ' +\
                f'{args.library}  predict {args.prediction}'

        axs[0].set_title( title )
        axs[0].set_ylabel('Predictions',   fontsize = 14)
        axs[1].set_ylabel('MDE',           fontsize = 14)
        axs[2].set_ylabel('Diffusion Map', fontsize = 14)
        axs[3].set_ylabel('PCA',           fontsize = 14)

        lw     = 2.5
        maxN_  = min( args.maxN, args.components )

        x      = arange( 1, data.shape[0] + 1 )
        x_lib  = [x for x in range(args.library[0],    args.library[1]    + 1)]
        x_pred = [x for x in range(args.prediction[0], args.prediction[1] + 1)]

        # Data & Predictions ------
        ax = axs[0]
        ax.plot( x_pred, predictVar_pred,
                 label = args.predictVar, color = 'black', lw = lw )
        ax.plot( x_pred, mde['Predictions'][args.Tp :],
                 label = f'MDE   {mdeCAE}', lw = lw )
        ax.plot( x_pred, dmapLinPred, label = f'D-Map {dmapLinCAE}', lw = lw )
        ax.plot( x_pred, pcaLinPred,  label = f'PCA    {pcaLinCAE}', lw = lw )
        ax.legend( title = 'CAE', bbox_to_anchor = (1., 1),
                   loc = 'upper left' )

        # MDE ---------------------
        ax = axs[1]
        for col in args.mde_columns[:maxN_] :
            ax.plot( x_pred, data_pred.loc[ :, col ], label = col, lw = lw )
        ax.legend( title = 'MDE', ncol = 1, bbox_to_anchor = (1., 1),
                   loc = 'upper left' )

        # Diffusion Map ------------
        ax = axs[2]
        for col in range( maxN_ ) :
            ax.plot( x_pred, dMap_pred[ :, col ], label = col, lw = lw )
        ax.legend( title = 'Diffusion Map', ncol = 1, bbox_to_anchor = (1., 1),
                   loc = 'upper left' )

        # PCA ----------------------
        ax = axs[3]
        for col in range( maxN_ ) : # pca_lib.shape[1] ) :
            ax.plot( x_pred, pca_pred[ :, col ], label = col, lw = lw )
        ax.legend( title = 'PCA', ncol = 1, bbox_to_anchor = (1., 1),
                   loc = 'upper left' )

        if args.xlim is not None :
            plt.xlim( args.xlim )

        plt.tight_layout()
        # Remove vertical space between Axes : After tight_layout() !!!
        fig.subplots_adjust(hspace = 0.04)
        plt.show()

#----------------------------------------------------------------------------
def CAE( x, y ):
    '''Cumulative absolute error.'''
    delta = sum( abs( x - y ) )
    return delta

#----------------------------------------------------------------------------
def ParseCmdLine():
    
    parser = argparse.ArgumentParser( description = 'MDE PCA DiffusionMap' )
    
    parser.add_argument('-d', '--dataFile',
                        dest    = 'dataFile',
                        type    = str,
                        action  = 'store',
                        default = None,
                        help    = 'Data file (csv)')
    
    parser.add_argument('-o', '--outFile',
                        dest   = 'outFile', type = str,
                        action = 'store', default = None,
                        help = 'Output file')

    # These columns arguments select columns from dataFile for data DF
    parser.add_argument('-c', '--columns',
                        dest   = 'columns', nargs = '+', type = str, 
                        action = 'store', default = None,
                        help = 'Data column names')

    parser.add_argument('-i', '--i_columns',
                        dest   = 'i_columns', nargs = '+', type = int, 
                        action = 'store', default = None,
                        help = 'Data column indices. 0-offset')

    parser.add_argument('-cr', '--columns_range',
                        dest   = 'columns_range', nargs = 2, type = int, 
                        action = 'store', default = None,
                        help = 'Data column index range [start,stop] 0-offset')

    # mde_columns are the columns used to predict predictVar with Simplex/MDE
    parser.add_argument('-m', '--mde_columns',
                        dest   = 'mde_columns', nargs = '+', type = str, 
                        action = 'store', default = None,
                        help = 'MDE data column labels')

    parser.add_argument('-r', '--predictVar',
                        dest   = 'predictVar', type = str,
                        action = 'store', default = None,
                        help = 'Variable to predict')
    
    parser.add_argument('-p', '--prediction',
                        dest   = 'prediction', nargs = '+', type = int, 
                        action = 'store', default = [],
                        help = 'Data prediction indices. 1-offset')

    parser.add_argument('-l', '--library',
                        dest   = 'library', nargs = '+', type = int, 
                        action = 'store', default = [],
                        help = 'Data library indices. 1-offset')

    parser.add_argument('-tau', '--tau',
                        dest   = 'tau', type = int, 
                        action = 'store', default = -1,
                        help = 'tau')

    parser.add_argument('-T', '--Tp',
                        dest   = 'Tp', type = int, 
                        action = 'store', default = 0,
                        help = 'Tp')

    parser.add_argument('-n', '--components',
                        dest   = 'components', type = int, 
                        action = 'store', default = 3,
                        help = 'components')

    parser.add_argument('-k', '--dmap_k',
                        dest   = 'dmap_k', type = int, 
                        action = 'store', default = 5,
                        help = 'Diffusion map kernel knn')

    parser.add_argument('-P', '--plot',
                        dest   = 'plot',
                        action = 'store_true', default = False,
                        help = 'Plot')

    parser.add_argument('-N', '--maxN',
                        dest   = 'maxN', type = int, 
                        action = 'store', default = 7,
                        help = 'Maximum number of modes to plot')

    parser.add_argument('-fs', '--figsize', nargs = 2,
                        dest   = 'figsize', type = int,
                        action = 'store', default = (8,8),
                        help = 'figsize')

    parser.add_argument('-xl', '--xlim',
                        dest   = 'xlim', nargs = 2, type = int, 
                        action = 'store', default = None,
                        help = 'Plot xlim [start,stop]')

    parser.add_argument('-v', '--verbose',
                        dest   = 'verbose',
                        action = 'store_true', default = False,
                        help = 'verbose')

    args = parser.parse_args()

    # zero offset
    # args.library    = [ x - 1 for x in args.library    ]
    # args.prediction = [ x - 1 for x in args.prediction ]

    return args

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading
if __name__ == "__main__":
    main()
