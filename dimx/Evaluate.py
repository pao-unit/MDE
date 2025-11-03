#! /usr/bin/env python3

# Python distribution modules
import argparse
from   os        import environ # Wayland Gnome warning
from   sys       import argv
from   datetime  import datetime
from   itertools import chain

# Community modules
from sklearn.decomposition import PCA
from sklearn.linear_model  import LinearRegression
#from sklearn.metrics       import root_mean_squared_error
from pydiffmap             import diffusion_map as dmap
#from pydiffmap.visualization import embedding_plot, data_plot
from pyEDM                 import Simplex, ComputeError

from numpy  import abs, arange, corrcoef, full, nanmin, nanmax, load, sum
from pandas import read_csv, DataFrame, read_feather

import matplotlib.pyplot as plt

# Silence: Warning: Ignoring XDG_SESSION_TYPE=wayland on Gnome
environ["XDG_SESSION_TYPE"] = "xcb"

#-----------------------------------------------------------------------
class Evaluate:
    '''Compute MDE, Diffusion Map, PCA decompositions for n = args.components
       Training   set: args.library of data columns. 
       Prediction set: args.prediction

       MDE uses multivariate Simplex projection of --mde_columns
       PCA & Diffusion Map project (transform) the modes onto the
       args.prediction test set columns.
       The projected (transformed) modes are used to predict args.predictVar
       over the test set span args.prediction by linear least squares.

       dataFrame is entire data set passed in or read from .csv .npy .feather
       data      is the subset without target to be decomposed by PCA, DMap

       Arguments : These select data columns for PCA & DMap processing
       -cr --columns_range : (start,stop) indices for data columns
       -i  --i_columns     : list of dataFile column indices for data block
       -c  --columnMatch   : list of columns (partial matching) for data block

       -r --predictVar  : target variable
       -m --mde_columns : MDE columns

       -E --E           : MDE E if adding time delays
       -t --tau         : MDE tau if adding time delays

       -n --components  : number of PCA, Diffusion Map components

       -mm --minMax     : apply min/max scaler to data and predict plot

       -rho --plotRho   : Label data / pred panel with R instead of CAE

       Test on Fly FWD
       ./Evaluate.py -d ../../CausalCompression/data/Fly80XY_norm_1061.csv
       -cr 1 80 -r FWD -n 5 -m TS33 TS4 TS8 TS9 TS32 -eps 0.05 -k 40 -P
    '''

    # Import class methods

    #-------------------------------------------------------------------
    def __init__( self, dataFrame = None, dataFile = None, outFile = None,
                  mde_columns = [],  columns_range = [], i_columns = [],
                  columnMatch = [], removeColumns = [], removeTime = False,
                  initDataColumns = [], predictVar = None, library = [],
                  prediction = [], E = 0, tau = -1, Tp = 0, components = 3,
                  dmap_k = 5, dmap_epsilon = 'bgh', dmap_alpha = 0.5,
                  plot = False, plotRho = False, minMax = False,
                  maxN = 7, figsize = (8,8), xlim = None, verbose = False,
                  args = None ):

        '''Constructor
           If dataFrame is None call ReadData()
        '''
        if args is None :
            args = ParseCmdLine( argv = [] ) # get default args
            # Set argument values into args
            args.dataFile        = dataFile
            args.outFile         = outFile
            args.mde_columns     = mde_columns
            args.columns_range   = columns_range
            args.i_columns       = i_columns
            args.columnMatch     = columnMatch
            args.removeColumns   = removeColumns
            args.removeTime      = removeTime
            args.initDataColumns = initDataColumns
            args.predictVar      = predictVar
            args.library         = library
            args.prediction      = prediction
            args.E               = E
            args.tau             = tau
            args.Tp              = Tp
            args.components      = components
            args.dmap_k          = dmap_k
            args.dmap_epsilon    = dmap_epsilon
            args.dmap_alpha      = dmap_alpha
            args.plot            = plot
            args.plotRho         = plotRho
            args.minMax          = minMax
            args.maxN            = maxN
            args.figsize         = figsize
            args.xlim            = xlim
            args.verbose         = verbose

        # Class members
        self.args            = args
        self.embedded        = True
        self.dataFrame       = dataFrame  # entire dataFrame
        self.data            = None       # data for PCA, DMap
        self.data_lib        = None
        self.data_pred       = None
        self.predictVar_lib  = None
        self.predictVar_pred = None

        self.mde             = None
        self.mdeCAE          = None
        self.mdeRMSE         = None
        self.mdeCorrCoeff    = None
        self.mdeRsqr         = None

        self.pca             = None
        self.pcaLinPred      = None
        self.pca_lib         = None
        self.pca_pred        = None
        self.pcaCAE          = None
        self.pcaRMSE         = None
        self.pcaCorrCoeff    = None
        self.pcaRsqr         = None

        self.dmap            = None
        self.dmapLinPred     = None
        self.dmap_lib        = None
        self.dmap_pred       = None
        self.dmapCAE         = None
        self.dmapRMSE        = None
        self.dmapCorrCoeff   = None
        self.dmapRsqr        = None

        self.startTime       = None
        self.elapsedTime     = None
        self.scaler          = None

        if dataFrame is None and dataFile is None :
            msg = f'Evaluate() dataFrame or dataFile required.'
            raise RuntimeError( msg )

        # Initialization
        if self.dataFrame is None :
            self.ReadData() # Load dataFile into self.dataFrame

        # If E > 0 set embedded = False : Adds E time delays
        if E > 0:
            self.embedded = False
            print( 'MDE.__init__() : E > 0 set embedded = False' )

        self.FilterData() # Subset dataFrame into data for PCA, DMap

        # Assign scaler function for data & prediction plot
        if args.minMax :
            def scalerFunc( arg ) :
                min_, max_ = nanmin( arg ), nanmax( arg )
                if ( max_ - min_ ) == 0 :
                    max_, min_ = 1, 0
                    print( 'WARN: minMax scaler divide by zero' )
                scaled = ( arg - min_ ) / ( max_ - min_ )
                return scaled
        else :
            def scalerFunc( arg ) :
                return arg

        self.scaler = scalerFunc

        self.Validate()

    #-------------------------------------------------------------------
    def Validate( self ):
        args = self.args

        if args.predictVar is None :
            msg = 'Validate(): predictVar (-r) required'
            raise RuntimeError( msg )

        if len( args.mde_columns ) < 1 :
            msg = 'Validate(): mde_columns (-m) required'
            raise RuntimeError( msg )

        if args.components is None :
            msg = 'Validate(): components (-n) required'
            raise RuntimeError( msg )

        # If lib & pred not specified, set to all rows
        if len( args.library ) == 0 :
            args.library = [ 1, self.dataFrame.shape[0] ]
            msg = f'Run() set empty lib to {args.library}'
            print( msg )

        if len( args.prediction ) == 0 :
            args.prediction = [ 1, self.dataFrame.shape[0] ]
            msg = f'Run() set empty pred to {args.prediction}'
            print( msg )

    #-------------------------------------------------------------------
    def Run( self ):
        args = self.args
        df   = self.dataFrame
        data = self.data

        self.startTime = datetime.now()
        if args.verbose :
            msg = '\nMDE Evaluate >------------------------\n' +\
                f'  {self.startTime}\n--------------------------------------\n'
            print( msg )

        # Subset data into training (library) and test (prediction) sets
        lib_i  = [ x-1 for x in range(args.library[0],    args.library[1] + 1) ]
        pred_i = [ x-1 for x in range(args.prediction[0], args.prediction[1]+1)]

        self.data_lib  = data.iloc[ lib_i,  : ]
        self.data_pred = data.iloc[ pred_i, : ]

        # Variable to predict : Presumed not in args.i_columns
        predictVar           = df[ args.predictVar ].values
        self.predictVar_lib  = predictVar[ lib_i  ]
        self.predictVar_pred = predictVar[ pred_i ]

        if args.verbose:
            print( "Run(): data", df.shape, "data_lib", self.data_lib.shape,
                   "data_pred", self.data_pred.shape )
            print( "predictVar", len(self.predictVar_lib),
                   len(self.predictVar_pred) )

        # MDE --------------------------------------------------------------
        self.mde = Simplex( dataFrame = df,
                            target = args.predictVar,
                            columns = args.mde_columns,
                            lib = args.library, pred = args.prediction,
                            E = args.E, tau = args.tau, Tp = args.Tp,
                            embedded = self.embedded, showPlot = False )

        self.mdeCAE = round( CAE( self.mde['Observations'],
                                  self.mde['Predictions'] ), 2 )
        mdeErr = ComputeError( self.mde['Observations'],
                               self.mde['Predictions'] )
        self.mdeRMSE      = round( mdeErr['RMSE'],  3 )
        self.mdeCorrCoeff = round( mdeErr['rho'],   3 )
        self.mdeRsqr      = round( self.mdeCorrCoeff**2, 3 )

        print( '\tMDE:  CAE', self.mdeCAE, ' RMSE', self.mdeRMSE,
               ' R', self.mdeCorrCoeff, ' R^2',  self.mdeRsqr )

        # PCA -------------------------------------------------------------
        # pca.components_ :
        #     Principal axes in feature space, representing the
        #     directions of maximum variance in the data.
        # transform(X)
        #     Apply dimensionality reduction to X.
        self.pca = PCA( n_components = args.components )
        self.pca.fit( self.data_lib ) # Fit data_lib to the PCA model

        # Project the pca.fit modes onto the data library and predictions
        self.pca_lib  = self.pca.transform( self.data_lib  )
        self.pca_pred = self.pca.transform( self.data_pred )

        # Linear least squares regression to predict predictVar over the
        # args.prediction set using the pca.fit library projection
        pcaLinMod = LinearRegression().fit( self.pca_lib, self.predictVar_lib )
        self.pcaLinPred = pcaLinMod.predict( self.pca_pred )

        self.pcaCAE = round( CAE( self.predictVar_pred, self.pcaLinPred ), 2 )
        pcaErr      = ComputeError( self.predictVar_pred, self.pcaLinPred )
        self.pcaRMSE      = round( pcaErr['RMSE'],  3 )
        self.pcaCorrCoeff = round( pcaErr['rho'],   3 )
        self.pcaRsqr      = round( self.pcaCorrCoeff**2, 3 )

        print( '\tPCA:  CAE', self.pcaCAE, ' RMSE', self.pcaRMSE,
               ' R', self.pcaCorrCoeff, ' R^2', self.pcaRsqr )

        if args.verbose :
            print( '\tPCA: Explained_variance ratio' )
            print( '\t', self.pca.explained_variance_ratio_ )

        # Diffusion Map ------------------------------------------------------
        # Initialize Diffusion map object.
        neighbor_params = {'n_jobs': -1, 'algorithm': 'kd_tree'}
        self.dmap = dmap.DiffusionMap.from_sklearn(
            n_evecs = args.components, k = args.dmap_k,
            epsilon = args.dmap_epsilon, alpha = args.dmap_alpha,
            neighbor_params = neighbor_params )

        # Fit to data and return the diffusion map.
        self.dmap.fit( self.data_lib )
        self.dmap_lib  = self.dmap.transform( self.data_lib )
        self.dmap_pred = self.dmap.transform( self.data_pred )

        # Linear least squares regression to predict predictVar over the
        # args.prediction set using the DMap.fit library projection
        dmapLinModel = LinearRegression().fit( self.dmap_lib,
                                               self.predictVar_lib )
        self.dmapLinPred = dmapLinModel.predict( self.dmap_pred )

        self.dmapCAE = round( CAE( self.predictVar_pred, self.dmapLinPred ), 2 )
        dmapErr      = ComputeError( self.predictVar_pred, self.dmapLinPred )
        self.dmapRMSE      = round( dmapErr['RMSE'], 3 )
        self.dmapCorrCoeff = round( dmapErr['rho'],  3 )
        self.dmapRsqr      = round( self.dmapCorrCoeff**2, 3 )

        print( '\tdMap: CAE', self.dmapCAE, ' RMSE', self.dmapRMSE,
               ' R', self.dmapCorrCoeff, ' R^2', self.dmapRsqr )

        self.elapsedTime = datetime.now() - self.startTime

        if args.verbose :
            msg = '\nMDE Evaluate <------------------------\n' +\
                  f'  ET {self.elapsedTime}' +\
                  '\n--------------------------------------\n'
            print( msg )

    #---------------------------------------------------------------------
    def Plot( self ) :
        args   = self.args
        scaler = self.scaler

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

        x_pred_mde = x_pred = [x for x in range(args.prediction[0],
                                                args.prediction[1] + 1)]
        x_i = arange( len( x_pred ) )

        if args.E > 0 :
            x_i        = arange( len( x_pred ) - (args.E-1) )
            x_pred_mde = x_pred[ (args.E-1): ]

        # Data & Predictions ------
        if args.plotRho :
            dataLabels = {'Type'  : 'R',
                          'MDE'   : f'MDE   {self.mdeCorrCoeff:.2f}',
                          'D-Map' : f'D-Map {self.dmapCorrCoeff:.2f}',
                          'PCA'   : f'PCA    {self.pcaCorrCoeff:.2f}'}
        else : # annotate rho instead of CAE
            dataLabels = {'Type'  : 'CAE',
                          'MDE'   : f'MDE   {self.mdeCAE:.2f}',
                          'D-Map' : f'D-Map {self.dmapCAE:.2f}',
                          'PCA'   : f'PCA    {self.pcaCAE:.2f}'}

        ax = axs[0]
        ax.plot( x_pred, scaler( self.dmapLinPred ),
                 label = dataLabels['D-Map'], lw = lw )
        ax.plot( x_pred, scaler( self.pcaLinPred ),
                 label = dataLabels['PCA'], lw = lw )
        ax.plot( x_pred_mde, scaler( self.mde.loc[x_i,'Predictions'] ),
                 label = dataLabels['MDE'], lw = lw )
        ax.plot( x_pred, scaler( self.predictVar_pred ),
                 label = args.predictVar, color = 'black', lw = lw )
        ax.legend( title = dataLabels['Type'], bbox_to_anchor = (1., 1),
                   loc = 'upper left' )

        # MDE ---------------------
        ax = axs[1]
        for col in args.mde_columns[:maxN_] :
            ax.plot( x_pred, self.data_pred.loc[ :, col ], label = col, lw = lw )
        ax.legend( title = 'MDE', ncol = 1, bbox_to_anchor = (1., 1),
                   loc = 'upper left' )

        # Diffusion Map ------------
        ax = axs[2]
        for col in range( maxN_ ) :
            ax.plot( x_pred, self.dmap_pred[ :, col ], label = col, lw = lw )
        ax.legend( title = 'Diffusion Map', ncol = 1, bbox_to_anchor = (1., 1),
                   loc = 'upper left' )

        # PCA ----------------------
        ax = axs[3]
        for col in range( maxN_ ) : # pca_lib.shape[1] ) :
            ax.plot( x_pred, self.pca_pred[ :, col ], label = col, lw = lw )
        ax.legend( title = 'PCA', ncol = 1, bbox_to_anchor = (1., 1),
                   loc = 'upper left' )

        if args.xlim is not None :
            plt.xlim( args.xlim )

        plt.tight_layout()
        # Remove vertical space between Axes : After tight_layout() !!!
        fig.subplots_adjust(hspace = 0.04)
        plt.show()

    #--------------------------------------------------------------
    def ReadData( self ) :
        '''Read data from .npy .csv .feather
        If dataFile csv feather : return DataFrame
        If dataFile npy : return DataFrame with columns [c0, c1, c2, ...]
                          First n column names can be specified with
                          self.args.initColumns
        Select columns by columns_range, i_columns or columnMatch
        if args.removeTime : drop first column from DataFrame
        '''
        args = self.args # Shorthand

        if args.verbose :
            msg = f'ReadData(): Reading {args.dataFile}'
            print( msg )

        df       = None
        dataFile = args.dataFile

        if '.csv' in dataFile[-4:] :
            df = read_csv( dataFile )

        elif '.feather' in dataFile[-8:] :
            df = read_feather( dataFile )

        elif '.npy' in dataFile[-4:] :
            data = load( dataFile )

            # Create vector of columns names c0, c1...
            colNames = [ f'c{col}' for col in range( data.shape[1] ) ]

            # if there are non-cell initial columns (Time, Epoch, lswim, rswim)
            # colNames will have too many entries. Insert the specified ones and
            # remove superflous ones
            if len( args.initDataColumns ) :
                args.initDataColumns.reverse()
                for initCol in args.initDataColumns :
                    colNames.insert( 0, initCol )
                    removed = colNames.pop() # remove last item

            df = DataFrame( data, columns = colNames )

        else:
            msg = f'\nReadData(): unrecognized file format: {args.dataFile}'
            raise RuntimeError( msg )

        if args.verbose :
            msg = f'    complete. Shape:{df.shape}'
            print( msg )

        if args.removeTime :
            df = df.drop( axis = 1, index = 0 )

        self.dataFrame = df

    #--------------------------------------------------------------
    def FilterData( self ) :
        '''Filter/subset columns for PCA, dMap data'''

        args = self.args # Shorthand
        df   = self.dataFrame

        if args.columns_range : # 0-offset column indices
            col_i   = arange( args.columns_range[0], args.columns_range[1] )
            columns = df.columns[ col_i ].to_list()

        elif args.i_columns :   # 0-offset column indices
            columns = df.columns[ args.i_columns ].to_list()

        elif len( args.columnMatch ) :
            # Filter df.columns if args.columnMatch specified
            # Any partial match of args.column in df.columnMatch will be included
            colD = {}
            for column in args.columnMatch :
                colD[ column ] = \
                    [ col for col in df.columns if column in col ]

            columns = list( chain.from_iterable( colD.values() ) )

            msg = f'FilterData(): columns matched to {len(columns)} columns.'
            print( msg )

        else :
            columns = df.columns.to_list() # All columns

        # In case predictVar was filtered out, replace it
        #if not args.predictVar in columns :
        #    columns.append( args.predictVar )

        self.data = df[ columns ] # select columns

        if args.verbose :
            if len( columns ) < 101 :
                columns_ = columns
            else :
                columns_ = columns[:5] + columns[-5:]
            print( "FilterData(): DataFrame shape:", self.dataFrame.shape,
                   "\n data shape:", self.data.shape,
                   "\ncolumns:", columns_, "\ntarget:", args.predictVar )

#----------------------------------------------------------------------------
def CAE( x, y ):
    '''Cumulative absolute error.'''
    delta = sum( abs( x - y ) )
    return delta

#----------------------------------------------------------------------------
def ParseCmdLine( argv = None ):

    parser = argparse.ArgumentParser( description = 'MDE Evaluation' )

    parser.add_argument('-d', '--dataFile',
                        dest    = 'dataFile',
                        type    = str,
                        action  = 'store',
                        default = None,
                        help    = 'Data file (csv or npy)')

    parser.add_argument('-o', '--outFile',
                        dest   = 'outFile', type = str,
                        action = 'store', default = None,
                        help = 'Output file')

    # mde_columns are the columns used to predict predictVar with Simplex/MDE
    parser.add_argument('-m', '--mde_columns',
                        dest   = 'mde_columns', nargs = '+', type = str, 
                        action = 'store', default = None,
                        help = 'MDE data column labels')

    # These columns arguments select columns from dataFile for data DF
    parser.add_argument('-cr', '--columns_range',
                        dest   = 'columns_range', nargs = 2, type = int,
                        action = 'store', default = [],
                        help = 'Data column index range [start,stop] 0-offset')

    parser.add_argument('-i', '--i_columns',
                        dest   = 'i_columns', nargs = '+', type = int,
                        action = 'store', default = [],
                        help = 'Data column indices. 0-offset')

    parser.add_argument('-c', '--columnMatch',
                        dest   = 'columnMatch', nargs = '+', type = str,
                        action = 'store', default = [],
                        help = 'Data column names partial match')

    parser.add_argument('-rc', '--removeColumns', nargs = '*',
                        dest    = 'removeColumns', type = str, 
                        action  = 'store', default = [],
                        help    = 'data columns to remove.')

    parser.add_argument('-rT', '--removeTime',
                        dest    = 'removeTime',
                        action  = 'store_true', default = False,
                        help    = 'removeTime.')

    parser.add_argument('-di', '--initDataColumns', nargs = '*',
                        dest    = 'initDataColumns', type = str, 
                        action  = 'store', default = [],
                        help    = 'Initial .npy column names.')

    parser.add_argument('-r', '--predictVar',
                        dest   = 'predictVar', type = str,
                        action = 'store', default = None,
                        help = 'Variable to predict')

    parser.add_argument('-l', '--library',
                        dest   = 'library', nargs = '+', type = int,
                        action = 'store', default = [],
                        help = 'Data library indices. 1-offset')

    parser.add_argument('-p', '--prediction',
                        dest   = 'prediction', nargs = '+', type = int,
                        action = 'store', default = [],
                        help = 'Data prediction indices. 1-offset')

    parser.add_argument('-E', '--E',
                        dest   = 'E', type = int,
                        action = 'store', default = 0,
                        help = 'E : sets embedded False')

    parser.add_argument('-t', '--tau',
                        dest   = 'tau', type = int,
                        action = 'store', default = -1,
                        help = 'tau')

    parser.add_argument('-T', '--Tp',
                        dest   = 'Tp', type = int,
                        action = 'store', default = 0,
                        help = 'Tp')

    parser.add_argument('-n', '--components',
                        dest   = 'components', type = int,
                        action = 'store', default = None,
                        help = 'components')

    parser.add_argument('-k', '--dmap_k',
                        dest   = 'dmap_k', type = int,
                        action = 'store', default = 5,
                        help = 'Diffusion map kernel knn')

    parser.add_argument('-eps', '--dmap_epsilon',
                        dest   = 'dmap_epsilon', type = str,
                        action = 'store', default = 'bgh',
                        help = 'Diffusion map epsilon. Can be scalar')

    parser.add_argument('-a', '--dmap_alpha',
                        dest   = 'dmap_alpha', type = float,
                        action = 'store', default = 0.5,
                        help = 'Diffusion map alpha (normalization exponent)' )

    parser.add_argument('-rho', '--plotRho',
                        dest   = 'plotRho',
                        action = 'store_true', default = False,
                        help = 'Annotate rho instead of CAE')

    parser.add_argument('-mm', '--minMax',
                        dest   = 'minMax',
                        action = 'store_true', default = False,
                        help = 'Apply min max scaler')

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

    args = parser.parse_args( argv ) # if argv is None : default = sys.argv[1:]

    # diffusion map epsilon can be: (string or scalar, optional)
    # – Method for choosing the epsilon. Currently, the only options
    # are to provide a scalar (epsilon is set to the provided scalar)
    # ‘bgh’ (Berry, Giannakis and Harlim), and ‘bgh_generous’
    # (‘bgh’ method, with answer multiplied by 2.
    if 'bgh' not in args.dmap_epsilon and \
       'bgh_generous' not in args.dmap_epsilon :
        args.dmap_epsilon = float( args.dmap_epsilon )
        print( f'dmap_epsilon set to {args.dmap_epsilon}' )

    # zero offset
    # args.library    = [ x - 1 for x in args.library    ]
    # args.prediction = [ x - 1 for x in args.prediction ]

    return args

#----------------------------------------------------------------------------
def EvaluateCLI():
    '''CLI wrapper.'''
    args = ParseCmdLine()
    Eval = Evaluate( dataFrame       = None,
                     dataFile        = args.dataFile,
                     outFile         = args.outFile,
                     mde_columns     = args.mde_columns,
                     columns_range   = args.columns_range,
                     i_columns       = args.i_columns,
                     columnMatch     = args.columnMatch,
                     removeColumns   = args.removeColumns,
                     removeTime      = args.removeTime,
                     initDataColumns = args.initDataColumns,
                     predictVar      = args.predictVar,
                     library         = args.library,
                     prediction      = args.prediction,
                     E               = args.E,
                     tau             = args.tau,
                     Tp              = args.Tp,
                     components      = args.components,
                     dmap_k          = args.dmap_k,
                     dmap_epsilon    = args.dmap_epsilon,
                     dmap_alpha      = args.dmap_alpha,
                     plot            = args.plot,
                     plotRho         = args.plotRho,
                     minMax          = args.minMax,
                     maxN            = args.maxN,
                     figsize         = args.figsize,
                     xlim            = args.xlim,
                     verbose         = args.verbose,
                     args            = args )
    Eval.Run()
    if args.plot :
        Eval.Plot()

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading
if __name__ == "__main__":
    EvaluateCLI()
