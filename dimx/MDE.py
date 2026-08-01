# Python distribution modules
from os       import mkdir
from os.path  import exists
from datetime import datetime
from pickle   import dump
from warnings import filterwarnings
from copy     import deepcopy
import gzip

# Community modules
from pandas     import read_csv, read_feather, DataFrame
from numpy      import array, load
from matplotlib import pyplot as plt

# Local modules
from .CLI_Parser import ParseCmdLine

# Ignore RuntimeWarning : Likely in pyEDM ComputeError 
#   lib/python3.13/site-packages/numpy/lib/_function_base_impl.py:3000:
#   RuntimeWarning: invalid value encountered in divide  c /= stddev[None, :]
filterwarnings( "ignore", category = RuntimeWarning )

#-----------------------------------------------------------------------
class MDE:
    '''Class for Manifold Dimensional Expansion
       ManifoldDimExpand.py is a CLI to instantiate, configure and Run().

       Uses Namespace object (args) from CLI_Parser.ParseCmdLine to store
       class arguments/parameters.
    '''

    # Import class methods
    from .Run import Run

    #-------------------------------------------------------------------
    def __init__( self,
                  dataFrame       = None,  # pandas DataFrame
                  slopeMatrix     = None,  # precomputed CCM slope DataFrame
                  dataFile        = None,  # file name for DataFrame
                  slopeMatrixFile = None,  # CCM slope matrix .csv / .feather
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
                  libSizes        = [],    # CCM libSizes
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
                  crossMapCores   = None,  # cross-map core cap; None=all cores
                  mpMethod        = None,  # multiprocessing start context
                  chunksize       = 1,     # multiprocessing chunksize
                  sharedMem       = 0.1,   # shared-mem threshold (decimal MB)
                  logPct          = 0,     # cross-map progress band
                  kdWorkers       = 1,     # KDTree.query workers in Simplex
                  maxLenRhoD      = None,  # Output() cap on rhoD per dim
                  maxLenRhoD_CCM  = None,  # Output() cap on rhoD_CCM per dim
                  outDir          = './',  # use pathlib for windog
                  outFile         = None,  # MDE object dumped to .pkl or .pkl.gz
                  outCSV          = None,  # MDEOut
                  logFile         = None,
                  consoleOut      = True,  # LogMsg() print() to console
                  verbose         = False,
                  debug           = False,
                  plot            = False,
                  args            = None ):

        if args is None:
            args = ParseCmdLine( argv = [] ) # get default args
            # Insert constructor arguments into args
            args.dataFile        = dataFile
            args.slopeMatrixFile = slopeMatrixFile
            args.dataName        = dataName
            args.removeTime      = removeTime
            args.noTime          = noTime
            args.columnNames     = columnNames
            args.initDataColumns = initDataColumns
            args.removeColumns   = removeColumns
            args.D               = D
            args.target          = target
            args.lib             = lib
            args.pred            = pred
            args.Tp              = Tp
            args.tau             = tau
            args.exclusionRadius = exclusionRadius
            args.sample          = sample
            args.libSizes        = libSizes
            args.pLibSizes       = pLibSizes
            args.noCCM           = noCCM
            args.ccmSlope        = ccmSlope
            args.ccmSeed         = ccmSeed
            args.E               = E
            args.crossMapRhoMin  = crossMapRhoMin
            args.embedDimRhoMin  = embedDimRhoMin
            args.maxE            = maxE
            args.firstEMax       = firstEMax
            args.timeDelay       = timeDelay
            args.crossMapCores   = crossMapCores
            args.mpMethod        = mpMethod
            args.chunksize       = chunksize
            args.sharedMem       = sharedMem
            args.logPct          = logPct
            args.kdWorkers       = kdWorkers
            args.maxLenRhoD      = maxLenRhoD
            args.maxLenRhoD_CCM  = maxLenRhoD_CCM
            args.outDir          = outDir
            args.outFile         = outFile
            args.outCSV          = outCSV
            args.logFile         = logFile
            args.consoleOut      = consoleOut
            args.plot            = plot
            args.verbose         = verbose
            args.debug           = debug
            args.plot            = plot

        # Class members
        self.args            = args
        self.dataFrame       = dataFrame
        self.slopeMatrix     = slopeMatrix
        self.target_i        = None
        self.libSizes        = libSizes
        self.libSizesVec     = None
        self.MDErho          = array( [], dtype = float )
        self.MDEcolumns      = []
        self.MDEOut          = None   # DataFrame : { rho, columns }
        self.EDim            = dict() # Map of [column:target] : E (accepted)
        self.rhoD            = dict() # Map of dimension : [L_rhoD]
        self.rhoD_CCM        = dict() # subset of L_rhoD passing CCM : slopeMatrix
        self.maxLenRhoD      = maxLenRhoD     # outFile len limit on rhoD
        self.maxLenRhoD_CCM  = maxLenRhoD_CCM # outFile len limit on rhoD_CCM
        self._edimCache      = dict() # column : (maxEDim, maxRhoEDim) compute cache
        self._ccmCache       = dict() # column : slope compute cache
        self.startTime       = None
        self.elapsedTime     = None

        # Initialization
        self.CreateOutDir()

        if self.args.verbose :
            msg = f'\nManifold Dimensional Expansion ' +\
                f'>------\n  {datetime.now()}' +\
                '\n--------------------------------------------\n'
            self.LogMsg( msg )

    #-------------------------------------------------------------------
    def LoadData( self ):
        '''Wrapper for ReadData() that reads .csv .npy .npz .feather
           Optionally filter columns with partial match to args.columnNames
        '''

        args = self.args

        # Read Data from dataFile
        df = self.ReadData()

        # Filter columns if columnNames specified
        # Any partial match of args.columnNames in columns will be included
        if len( args.columnNames ) :
            colD = {}
            for columnName in args.columnNames :
                colD[ columnName ] = \
                    [ col for col in columns if columnName in col ]

            columns = list( chain.from_iterable( colD.values() ) )

            # In case the target vector was filtered out, replace it
            if not args.target in columns :
                columns.append( args.target )

            df = df[ columns ]

            msg = f'LoadData(): columns filtered to {len(columns)} columns.'
            self.LogMsg( msg )

        # Column index of target in data
        self.target_i = df.columns.get_loc( args.target )

        self.dataFrame = df

        if args.verbose :
            self.LogMsg( f'LoadData(): shape {df.shape}\n' )

    #--------------------------------------------------------------
    def ReadData( self ) :
        '''Read data from .npy .npz .feather or .csv
        If dataFile csv     : return DataFrame
        If dataFile npy npz : return DataFrame with columns [c0, c1, c2, ...]
                              First n column names can be specified with
                              self.args.initColumns
        if dataFile npz     : Select the args.dataName from npz archive
        if args.removeTime  : drop first column from DataFrame copy
        '''
        args = self.args

        if args.verbose :
            msg = f'ReadData(): Reading {args.dataFile}'
            self.LogMsg( msg )

        df       = None
        dataFile = args.dataFile

        if '.csv' in dataFile[-4:] :
            df = read_csv( dataFile )

        elif '.feather' in dataFile[-8:] :
            df = read_feather( dataFile )

        elif '.npz' in dataFile[-4:] or '.npy' in dataFile[-4:] :
            if '.npz' in dataFile[-4:] :
                data_npz = load( dataFile )
                try:
                    data = data_npz[ args.dataName ]
                except KeyError as kerr:
                    msg = f'\nReadData(): Error: .npz keys: {data_npz.files}\n'
                    self.LogMsg( msg )
                    raise KeyError( kerr )
            else :
                data = load( dataFile )

            # Create vector of columns names c0, c1...
            cells = [ f'c{col}' for col in range( data.shape[1] ) ]

            # if there are non-cell initial columns (Time, Epoch, lswim, rswim)
            # cells will have too many entries. Insert the specified ones and
            # remove superflous ones
            if len( args.initDataColumns ) :
                args.initDataColumns.reverse()
                for initCol in args.initDataColumns :
                    cells.insert( 0, initCol )
                    removed = cells.pop() # remove last item

            df = DataFrame( data, columns = cells )

        else:
            msg = f'\nReadData(): unrecognized file format: {args.dataFile}'
            self.LogMsg( msg )
            raise RuntimeError( msg )

        if args.verbose :
            msg = f' complete. Shape:{df.shape}'
            self.LogMsg( msg )

        return df

    #----------------------------------------------------------
    def LoadSlopeMatrix( self ):
        '''Resolve the optional CCM slope matrix onto self.args.slopeMatrix.

        Precedence:
          noCCM True             : ignored (logged), set to None.
          slopeMatrix provided   : used as-is (API path, DataFrame).
          slopeMatrixFile set    : read .csv or .feather into a DataFrame.
          neither                : None.

        Convention: the matrix is square with identical labels on .index
        (source / embedded dimension) and .columns (predicted dimension);
        the Run() lookup is slopeMatrix.loc[target, column]. CCM slope is
        directional, so the matrix is not symmetric.

        .csv     : written without an index (pure float matrix). The first
                   column must be float - a written index column would parse
                   as object - and the row labels are reconstructed from the
                   columns.
        .feather : read_feather preserves the written index; .index and
                   .columns must already match.
        '''
        args = self.args

        slopeMatrix     = self.slopeMatrix
        slopeMatrixFile = args.slopeMatrixFile

        # noCCM disables CCM qualification entirely; any matrix is ignored.
        if args.noCCM :
            if slopeMatrix is not None or slopeMatrixFile :
                self.LogMsg( 'LoadSlopeMatrix(): noCCM = True. '
                             'slope matrix ignored.' )
            return

        # API path: slopeMatrix DataFrame supplied directly, no file read.
        if slopeMatrix is not None :
            return

        # No matrix and no file: nothing to do.
        if not slopeMatrixFile :
            return

        if args.verbose :
            self.LogMsg( f'LoadSlopeMatrix(): Reading {slopeMatrixFile}' )

        if '.csv' in slopeMatrixFile[-4:] :
            df = read_csv( slopeMatrixFile )

            # A written index column would parse as object, not float.
            if df.iloc[ :, 0 ].dtype.kind != 'f' :
                msg = ( 'LoadSlopeMatrix(): .csv first column is not float; '
                        'expected an index-less float matrix (got dtype '
                        f'{df.iloc[ :, 0 ].dtype}).' )
                self.LogMsg( msg )
                raise RuntimeError( msg )

            if df.shape[0] != df.shape[1] :
                msg = ( 'LoadSlopeMatrix(): .csv matrix is not square '
                        f'{df.shape}; cannot map index to columns.' )
                self.LogMsg( msg )
                raise RuntimeError( msg )

            # Reconstruct row labels from column labels (identical convention).
            df.index = df.columns

            if args.verbose :
                self.LogMsg( 'LoadSlopeMatrix(): .csv float matrix verified; '
                             f'index reconstructed from {df.shape[1]} columns.' )

        elif '.feather' in slopeMatrixFile[-8:] :
            df = read_feather( slopeMatrixFile )

            # read_feather preserves the written index; require it to match.
            if not df.index.equals( df.columns ) :
                msg = ( 'LoadSlopeMatrix(): .feather .index does not match '
                        '.columns; slope matrix index was not preserved.' )
                self.LogMsg( msg )
                raise RuntimeError( msg )

            if args.verbose :
                self.LogMsg( 'LoadSlopeMatrix(): .feather index verified to '
                             f'match {df.shape[1]} columns.' )

        else :
            msg = ( 'LoadSlopeMatrix(): unrecognized slope matrix format: '
                    f'{slopeMatrixFile}' )
            self.LogMsg( msg )
            raise RuntimeError( msg )

        if args.verbose :
            self.LogMsg( f'LoadSlopeMatrix(): complete. Shape: {df.shape}' )

        self.slopeMatrix = df

    #----------------------------------------------------------
    def Validate( self ):
        '''Require input data and target.
        If lib & pred not specified, set to [1,N/2], [N/2+1,N]'''
        args = self.args

        if args.target is None :
            msg = f'Validate() target required.'
            self.LogMsg( msg )
            raise RuntimeError( msg )

        if self.dataFrame is None and args.dataFile is None :
            msg = f'Validate() dataFrame or dataFile required.'
            self.LogMsg( msg )
            raise RuntimeError( msg )

        if self.dataFrame is None :
            self.LoadData()

        if args.removeTime :
            self.dataFrame = \
                self.dataFrame.copy().drop(columns = self.dataFrame.columns[0])

        if not isinstance( args.removeColumns, list ) :
            msg = f'Validate() removeColumns must be list.'
            self.LogMsg( msg )
            raise RuntimeError( msg )

        if not isinstance( args.columnNames, list ) :
            msg = f'Validate() columnNames must be list.'
            self.LogMsg( msg )
            raise RuntimeError( msg )

        if not isinstance( args.initDataColumns, list ) :
            msg = f'Validate() initDataColumns must be list.'
            self.LogMsg( msg )
            raise RuntimeError( msg )

        if not isinstance( args.lib, list ) :
            msg = f'Validate() lib must be list.'
            self.LogMsg( msg )
            raise RuntimeError( msg )

        if not isinstance( args.pred, list ) :
            msg = f'Validate() pred must be list.'
            self.LogMsg( msg )
            raise RuntimeError( msg )

        if len( args.lib ) == 0 :
            args.lib = [ 1, int( self.dataFrame.shape[0]/2 ) ]
            msg = f'Validate() set empty lib to {args.lib}'
            self.LogMsg( msg )

        if len( args.pred ) == 0 :
            args.pred = [ int( self.dataFrame.shape[0]/2 ) + 1,
                               self.dataFrame.shape[0] ]
            msg = f'Validate() set empty pred to {args.pred}'
            self.LogMsg( msg )

        # Resolve optional CCM slope matrix (file -> DataFrame, or pass-through)
        self.LoadSlopeMatrix() # -> self.slopeMatrix

        if self.slopeMatrix is None :
            if len( self.libSizes ) == 0 :
                # CCM libSizes from percentiles in pLibSizes
                self.libSizes = [ int( self.dataFrame.shape[0] * (p/100) )
                                  for p in args.pLibSizes ]

                if args.verbose :
                    msg = f'Validate(): libSizes from pLibSizes: {self.libSizes}'
                    self.LogMsg( msg )

            if min( self.libSizes ) < 5 :
                msg = 'Validate(): libSizes min must be at least 5'
                self.LogMsg( msg )
                raise RuntimeError( msg )

            if max( self.libSizes ) > len( self.dataFrame ) :
                msg = f'Validate(): libSizes max exceeds {len(self.dataFrame)}'
                self.LogMsg( msg )
                raise RuntimeError( msg )

            if len( self.libSizes ) < 3 :
                msg = 'Validate(): at least 2 libSizes required.'
                self.LogMsg( msg )
                raise RuntimeError( msg )

        else : # self.slopeMatrix is not None
            sM = self.slopeMatrix

            if not isinstance( sM, DataFrame ) :
                msg = 'Validate() slopeMatrix must be a pandas DataFrame.'
                self.LogMsg( msg )
                raise RuntimeError( msg )

            if not sM.index.equals( sM.columns ) :
                msg = ( 'Validate() slopeMatrix .index and .columns must be '
                        'identical (same labels, same order).' )
                self.LogMsg( msg )
                raise RuntimeError( msg )

            if args.target not in sM.columns :
                msg = ( 'Validate() slopeMatrix does not contain target '
                        f'{args.target}.' )
                self.LogMsg( msg )
                raise RuntimeError( msg )

            self.LogMsg( 'Validate(): slope matrix in use; CCM slopes read '
                         'from matrix, EmbedDimension / CCM skipped, '
                         'embedDimRhoMin inactive.' )
            if args.verbose :
                self.LogMsg( f'Validate(): slope matrix {sM.shape[0]}x'
                             f'{sM.shape[1]}, target '
                             f'{args.target} present.' )

    #-----------------------------------------------------------
    def CreateOutDir( self ):
        '''Probe outDir and create if needed'''
        outDir = self.args.outDir
        if not outDir :
            self.args.outDir = outDir = './'

        if not exists( outDir ) :
            try :
                mkdir( outDir )
                msg = 'CreateOutDir() Created directory ' + outDir
                self.LogMsg( msg )

            except FileNotFoundError :
                msg = f'CreateOutDir() Invalid output path {outDir}'
                self.LogMsg( msg )
                raise RuntimeError( msg )

            if not exists( outDir ) :
                msg = f'CreateOutDir() Failed to mkdir {outDir}'
                self.LogMsg( msg )
                raise RuntimeError( msg )

    #----------------------------------------------------------
    def Output( self ):
        '''MDE output:
             MDEOut DataFrame to args.outCSV
             MDE class object to args.outFile as .pkl or .pkl.gz'''
        args = self.args

        if args.outCSV :
            outFile = f'{args.outDir}/{args.outCSV}'
            self.MDEOut.to_csv( outFile, index = False )

        if args.outFile :
            # Do not include self.dataFrame or self.slopeMatrix in the dump
            dataFrame_copy = self.dataFrame.copy()
            self.dataFrame = None

            slopeMatrix_copy = None
            if self.slopeMatrix is not None:
                slopeMatrix_copy = self.slopeMatrix.copy()
                self.slopeMatrix = None

            # If number of items in rhoD exceed maxLenRhoD, limit in dump
            if self.maxLenRhoD is not None:
                rhoD_copy = deepcopy(self.rhoD)
                for i in range( 1, len( self.rhoD ) + 1 ):
                    if len( self.rhoD[i] ) > self.maxLenRhoD :
                        self.rhoD[i] = self.rhoD[i][:self.maxLenRhoD]

            # Likewise for rhoD_CCM. Bounded by its own length: rhoD_CCM may
            # hold one more (terminal, empty) key than rhoD when expansion
            # ends at the crossMapRhoMin gate. Both dicts are contiguous
            # from dimension 1, so range iteration is safe. Truncation keeps
            # the head, i.e. the highest-rho passing entries.
            if self.maxLenRhoD_CCM is not None:
                rhoD_CCM_copy = deepcopy(self.rhoD_CCM)
                for i in range( 1, len( self.rhoD_CCM ) + 1 ):
                    if len( self.rhoD_CCM[i] ) > self.maxLenRhoD_CCM :
                        self.rhoD_CCM[i] = self.rhoD_CCM[i][:self.maxLenRhoD_CCM]

            # .pkl or .pkl.gz supported
            outFile = f'{args.outDir}/{args.outFile}'

            if '.pkl.gz' in outFile[-7:] :
                with gzip.open( outFile, 'wb' ) as f:
                    dump( self, f )
            else :
                if '.pkl' not in outFile[-4:] :
                    outFile = outFile + '.pkl'
                    msg = f'Output() MDE pickle dump to {outFile}'
                    self.LogMsg( msg )

                with open( outFile, 'wb' ) as f :
                    dump( self, f )

            # Reinstate DataFrame & slopeMatrix
            self.dataFrame   = dataFrame_copy
            self.slopeMatrix = slopeMatrix_copy
            # Reinstate rhoD and rhoD_CCM if needed
            if self.maxLenRhoD is not None:
                self.rhoD = rhoD_copy
            if self.maxLenRhoD_CCM is not None:
                self.rhoD_CCM = rhoD_CCM_copy

    #----------------------------------------------------------
    def Plot( self, title = '', table_xy = (0.6, 0.85),
              maxTable = None, fontsize = 12, figsize = (6,5) ):
        '''Plot an MDEOut DataFrame from MDE.Run()'''
        df = self.MDEOut.copy()
        D  = [d+1 for d in range(df.shape[0])]
        df.insert(0,"D",D)

        if maxTable is None:
            maxTable = df.shape[0]

        df_string = df.iloc[:maxTable,:].round(3).to_string(index=False)

        ax = df.plot( 'D', 'rho', lw = 4, title = title, figsize = figsize )
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.xaxis.label.set_size(fontsize)
        ax.yaxis.label.set_size(fontsize)
        ax.set_ylabel('MDE ρ', fontsize = fontsize)
        ax.annotate( df_string, 
                     xy = table_xy, xycoords = 'axes fraction',
                     annotation_clip = False, fontsize = 11,
                     verticalalignment = 'top', wrap = True,
                     fontproperties = 'monospace' )
        plt.show()

    #-----------------------------------------------------------
    def LogMsg( self, msg, end = '\n', mode = 'a' ):
        '''Log msg to stdout and logFile'''
        args = self.args

        if args.consoleOut :
            print( msg, end = end, flush = True )

        if args.logFile :
            outFile = f'{args.outDir}/{args.logFile}'
            with open( outFile, mode ) as f:
                print( msg, end = end, file = f, flush = True )
