# Python distribution modules
from os       import mkdir
from os.path  import exists
from datetime import datetime
from pickle   import dump
from warnings import filterwarnings
import gzip

# Community modules
from pandas     import read_csv, read_feather, DataFrame
from numpy      import array, load
from matplotlib import pyplot as plt

# Local modules
from .CLI_Parser import ParseCmdLine

# Ignore DeprecationWarning for multiprocessing start_method fork :
# docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
filterwarnings( "ignore", category = DeprecationWarning )

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
                  dataFile        = None,  # file name for DataFrame
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
                  mpMethod        = None,  # multiprocessing start context
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
                  args            = None ):

        if args is None:
            args = ParseCmdLine( argv = [] ) # get default args
            # Insert constructor arguments into args
            args.dataFile        = dataFile
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
            args.cores           = cores
            args.mpMethod        = mpMethod
            args.chunksize       = chunksize
            args.outDir          = outDir
            args.outFile         = outFile
            args.outCSV          = outCSV
            args.logFile         = logFile
            args.consoleOut      = consoleOut
            args.plot            = plot
            args.verbose         = verbose
            args.debug           = debug
            args.plot            = plot
            args.title           = title

        # Class members
        self.args        = args
        self.dataFrame   = dataFrame
        self.target_i    = None
        self.libSizes    = None
        self.libSizesVec = None
        self.MDErho      = array( [], dtype = float )
        self.MDEcolumns  = []
        self.MDEOut      = None   # DataFrame : { rho, columns }
        self.EDim        = dict() # Map of [column:target] : E
        self.rhoD        = dict() # Map of dimension : [L_rhoD]
        self.startTime   = None
        self.elapsedTime = None

        # These should be options, but hardcoded for now
        self.maxOutFileDFcolumns = 50  # Limit on dataFrame columns Output()
        self.maxRhoDlength       = 500 # Limit on number of rhoD Output()

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

        # Read Data from dataFile
        df = self.ReadData()

        # Filter columns if columnNames specified
        # Any partial match of args.columnNames in columns will be included
        if len( self.args.columnNames ) :
            colD = {}
            for columnName in self.args.columnNames :
                colD[ columnName ] = \
                    [ col for col in columns if columnName in col ]

            columns = list( chain.from_iterable( colD.values() ) )

            # In case the target vector was filtered out, replace it
            if not self.args.target in columns :
                columns.append( self.args.target )

            df = df[ columns ]

            msg = f'LoadData(): columns filtered to {len(columns)} columns.'
            self.LogMsg( msg )

        # Column index of target in data
        self.target_i = df.columns.get_loc( self.args.target )

        self.dataFrame = df

        if self.args.verbose :
            self.LogMsg( f'LoadData(): shape {df.shape}\n' )

    #--------------------------------------------------------------
    def ReadData( self ) :
        '''Read data from .npy .npz .feather or .csv
        If dataFile csv     : return DataFrame
        If dataFile npy npz : return DataFrame with columns [c0, c1, c2, ...]
                              First n column names can be specified with
                              self.args.initColumns
        if dataFile npz     : Select the args.dataName from npz archive
        if args.removeTime  : drop first column from DataFrame
        '''

        if self.args.verbose :
            msg = f'ReadData(): Reading {self.args.dataFile}'
            self.LogMsg( msg )

        df       = None
        dataFile = self.args.dataFile

        if '.csv' in dataFile[-4:] :
            df = read_csv( dataFile )

        elif '.feather' in dataFile[-8:] :
            df = read_feather( dataFile )
            
        elif '.npz' in dataFile[-4:] or '.npy' in dataFile[-4:] :
            if '.npz' in dataFile[-4:] :
                data_npz = load( dataFile )
                try:
                    data = data_npz[ self.args.dataName ]
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
            if len( self.args.initDataColumns ) :
                self.args.initDataColumns.reverse()
                for initCol in self.args.initDataColumns :
                    cells.insert( 0, initCol )
                    removed = cells.pop() # remove last item

            df = DataFrame( data, columns = cells )

        else:
            msg = f'\nReadData(): unrecognized file format: {self.args.dataFile}'
            self.LogMsg( msg )
            raise RuntimeError( msg )

        if self.args.verbose :
            msg = f' complete. Shape:{df.shape}'
            self.LogMsg( msg )

        if self.args.removeTime :
            df = df.drop( columns = df.columns[0] )

        return df

    #----------------------------------------------------------
    def Validate( self ):
        '''Require input data and target.
        If lib & pred not specified, set to all rows.'''
        if self.args.target is None :
            msg = f'Validate() target required.'
            self.LogMsg( msg )
            raise RuntimeError( msg )

        if self.dataFrame is None and self.args.dataFile is None :
            msg = f'Validate() dataFrame or dataFile required.'
            self.LogMsg( msg )
            raise RuntimeError( msg )

        if self.dataFrame is None :
            self.LoadData()

        if len( self.args.lib ) == 0 :
            self.args.lib = [ 1, self.dataFrame.shape[0] ]
            msg = f'Validate() set empty lib to  {self.args.lib}'
            self.LogMsg( msg )

        if len( self.args.pred ) == 0 :
            self.args.pred = [ 1, self.dataFrame.shape[0] ]
            msg = f'Validate() set empty pred to {self.args.pred}'
            self.LogMsg( msg )

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
        if self.args.outCSV :
            outFile = f'{self.args.outDir}/{self.args.outCSV}'
            self.MDEOut.to_csv( outFile, index = False )

        if self.args.outFile :
            resetDataFrame = False
            # If self.dataFrame is Huge, limit to something manageable
            # then reset to empty in case Run() is called after this
            if self.dataFrame.shape[1] > self.maxOutFileDFcolumns :
                self.dataFrame = self.dataFrame.iloc[:,:self.maxOutFileDFcolumns]
                resetDataFrame = True

            # If number of items in rhoD exceed self.maxRhoDlength, limit
            for i in range( 1, len( self.rhoD ) + 1 ):
                if len( self.rhoD[i] ) > self.maxRhoDlength :
                    self.rhoD[i] = self.rhoD[i][:self.maxRhoDlength]

            # .pkl or .pkl.gz supported
            outFile = f'{self.args.outDir}/{self.args.outFile}'

            if '.pkl.gz' in outFile[-7:] :
                with gzip.open( outFile, 'wb' ) as f:
                    dump( self, f )
            else :
                if '.pkl' not in outFile[-4:] :
                    outFile = outFile + '.pkl'
                    msg = f'Output() MDE pickle dump to {outFile}'
                    LogMsg( msg )

                with open( outFile, 'wb' ) as f :
                    dump( self, f )

            if resetDataFrame :
                self.dataFrame = DataFrame() # Klunky, but effective

    #----------------------------------------------------------
    def Plot( self ):
        '''MDE plot'''
        ax = self.MDEOut.plot( 'variables', 'rho', lw = 4,
                               title = self.args.title )
        ax.annotate( '\n'.join( self.MDEcolumns ),
                     xy = (0.65, 0.85), xycoords = 'axes fraction',
                     annotation_clip = False, fontsize = 11,
                     verticalalignment = 'top', wrap = True )
        plt.show()

    #-----------------------------------------------------------
    def LogMsg( self, msg, end = '\n', mode = 'a' ):
        '''Log msg to stdout and logFile'''
        if self.args.consoleOut :
            print( msg, end = end, flush = True )

        if self.args.logFile :
            outFile = f'{self.args.outDir}/{self.args.logFile}'
            with open( outFile, mode ) as f:
                print( msg, end = end, file = f, flush = True )
