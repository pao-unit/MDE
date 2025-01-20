# Python distribution modules
from os       import mkdir
from os.path  import exists
from datetime import datetime
from pickle   import dump
from math     import nan
import warnings

# Community modules
from pandas     import read_csv, read_feather, DataFrame
from numpy      import array, load
from matplotlib import pyplot as plt

# Local modules
from CLI_Parser import ParseCmdLine

# Ignore DeprecationWarning for multiprocessing start_method fork :
# docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
warnings.filterwarnings( "ignore", category = DeprecationWarning )

#-----------------------------------------------------------------------
class MDE:
    '''Class for Manifold Dimensional Expansion
       ManifoldDimExpand.py is a CLI to instantiate, configure and Run().

       Uses args object from CLI_Parser.ParseCmdLine to store class
       arguments/parameters.
    '''

    # Import class methods
    from Run import Run

    #-------------------------------------------------------------------
    def __init__( self, dataFrame = None, dataFile = None,
                  dataName = None, removeTime = False, noTime = False,
                  columnNames = [], initDataColumns = [], removeColumns = [],
                  D = 3, target = None, lib = [], pred = [],
                  Tp = 1, tau = -1, exclusionRadius = 0,
                  sample = 20, pLibSizes = [10, 15, 85, 90],
                  noCCM = False, ccmSlope = 0.01,
                  E = 0, embedDimRhoMin = 0.5, firstEMax = False,
                  outDir = None, outFile = None, logFile = None,
                  cores = 5, consoleOut = True,
                  verbose = False, debug = False,
                  plot = False, title = None, args = None ):

        '''Constructor

           If dataFrame is None LoadData() / ReadData() called at end
           of constructor.
        '''
        
        if args is None:
            args = ParseCmdLine() # set default args
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
            args.E               = E
            args.embedDimRhoMin  = embedDimRhoMin
            args.firstEMax       = firstEMax
            args.outDir          = outDir
            args.outFile         = outFile
            args.logFile         = logFile
            args.cores           = cores
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
        self.startTime   = None
        self.elapsedTime = None

        if self.dataFrame is None and self.args.dataFile is None :
            msg = f'MDE() dataFrame or dataFile required.'
            self.LogMsg( msg )
            raise RuntimeError( msg )

        if self.args.target is None :
            msg = f'MDE() target required.'
            self.LogMsg( msg )
            raise RuntimeError( msg )

        # Initialization
        self.CreateOutDir()

        if self.args.verbose :
            msg = f'\nManifold Dimensional Expansion >------\n' +\
                f'  {datetime.now()}\n--------------------------------------\n'
            self.LogMsg( msg )

        if self.dataFrame is None :
            self.LoadData()

        self.Validate()

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
        '''Read data from .npy .npz or .csv
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
        '''If lib & pred not specified, set to all rows'''
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
        '''MDE output: pickle dump of the MDE class object'''
        if self.args.outFile :
            #self.MDEOut.to_csv( self.args.outFile, index = False )

            with open( self.args.outFile, 'wb' ) as f :
                dump( self, f )

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

    #--------------------------------------------------------------------
    def LogMsg( self, msg, end = '\n', mode = 'a' ):
        '''Log msg to stdout and logFile'''
        if self.args.consoleOut :
            print( msg, end = end, flush = True )

        if self.args.logFile :
            outFile = f'{self.args.outDir}/{self.args.logFile}'
            with open( outFile, mode ) as f:
                print( msg, end = end, file = f, flush = True )
