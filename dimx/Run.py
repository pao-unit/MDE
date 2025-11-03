# Distribution modules
from datetime import datetime

# Community modules
from pandas import DataFrame, concat
from numpy  import append, array, greater, nan_to_num
from pyEDM  import ComputeError, Embed, EmbedDimension, CCM, Simplex
from scipy.signal         import argrelextrema
from sklearn.linear_model import LinearRegression

# Local module from CausalCompression/src
from .CrossMapColumns import CrossMapColumns

#-------------------------------------------------------------------
#-------------------------------------------------------------------
def Run( self ):
    '''Execute MDE workflow pipeline'''

    self.startTime = datetime.now()

    # Shortcuts in local scope
    a      = self.args
    LogMsg = self.LogMsg

    self.Validate()

    # CCM libSizes from percentiles in pLibSizes
    self.libSizes = [ int( self.dataFrame.shape[0] * (p/100) )
                      for p in a.pLibSizes ]
    # libSizes ndarray for CCM convergence slope esimate
    self.libSizesVec = array( self.libSizes, dtype = float ).reshape(-1, 1)
    # normalize libSizesVec to [0,1] for equitable CCM slope comparison
    self.libSizesVec = self.libSizesVec / self.libSizesVec[ -1 ]

    if a.debug :
        LogMsg( f'libSizes {self.libSizes}  libSizesVec {self.libSizesVec}')

    # Set initial dataColumns as allColumns minus removeColumns
    # Order doesn't matter, use set for efficiency
    allColumns  = self.dataFrame.columns
    dataColumns = list( set( allColumns ) - set( a.removeColumns ) )

    for d in range( 1, a.D + 1 ) :
        # For each d evaluate nested list of columns
        # Each sublist consists of current d MDEcolumns plus columns
        # Order doesn't matter, use set for efficiency
        # First, set columns to list of all available non MDEcolumns
        columns = list( set( dataColumns ) - set( self.MDEcolumns ) )
        # Create nested list of each available column with MDEcolumns
        columns = [ [c] + self.MDEcolumns for c in columns ]

        if a.debug:
            LogMsg(f'\n{d}-D Cross Map {a.target} -> {columns[:5]}')
            LogMsg( '---------------------------------------------------------' )
            LogMsg( f'   CrossMapColumns -> {datetime.now()}' )

        # rhoD is dict of 'columns:target' : (rho, [columns]) pairs.
        # Note embedded = True
        rhoD = CrossMapColumns( self.dataFrame,
                                columns         = columns,
                                target          = a.target, # E = 0,
                                Tp              = a.Tp,
                                tau             = a.tau,
                                exclusionRadius = a.exclusionRadius,
                                lib             = a.lib,
                                pred            = a.pred,
                                embedded        = True,
                                cores           = a.cores,
                                mpMethod        = a.mpMethod,
                                chunksize       = a.chunksize,
                                noTime          = a.noTime,
                                verbose         = a.verbose )

        # Sort rhoD values by decreasing rho
        # L_rhoD is ranked list of tuples : (rho, [columns])
        L_rhoD = sorted( rhoD.values(), key = lambda x:x[0], reverse = True )

        # Discard L_rhoD elements below a.crossMapRhoMin
        rhoD_ = array( [ _[0] for _ in L_rhoD ] )
        rhoD_crossMapRhoMin = rhoD_ > a.crossMapRhoMin # boolean array
        rhoD_N = len( rhoD_[ rhoD_crossMapRhoMin ] )   # number of [] > RhoMin

        if rhoD_N < 1 :
            # Failed to pass crossMapRhoMin criteria
            LogMsg( f'{d}-D failed to find valid cross map.' )
            continue

        elif rhoD_N < len( L_rhoD ) :
            L_rhoD = L_rhoD[:rhoD_N] # truncate L_rhoD

        # Add this L_rhoD to rhoD
        self.rhoD[ d ] = L_rhoD

        if a.debug:
            LogMsg( f'   CrossMapColumns <- {datetime.now()}' )
            LogMsg( f'      L_rhoD[:3] {L_rhoD[:3]}' )

        if a.noCCM :
            # Take first L_rhoD column with highest ranked rho
            newColumn = L_rhoD[0][1]
            self.MDEcolumns.append( newColumn[0] )
            self.MDErho = append( self.MDErho, L_rhoD[0][0] )
        else :
            #------------------------------------------------------------
            # Validate addition of newColumn with CCM
            #------------------------------------------------------------
            maxCol_i  = None
            newColumn = None

            for col_i in range( len( L_rhoD )  ) :

                columns_i = L_rhoD[col_i][1] # List of columns
                newColumn = columns_i[ 0 ]   # First item is new column

                #--------------------------------------------------------
                # First need CCM embedding dimension
                #   If args.E provided use it for CCM with the maxRhoEDim
                #   threshold criteria from CrossMapColumns rho
                #   If args.E not provided: use EmbedDimension() to find
                #   EDim and maxRhoEDim
                if a.E > 0:
                    # Takens embedding dimension specified in args
                    maxEDim    = a.E
                    maxRhoEDim = L_rhoD[col_i][0].round(4) # CrossMapColumns rho
                else :
                    # Estimate E as local or global maximum EmbedDimension()
                    if a.debug :
                        LogMsg( f'   EmbedDimension -> {datetime.now()}' )
                        LogMsg( f'      L_rhoD[col_i] {columns_i}' )

                    EDimDF = EmbedDimension( dataFrame       = self.dataFrame,
                                             columns         = newColumn,
                                             target          = a.target,
                                             maxE            = a.maxE,
                                             lib             = a.lib,
                                             pred            = a.pred,
                                             Tp              = a.Tp,
                                             tau             = a.tau,
                                             exclusionRadius = a.exclusionRadius,
                                             validLib        = [],
                                             noTime          = a.noTime,
                                             verbose         = a.verbose,
                                             numProcess      = a.maxE,
                                             showPlot        = False )

                    if a.debug :
                        LogMsg( f'   EmbedDimension <- {datetime.now()}' )

                    # If firstEMax True, return first (lowest E) local maximum
                    # Find max E(rho)
                    if a.firstEMax :
                        iMax = argrelextrema(EDimDF['rho'].to_numpy(),greater)[0]

                        if len( iMax ) :
                            iMax = iMax[0] # first element of array
                        else :
                            # no local maxima, last is max
                            iMax = len( EDimDF['E'] ) - 1
                    else :
                        iMax = EDimDF['rho'].round(4).argmax() # global maximum

                    maxRhoEDim = EDimDF['rho'].iloc[ iMax ].round(4)
                    maxEDim    = EDimDF['E'].iloc[ iMax ]

                if a.debug :
                    LogMsg(f'      EDim {columns_i[0]} ' +\
                           f'E {maxEDim} rho {maxRhoEDim}')

                # Qualify embedding/EDim with maxRhoEDim threshold
                if maxRhoEDim < a.embedDimRhoMin :
                    continue # Keep looking

                self.EDim[ f'{newColumn}:{a.target}' ] = maxEDim
                # Finished looking for maxEDim
                #--------------------------------------------------------

                #-------------------------------------------------------------
                # CCM
                #-------------------------------------------------------------
                if a.debug :
                    LogMsg( f'   CCM -> {datetime.now()}' )
                    LogMsg( f'      {newColumn}:{a.target}' )

                ccmDF = CCM( dataFrame       = self.dataFrame,
                             columns         = newColumn,
                             target          = a.target,
                             libSizes        = self.libSizes,
                             sample          = a.sample,
                             E               = maxEDim,
                             Tp              = a.Tp,
                             tau             = a.tau,
                             exclusionRadius = a.exclusionRadius,
                             seed            = a.ccmSeed,
                             noTime          = a.noTime )

                if a.debug:
                    LogMsg( f'   CCM <- {datetime.now()}' )
                    LogMsg( ccmDF.to_string() )

                ccmVals = ccmDF[ f'{a.target}:{newColumn}' ].to_numpy()

                # Slope of linear fit to rho(libSizes)
                lm = LinearRegression().fit( self.libSizesVec,
                                             nan_to_num( ccmVals ) )
                slope = round( lm.coef_[0], 5 )
            
                if a.debug :
                    LogMsg( f'   {a.target}:{newColumn} slope {slope}' )

                if slope > a.ccmSlope :
                    maxCol_i = col_i
                    break # This vector is good
                else :
                    maxCol_i = None

            # <---- for col_i in range( len( L_rhoD )  ) :
            # <-------------------------------------------

            if maxCol_i is not None :
                # Add newColumn to MDEcolumns
                self.MDEcolumns.append( newColumn )
                self.MDErho = append( self.MDErho, L_rhoD[maxCol_i][0] )
            else :
                # Failed to pass CCM criteria
                LogMsg( f'{d}-D CCM failed to find columns.' )
                break

        if a.verbose :
            LogMsg(f'{d}-D {self.MDEcolumns} rho {self.MDErho[-1]}')

    #-------------------------------------------------------------
    # Auxiliary time delays
    #-------------------------------------------------------------
    if a.timeDelay :
        # i of maximal rho in MDEColumns
        MDE_iMax   = self.MDErho.round(4).argmax() # global maximum
        MDE_rhoMax = self.MDErho[MDE_iMax]

        if a.verbose :
            LogMsg( f'  Max rho {MDE_rhoMax} D {MDE_iMax + 1}' )

        # Add time delays
        # For each of the 1:D (D == MDE_iMax + 1 ) add args.timeDelay
        # delays to the MDEColumms, see if rho increases
        # Include target as first column to evaluate
        MDEcolumns_ = self.MDEcolumns[:MDE_iMax + 1]
        evalColumns = [a.target] + MDEcolumns_

        for evalColumn in evalColumns :

            embd = Embed(dataFrame = self.dataFrame, E = a.timeDelay + 1,
                         tau = a.tau, columns = evalColumn, includeTime = False)

            # Drop (t-0) first column
            embd = embd.iloc[:,1:]

            # DataFrame for cross mapping, target is first column
            tD_df = concat( [self.dataFrame.loc[:,evalColumns], embd], axis = 1 )

            # Cross map
            if evalColumn == a.target :
                columns = tD_df.columns
            else :
                columns = tD_df.columns[1:] # ignore target

            cmap_tD = Simplex( dataFrame = tD_df,
                               target = a.target, columns = columns,
                               lib = a.lib, pred = a.pred, Tp = a.Tp,
                               tau = a.tau, embedded = True, noTime = True )
            cmap_tD_rho = ComputeError( cmap_tD['Observations'],
                                        cmap_tD['Predictions'] )['rho']

            if a.verbose :
                msg = f'Embed [{evalColumn}] + {a.timeDelay} ' +\
                      f'rho {cmap_tD_rho}'
                self.LogMsg( msg )

    self.elapsedTime = datetime.now() - self.startTime

    self.MDEOut = DataFrame( { 'variables' : self.MDEcolumns,
                               'rho'       : self.MDErho } )

    if a.verbose :
        msg = f'\nManifold Dimensional Expansion <------\n' +\
              f'  Run() Elapsed time {self.elapsedTime}\n'  +\
              f'  {datetime.now()}\n--------------------------------------'
        self.LogMsg( msg )
        self.LogMsg( self.MDEOut.to_string() )

    if a.outFile or a.outCSV :
        self.Output()

    if a.plot :
        self.Plot()
