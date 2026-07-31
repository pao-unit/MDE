# Distribution modules
from datetime import datetime
from os       import cpu_count as _cpu_count

# Community modules
from pandas import DataFrame, concat
from numpy  import append, array, greater, nan_to_num, float32
from pyEDM  import ComputeError, Embed, EmbedDimension, CCM, Simplex
from scipy.signal         import argrelextrema
from sklearn.linear_model import LinearRegression

# Local modules
from .Parallel import CrossMapPool, PrepareNumericFrame, ResolveStartMethod

#-------------------------------------------------------------------
#-------------------------------------------------------------------
def Run( self ):
    '''Execute MDE workflow'''

    self.startTime = datetime.now()

    # Shortcuts in local scope
    a      = self.args
    LogMsg = self.LogMsg

    self.Validate() # Sets self.libSizes from pLibSizes if not specified

    # libSizes ndarray for CCM convergence slope estimate.
    # Normalize by data length N (pyEDM / CCM_Matrix convention) so the
    # internally computed slope is on the same scale as a supplied matrix.
    self.libSizesVec = array( self.libSizes, dtype = float ).reshape(-1, 1)
    self.libSizesVec = self.libSizesVec / self.dataFrame.shape[0]

    if a.debug :
        LogMsg( f'libSizes {self.libSizes}  libSizesVec {self.libSizesVec}')

    # Numeric-only frame used by both the worker sweep and the parent-side
    # EmbedDimension / CCM validation.  The time column (if noTime is False)
    # is dropped; pyEDM noTime is therefore forced True everywhere below.
    numericDF, _ = PrepareNumericFrame( self.dataFrame, a.noTime, LogMsg )

    # mpMethod for pyEDM's own internal pools (EmbedDimension/CCM) - never fork
    edmMethod = ResolveStartMethod( a.mpMethod )
    # EmbedDimension parallelizes its E-sweep across processes; cap that count
    # to available cores so a large maxE does not spawn a process storm.
    edmProcs = max( 1, min( a.maxE, _cpu_count() or 1 ) )

    # Set initial dataColumns as allColumns minus removeColumns
    # Candidate columns come from the numeric frame (the time column is never
    # a candidate).  Order doesn't matter; use set for efficiency.
    dataColumns = list( set( numericDF.columns ) - set( a.removeColumns ) )

    # Slope-matrix coverage: every candidate column must be a predicted
    # (column) label in the matrix, since lookups are slopeMatrix.loc[
    # target, candidate ].  Fail fast on a matrix that does not match data.
    if self.slopeMatrix is not None :
        missing = [ c for c in dataColumns if c not in self.slopeMatrix.columns ]
        if missing :
            msg = f'Run() slope matrix missing candidate columns: {missing}'
            LogMsg( msg )
            raise RuntimeError( msg )
        if a.verbose :
            LogMsg( f'Run(): slope matrix covers all {len(dataColumns)} '
                    'candidate columns.' )

    # Per-run compute caches (column -> result), cleared each Run().
    # These hold every evaluated column, including ones that fail a threshold
    # (negative caching), so a recurring candidate is never recomputed.
    self._edimCache = {}   # column : (maxEDim, maxRhoEDim)
    self._ccmCache  = {}   # column : slope

    # CCM cache validity precondition: a recurring column's slope is reused,
    # which only matches a fresh recompute when CCM is deterministic, i.e.
    # ccmSeed is fixed.  Warn once if caching CCM under a random seed.
    # Not applicable when slopes are read from a supplied matrix (no CCM run).
    if ( not a.noCCM ) and self.slopeMatrix is None and a.ccmSeed is None :
        LogMsg( 'Run() warning: ccmSeed is None; CCM uses random library '
                'samples, so cached slopes will not match a recompute. '
                'Set ccmSeed to a fixed integer for reproducible selection.' )

    # Run-constant args for the worker Simplex (noTime forced True in worker).
    # kdWorkers=1: the pool parallelizes across candidates, so each query is
    # single threaded (CCM is always workers=1 in pyEDM 2.5.3).
    # Note embedded = True
    argsD = { 'target'          : a.target,
              'lib'             : a.lib,
              'pred'            : a.pred,
              'E'               : 0,
              'embedded'        : True,
              'exclusionRadius' : a.exclusionRadius,
              'Tp'              : a.Tp,
              'tau'             : a.tau,
              'kdWorkers'       : getattr( a, 'kdWorkers', 1 ) }

    # Widest sweep is d = 1 : every non-removed candidate column.
    maxTasks = len( dataColumns )

    # One persistent pool for the whole run; teardown guaranteed in finally.
    pool = CrossMapPool( numericDF, argsD,
                         crossMapCores = a.crossMapCores,
                         mpMethod  = a.mpMethod,
                         sharedMem = getattr( a, 'sharedMem', 0.1 ),
                         maxTasks  = maxTasks,
                         logMsg    = LogMsg if a.verbose else None )
    try :
        logPct = getattr( a, 'logPct', 0 )

        for d in range( 1, a.D + 1 ) :
            # For each d evaluate nested list of columns
            # Each sublist consists of current d MDEcolumns plus columns
            # Order doesn't matter, use set for efficiency
            # First, set columns to list of all available non MDEcolumns
            columns = list( set( dataColumns ) - set( self.MDEcolumns ) )
            # Each candidate combination = new column + current MDEcolumns.
            columns = [ [c] + self.MDEcolumns for c in columns ]

            if a.debug:
                LogMsg(f'\n{d}-D Cross Map {a.target} -> {columns[:5]}')
                LogMsg( '---------------------------------------------------' )
                LogMsg( f'   CrossMapColumns -> {datetime.now()}' )

            # rhoD is dict of 'columns:target' : (rho, [columns]) pairs.
            # Note embedded = True
            rhoD_cmap = pool.CrossMap( columns, dimension = d,
                                       logPct = logPct, verbose = a.verbose )

            # Rank by decreasing rho
            L_rhoD = sorted( rhoD_cmap.values(), key = lambda x:x[0], reverse = True )

            # Discard elements below crossMapRhoMin
            rhoD_  = array( [ _[0] for _ in L_rhoD ] )
            rhoD_N = int( ( rhoD_ > a.crossMapRhoMin ).sum() )

            if rhoD_N < 1 :
                # No candidate exceeds crossMapRhoMin: dimension d has no
                # manifold to expand from, so dimensions d+1 .. D are
                # undefined. Terminate expansion, mirroring the CCM
                # dead-end (maxCol_i is None) break below. Record an
                # explicit empty passing subset at the terminal dimension.
                if self.slopeMatrix is not None :
                    self.rhoD_CCM[ d ] = []
                LogMsg( f'{d}-D failed to find valid cross map; '
                        'terminating expansion.' )
                break
            elif rhoD_N < len( L_rhoD ) :
                L_rhoD = L_rhoD[:rhoD_N]

            # Store only the first column of each candidate. columns[1:] are
            # the previously-selected components, already in self.MDEcolumns
            # L_rhoD keeps full column lists for the selection loop below.
            self.rhoD[ d ] = [ ( rho_i.astype( float32 ), cols_i[0] )
                               for rho_i, cols_i in L_rhoD ]

            # When a slopeMatrix is supplied, the CCM slope of every candidate
            # is a free lookup, so record the full subset of L_rhoD passing
            # CCM convergence (slope > ccmSlope) here, before the greedy loop
            # below stops at the first passer. Uses the identical lookup the
            # selection loop uses, so the subset matches selection exactly
            # (NaN cells fail > ccmSlope and are excluded, as in selection).
            # L_rhoD is ranked by decreasing rho; filtering preserves that
            # order. An empty list marks a dimension where nothing passed.
            if self.slopeMatrix is not None :
                passed = []
                for rho_i, cols_i in L_rhoD :
                    newCol_i = cols_i[ 0 ]
                    slope_i  = self.slopeMatrix.loc[ newCol_i, a.target ]
                    if slope_i > a.ccmSlope :
                        passed.append( ( rho_i.astype( float32 ), newCol_i, slope_i ) )

                self.rhoD_CCM[ d ] = passed

            if a.debug:
                LogMsg( f'   CrossMapColumns <- {datetime.now()}' )
                LogMsg( f'      L_rhoD[:3] {L_rhoD[:3]}' )

            if a.noCCM :
                newColumn = L_rhoD[0][1]
                self.MDEcolumns.append( newColumn[0] )
                self.MDErho = append( self.MDErho, L_rhoD[0][0] )
            else :
                #--------------------------------------------------------
                # Validate addition of newColumn with CCM
                #--------------------------------------------------------
                maxCol_i  = None
                newColumn = None

                for col_i in range( len( L_rhoD ) ) :

                    columns_i = L_rhoD[col_i][1] # list of columns
                    newColumn = columns_i[ 0 ]   # first item is new column

                    if self.slopeMatrix is not None :
                        #------------------------------------------------
                        # Slope read from the supplied CCM slope matrix.
                        # No EmbedDimension, no CCM, no embedDimRhoMin gate.
                        # NaN cells (diagonal / sparse) fail slope > ccmSlope
                        # and the candidate is skipped, as a failed CCM would.
                        #------------------------------------------------
                        slope = self.slopeMatrix.loc[ newColumn, a.target ]

                        if a.debug :
                            LogMsg( f'   slope matrix {newColumn}:{a.target} '
                                    f'slope {slope}' )
                    else :
                        #------------------------------------------------
                        # CCM embedding dimension (cached per column)
                        #------------------------------------------------
                        if a.E > 0 :
                            # Static E specified : use it with CrossMap rho
                            maxEDim    = a.E
                            maxRhoEDim = round( float( L_rhoD[col_i][0] ), 4 )
                        elif newColumn in self._edimCache :
                            maxEDim, maxRhoEDim = self._edimCache[ newColumn ]
                        else :
                            if a.debug :
                                LogMsg( f'   EmbedDimension -> {datetime.now()}' )
                                LogMsg( f'      {columns_i}' )

                            EDimDF = EmbedDimension(
                                         dataFrame       = numericDF,
                                         columns         = newColumn,
                                         target          = a.target,
                                         maxE            = a.maxE,
                                         lib             = a.lib,
                                         pred            = a.pred,
                                         Tp              = a.Tp,
                                         tau             = a.tau,
                                         exclusionRadius = a.exclusionRadius,
                                         validLib        = [],
                                         noTime          = True,
                                         mpMethod        = edmMethod,
                                         verbose         = a.verbose,
                                         numProcess      = edmProcs,
                                         kdWorkers       = 1,
                                         showPlot        = False )

                            if a.debug :
                                LogMsg( f'   EmbedDimension <- {datetime.now()}' )

                            if a.firstEMax :
                                iMax = argrelextrema( EDimDF['rho'].to_numpy(),
                                                      greater )[0]
                                iMax = iMax[0] if len( iMax ) \
                                       else len( EDimDF['E'] ) - 1
                            else :
                                iMax = EDimDF['rho'].round(4).argmax()

                            maxRhoEDim = EDimDF['rho'].iloc[ iMax ].round(4)
                            maxEDim    = EDimDF['E'].iloc[ iMax ]

                            # Cache before any threshold test (negative cache).
                            self._edimCache[ newColumn ] = ( maxEDim,
                                                             maxRhoEDim )

                        if a.debug :
                            LogMsg( f'      EDim {newColumn} '
                                    f'E {maxEDim} rho {maxRhoEDim}' )

                        # Qualify EDim with maxRhoEDim threshold
                        if maxRhoEDim < a.embedDimRhoMin :
                            continue # keep looking

                        self.EDim[ f'{newColumn}:{a.target}' ] = maxEDim

                        #------------------------------------------------
                        # CCM convergence slope (cached per column)
                        #------------------------------------------------
                        if newColumn in self._ccmCache :
                            slope = self._ccmCache[ newColumn ]
                        else :
                            if a.debug :
                                LogMsg( f'   CCM -> {datetime.now()}' )
                                LogMsg( f'      {newColumn}:{a.target}' )

                            ccmDF = CCM( dataFrame       = numericDF,
                                         columns         = newColumn,
                                         target          = a.target,
                                         libSizes        = self.libSizes,
                                         sample          = a.sample,
                                         E               = maxEDim,
                                         Tp              = a.Tp,
                                         tau             = a.tau,
                                         exclusionRadius = a.exclusionRadius,
                                         seed            = a.ccmSeed,
                                         mpMethod        = edmMethod,
                                         noTime          = True )

                            if a.debug:
                                LogMsg( f'   CCM <- {datetime.now()}' )
                                LogMsg( ccmDF.to_string() )

                            ccmVals = \
                                ccmDF[ f'{a.target}:{newColumn}' ].to_numpy()
                            lm = LinearRegression().fit( self.libSizesVec,
                                                         nan_to_num( ccmVals ) )
                            slope = round( lm.coef_[0], 5 )
                            self._ccmCache[ newColumn ] = slope

                        if a.debug :
                            LogMsg( f'   {a.target}:{newColumn} slope {slope}' )
                    # <---- else: if self.slopeMatrix is not None :

                    # Validate causality with slope exceedance
                    if slope > a.ccmSlope :
                        maxCol_i = col_i
                        break # this vector is good
                    else :
                        maxCol_i = None

                # <---- for col_i ...

                if maxCol_i is not None :
                    self.MDEcolumns.append( newColumn )
                    self.MDErho = append( self.MDErho, L_rhoD[maxCol_i][0] )
                else :
                    LogMsg( f'{d}-D CCM failed to find columns.' )
                    break

            if a.verbose :
                LogMsg( f'{d}-D {self.MDEcolumns} rho {self.MDErho[-1]}' )

    finally :
        # Guaranteed teardown: pool join + shared-memory unlink, even on error.
        pool.close()

    #-------------------------------------------------------------
    # Auxiliary time delays
    #-------------------------------------------------------------
    if a.timeDelay :
        MDE_iMax   = self.MDErho.round(4).argmax()
        MDE_rhoMax = self.MDErho[ MDE_iMax ]

        if a.verbose :
            LogMsg( f'  Max rho {MDE_rhoMax} D {MDE_iMax + 1}' )

        MDEcolumns_ = self.MDEcolumns[:MDE_iMax + 1]
        evalColumns = [ a.target ] + MDEcolumns_

        for evalColumn in evalColumns :

            embd = Embed( dataFrame = numericDF, E = a.timeDelay + 1,
                          tau = a.tau, columns = evalColumn,
                          includeTime = False )
            embd = embd.iloc[ :, 1: ] # drop (t-0) first column

            tD_df = concat( [ numericDF.loc[ :, evalColumns ], embd ],
                            axis = 1 )

            if evalColumn == a.target :
                columns = tD_df.columns
            else :
                columns = tD_df.columns[1:] # ignore target

            # Note embedded = True
            cmap_tD = Simplex( dataFrame = tD_df,
                               target    = a.target,
                               columns   = columns,
                               lib       = a.lib,
                               pred      = a.pred,
                               Tp        = a.Tp,
                               tau       = a.tau,
                               embedded  = True,
                               noTime    = True )
    
            cmap_tD_rho = ComputeError( cmap_tD['Observations'],
                                        cmap_tD['Predictions'] )['rho']

            if a.verbose :
                LogMsg( f'Embed [{evalColumn}] + {a.timeDelay} '
                        f'rho {cmap_tD_rho}' )

    self.elapsedTime = datetime.now() - self.startTime

    self.MDEOut = DataFrame( { 'variables' : self.MDEcolumns,
                               'rho'       : self.MDErho } )

    if a.verbose :
        LogMsg( f'\nManifold Dimensional Expansion <------\n'
                f'  Run() Elapsed time {self.elapsedTime}\n'
                f'  {datetime.now()}\n'
                '--------------------------------------' )
        LogMsg( self.MDEOut.to_string() )

    if a.outFile or a.outCSV :
        self.Output()

    if a.plot :
        self.Plot()
