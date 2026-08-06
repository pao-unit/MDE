# Python distribution modules
from collections import deque
from dataclasses import dataclass, fields, replace
from datetime    import datetime
from typing      import Optional
from pickle      import dump
import gzip
import os

# Community modules
from pandas import DataFrame

# Local modules
from .Config import MDEConfig
from .MDE    import MDE

#-----------------------------------------------------------------------
@dataclass
class Task:
    '''One pending MDE run and the context its removeColumns needs.'''
    target: str                  # column MDE explains ( driven variable )
    parent: Optional[str] = None # target one level up; None at the root
    depth:  int           = 0    # 0 at the root, +1 per level below

#-----------------------------------------------------------------------
class ReverseMDE:
    '''Reverse ( backward ) driver discovery over an MDE variable set.

    Runs MDE.Run() on a root target, then re-runs MDE on each
    discovered driver as its own target, walking backward along
    causal influence until every reachable variable is expanded, or
    maxDepth is reached. Runs are serial: MDE self-parallelizes via
    crossMapCores.

    Configuration reuses dimx.Config.MDEConfig ( single source of
    truth ). The root target and root removeColumns are read from the
    supplied config; child runs derive a per-target config with
    dataclasses.replace(), overriding target, removeColumns and plot.

    Two entry modes, distinguished in __init__ :

      1. Compute the root. Supply target ( via config or overrides )
         and nothing else. The root is run to discover its drivers.

      2. Supply the root. Provide a pre-existing MDEOut ( DataFrame )
         or a list of reverseVariables together with the target they
         belong to. The root is NOT run; its drivers seed the depth-1
         children directly. target is required: it labels the root
         node in GraphOut and parents the depth-1 children.

    A user-supplied slopeMatrix ( optional ) is a runtime object, not
    configuration: it is passed unchanged to every MDE run. Supplying
    it skips per-run EmbedDimension & CCM, amortizing the cross-map
    cost across the entire driver graph.

    verbose       adds per-target headers and driver counts
    logProgress   adds a one-time context header plus a slim progress line
                  throttled to every logEvery expansions or every logEveryPct
                  percent of N ( which wins over logEvery when set )
    quietChildren derives child runs with verbose = False.

    The DataFrame is obtained in __init__ : passed directly, or read
    once from config.dataFile via the shared MDE.LoadDataFrame seam -
    raw read; columnNames / removeTime stay per MDE run.
    '''

    #-------------------------------------------------------------------
    def __init__( self,
                  dataFrame        = None,  # pandas DataFrame ( runtime )
                  slopeMatrix      = None,  # CCM slope DataFrame ( runtime )
                  config           = None,  # MDEConfig; config.target = root
                  args             = None,  # argparse Namespace ( CLI path )
                  MDEOut           = None,  # supplied root drivers ( frame )
                  reverseVariables = None,  # supplied root drivers ( list )
                  maxDepth         = None,  # None => bounded only by Visited
                  logProgress      = False, # False => no progress line
                  logEvery         = 1,     # emit every n completed runs
                  logEveryPct      = None,  # or every p percent of N ( wins )
                  outFileInterval  = None,  # checkpoint every N minutes ( None off )
                  quietChildren    = False, # False => children inherit verbose
                  **overrides ):            # MDEConfig field overrides
        '''Resolve one base MDEConfig, seed the work list, and
        initialize the result graph. No MDE is run until Run().

        Config precedence matches MDE.__init__ : args ( CLI ) if
        given; else an explicit config with optional **overrides;
        else a config built from **overrides on MDEConfig defaults.
        '''
        if args is not None :
            # CLI path : map the argparse Namespace onto MDEConfig by
            # field name; unknown Namespace keys are ignored.
            known = { f.name for f in fields( MDEConfig ) }
            config = MDEConfig( **{ k : v for k, v in vars( args ).items()
                                    if k in known } )
            # Reverse-only fields are not MDEConfig fields: read them
            # from the Namespace so the CLI wrapper stays a one-liner.
            # An explicit keyword still wins over the Namespace value.
            maxDepth      = getattr( args, 'maxDepth',      maxDepth )
            logProgress   = getattr( args, 'logProgress',   logProgress )
            logEvery      = getattr( args, 'logEvery',      logEvery )
            logEveryPct   = getattr( args, 'logEveryPct',   logEveryPct )
            outFileInterval = getattr( args, 'outFileInterval',
                                       outFileInterval )
            quietChildren = getattr( args, 'quietChildren', quietChildren )

            if reverseVariables is None :
                reverseVariables = getattr( args, 'reverseVariables', None )

        elif config is None :
            # Programmatic keyword path : defaults + caller overrides.
            config = MDEConfig( **overrides )

        elif overrides :
            # Explicit config object plus keyword overrides.
            config = replace( config, **overrides )
        # else : caller-supplied config used as-is.

        self.baseConfig    = config
        self.dataFrame     = dataFrame
        self.slopeMatrix   = slopeMatrix
        self.maxDepth      = maxDepth
        self.logProgress   = logProgress
        self.logEvery      = logEvery
        self.logEveryPct   = logEveryPct
        self.outFileInterval = outFileInterval
        self.quietChildren = quietChildren

        # Ensure the frame is in hand, then resolve lib / pred - the
        # same load-then-resolve order MDE.Validate() uses, so it is
        # unconditional here too. If no frame was passed but a dataFile
        # is set, read it once up front via the shared MDE.LoadDataFrame
        # seam ( raw read; columnNames / removeTime stay per MDE run ).
        # This makes N known at construction on the dataFile path, so
        # the ceiling and the percentage throttle work there as well.
        if self.dataFrame is None :
            if config.dataFile :
                self.dataFrame = MDE.LoadDataFrame( config )
            else :
                raise RuntimeError(
                    'ReverseMDE: dataFrame or dataFile required.' )

        # Resolve empty lib / pred once, before any MDE run, so no MDE
        # instance sees empty lists and Validate() emits no per-run
        # mutation notice. MDE.ResolveLibPred matches standalone MDE.
        # Track which were mutated so the banner reports only those.
        self._libMutated  = len( config.lib )  == 0
        self._predMutated = len( config.pred ) == 0

        if self._libMutated or self._predMutated :
            lib, pred = MDE.ResolveLibPred( self.dataFrame.shape[0] )
            if self._libMutated :
                config = replace( config, lib = lib )
            if self._predMutated :
                config = replace( config, pred = pred )
            self.baseConfig = config

        # Result members ( populated by Run() )
        self.GraphOut = dict()  # target -> MDEOut DataFrame ( the graph )
        self.Visited  = set()   # targets already expanded ( once each )
        self.Order    = []      # targets in expansion order, for tracing
        self._queue   = deque() # BFS work list of Task

        # Progress members ( updated by Run() )
        self._nDone    = 0    # expansions completed ( MDE runs finished )
        self._elapsed  = 0.0  # cumulative _RunMDE seconds, for the average
        self._nCeiling = None # upper bound on total runs ( column count )
        self._nextLog  = 1    # next _nDone at which to emit ( throttle )
        self._logStep  = 1    # threshold advance, set from N in Run()

        rootTarget = config.target
        if rootTarget is None :
            # ReverseMDE needs a root label to key the graph and to
            # parent the children; MDE.Validate() would only catch
            # this after a run, never in the supplied-root mode.
            raise RuntimeError( 'ReverseMDE: config.target required.' )

        if MDEOut is not None or reverseVariables is not None :
            # Mode 2 : root drivers supplied; the root is not run.
            if MDEOut is None :
                MDEOut = self._AsMDEOut( reverseVariables )
            self.GraphOut[ rootTarget ] = MDEOut
            self.Visited.add( rootTarget )
            self.Order.append( rootTarget )
            for driver in self._Drivers( MDEOut ) :
                if driver != rootTarget :
                    self._queue.append( Task( driver, rootTarget, 1 ) )
        else :
            # Mode 1 : compute the root.
            self._queue.append( Task( rootTarget, None, 0 ) )

        # Instantiation header :
        if config.verbose:
            if len( self.Order ) :
                mode = f'supplied root, {len(self._queue)} seed drivers'
            else :
                mode = 'computed root'

            msg = ( f'\nReverse Manifold Dimensional Expansion '
                    f'>------\n {datetime.now()}'
                    f'\n root {rootTarget}, maxDepth {self.maxDepth}, {mode}' )

            if self._libMutated :
                msg += f'\n lib set to {config.lib}'
            if self._predMutated :
                msg += f'\n pred set to {config.pred}'
            if self.outFileInterval and config.outFile :
                msg += ( f'\n checkpointing GraphOut every '
                         f'{self.outFileInterval}m to {config.outFile}' )

            msg += '\n-----------------------------------------------\n'
            self.LogMsg( msg )

    #-------------------------------------------------------------------
    def Run( self ):
        '''Drain the work list and populate self.GraphOut.

        Pop a Task; skip if its target is already Visited or its depth
        exceeds maxDepth; else mark Visited, run MDE, store the
        MDEOut, and enqueue each discovered driver as a deeper Task.
        Edges are recorded on enqueue; re-expansion is gated at pop.
        Serial by design: MDE self-parallelizes through crossMapCores.
        Results are read from self.GraphOut ( as MDE.Run() populates
        self.MDEOut ); Run() returns None so an interactive call does
        not echo the whole graph.

        verbose logs a per-target header before each run. logProgress
        logs a one-time context header, then a slim progress line every
        logEvery runs ( or every logEveryPct percent of N when set ),
        always including the first and final expansions.
        '''
        # Progress Logging ---------------------------------------------
        # N : the finite column universe. The visited set expands each
        # variable at most once, so N is a hard upper bound on total
        # runs. Frame is guaranteed in hand ( loaded in __init__ ).
        self._nCols    = self.dataFrame.shape[1]
        # Ceiling starts at N and only ratchets downward as the frontier
        # reveals a tighter exact bound ( see _Ceiling ). Monotone, so
        # ~% never runs backward.
        self._nCeiling = self._nCols
 
        # Throttle step : logEveryPct wins when set ( scale-invariant
        # for large N ); else logEvery. Pinned to N ( not the shrinking
        # ceiling ) so the emission cadence stays fixed. max( 1, . )
        # guards tiny p * N.
        if self.logEveryPct and self._nCols :
            step = round( self._nCols * self.logEveryPct / 100.0 )
            self._logStep = max( 1, step )
        else :
            self._logStep = max( 1, self.logEvery )

        self._nextLog = 1   # always emit the first completed expansion
        # Checkpoint clock : wall-time of the last GraphOut dump. Reset
        # after each write so the interval measures gap-between-writes.
        self._lastCkpt = datetime.now()
 
        if self.logProgress :
            # One-time header : the fixed context ( ceiling, ETA nature )
            # stated once so recurring lines carry only moving values.
            self.LogMsg( f'ReverseMDE: progress vs <= {self._nCols} runs '
                         f'ETA is an upper bound off a running average; '
                         f'every {self._logStep} runs' )
        #----------------------------------------------------------------
  
        while self._queue :
            task = self._queue.popleft()

            if task.target in self.Visited :
                continue
            if self.maxDepth is not None and task.depth > self.maxDepth :
                continue

            self.Visited.add( task.target )
            self.Order.append( task.target )

            if self.baseConfig.verbose :
                self.LogMsg( f"ReverseMDE: depth {task.depth} target "
                             f"'{task.target}' ( parent '{task.parent}' )" )

            t0     = datetime.now()
            MDEOut = self._RunMDE( task )
            self._elapsed += ( datetime.now() - t0 ).total_seconds()
            self._nDone   += 1

            self.GraphOut[ task.target ] = MDEOut

            # Periodic checkpoint : dump the partial GraphOut every
            # outFileInterval minutes so an unexpected termination on a
            # multi-day run leaves partial results, not nothing. Pickle
            # only ( no per-target CSV fan-out ) and atomic ( see
            # _DumpGraph ). outCSV is written once, at the end.
            if self.outFileInterval and self.baseConfig.outFile :
                if ( ( datetime.now() - self._lastCkpt ).total_seconds()
                     >= self.outFileInterval * 60 ) :
                    self._DumpGraph()
                    self._lastCkpt = datetime.now()
                    if self.logProgress :
                        self.LogMsg( f'ReverseMDE: checkpoint '
                                     f'{self._nDone} targets -> '
                                     f'{self.baseConfig.outFile}' )

            nDrivers = 0
            for driver in self._Drivers( MDEOut ) :
                if driver != task.target :
                    self._queue.append(
                        Task( driver, task.target, task.depth + 1 ) )
                    nDrivers += 1

            if self.baseConfig.verbose :
                self.LogMsg( f'ReverseMDE: -> {nDrivers} drivers' )

            if self.logProgress and self._nDone >= self._nextLog :
                self._nCeiling = self._Ceiling()
                self.LogMsg( self._ProgressLine() )
                # Advance to the next threshold past the current count,
                # so a large step never lands behind _nDone.
                while self._nextLog <= self._nDone :
                    self._nextLog += self._logStep

        # Always emit a final line so the terminal totals are logged
        # even when the throttle would have skipped the last expansion.
        if self.logProgress and self._nDone :
            self._nCeiling = self._Ceiling()
            self.LogMsg( self._ProgressLine() )

        # Persist GraphOut, mirroring MDE.Run()'s own Output() call.
        # Child MDE runs never write ( _ConfigFor blanks their outFile /
        # outCSV ), so this is the single, whole-graph write.
        if self.baseConfig.outFile or self.baseConfig.outCSV :
            self.Output()

    #-------------------------------------------------------------------
    def Output( self ):
        '''Persist GraphOut using the MDEConfig output fields.

        outCSV : write per-target CSVs, one file per node of the
                 driver graph, in Order for deterministic output.
                 Each file is {outCSV-stem}_{target}.csv in outDir,
                 written index = False as MDE.Output() does.
        outFile : pickle the whole GraphOut dict once as .pkl, or
                  .pkl.gz when the name ends in .pkl.gz.
        The two are independent: neither, either, or both are written
        by the presence of each flag, mirroring MDE.Output().
        '''
        args = self.baseConfig

        if args.outCSV :
            stem = args.outCSV
            if '.csv' in stem[-4:] :
                stem = stem[:-4]
            for target in self.Order :
                MDEOut = self.GraphOut[ target ]
                outFile = f'{args.outDir}/{stem}_{target}.csv'
                MDEOut.to_csv( outFile, index = False )

        self._DumpGraph()

    #-------------------------------------------------------------------
    def _DumpGraph( self ):
        '''Atomically pickle GraphOut to outFile ( no-op if unset ).

        Writes to outFile + '.tmp' in the same directory, then
        os.replace() onto outFile - atomic on POSIX, so a reader or the
        next run always sees either the previous complete checkpoint or
        the new one, never a torn file if the process dies mid-write.
        .pkl.gz when the name ends that way, else .pkl. Shared by the
        periodic checkpoint and the final Output() so the pickle logic
        lives in one place.
        '''
        args = self.baseConfig
        if not args.outFile :
            return
        outFile = f'{args.outDir}/{args.outFile}'
        if '.pkl.gz' not in outFile[-7:] and '.pkl' not in outFile[-4:] :
            outFile = outFile + '.pkl'
        tmpFile = outFile + '.tmp'
        if '.pkl.gz' in outFile[-7:] :
            with gzip.open( tmpFile, 'wb' ) as f :
                dump( self.GraphOut, f )
        else :
            with open( tmpFile, 'wb' ) as f :
                dump( self.GraphOut, f )
        os.replace( tmpFile, outFile )

    #-------------------------------------------------------------------
    def LogMsg( self, msg, end = '\n', mode = 'a' ):
        '''Log msg to stdout and logFile, reading self.baseConfig.

        A clone of MDE.LogMsg so reverse-level and child MDE lines
        share one console / file sink ( consoleOut, logFile, outDir ).
        '''
        args = self.baseConfig
        if args.consoleOut :
            print( msg, end = end, flush = True )
        if args.logFile :
            outFile = f'{args.outDir}/{args.logFile}'
            with open( outFile, mode ) as f:
                print( msg, end = end, file = f, flush = True )

    #-------------------------------------------------------------------
    def _ProgressLine( self ):
        '''Recurring progress line : only the moving values.

        Fixed context ( ceiling, upper-bound / average nature, step )
        is stated once in the Run() header, so this carries just done,
        queued, ~% ( when the ceiling is known ), and ~Nm left ( the
        single most actionable number ). The ETA is avg per-run times
        the remaining upper bound; omitted until a run has completed
        and whenever the ceiling is unknown.
        '''
        nQueued = len( self._queue )
        line = f'{self._nDone} done, {nQueued} queued'

        if self._nCeiling :
            pct = 100.0 * self._nDone / self._nCeiling
            line += f' (~{pct:.0f}%)'

        if self._nDone and self._nCeiling :
            avg    = self._elapsed / self._nDone
            remain = avg * max( self._nCeiling - self._nDone, 0 )
            if remain < 60 :
                line += f' ~{remain:.0f}s'
            else :
                line += f' ~{remain/60:.0f}m'

        return line

    #-------------------------------------------------------------------
    def _Ceiling( self ):
        '''Monotone upper bound on total runs, tightened per emission.

        Only queued tasks with depth <= maxDepth will ever run. When
        the deepest such task sits at maxDepth, no descendant can
        expand further, so the remaining work is exactly the unique
        unvisited targets among them - the bound is then EXACT. While
        a shallower expandable frontier remains, branching below is
        unknown, so N stands as the loose bound. maxDepth is None ->
        always N. The result never increases ( ratchets down ), so ~%
        cannot run backward.
        '''
        if self.maxDepth is None :
            return self._nCeiling   # already N; nothing to tighten

        expandable = [ t for t in self._queue if t.depth <= self.maxDepth ]
        if not expandable :
            return self._nDone      # nothing left : done == 100%

        pending = { t.target for t in expandable } - self.Visited
        if min( t.depth for t in expandable ) == self.maxDepth :
            bound = self._nDone + len( pending )   # exact
        else :
            bound = self._nCols                    # loose fallback

        return min( self._nCeiling, bound )        # monotone

    #-------------------------------------------------------------------
    def _RunMDE( self, task ):
        '''Run one MDE for task and return its MDEOut DataFrame.

        The only dimx-facing seam. slopeMatrix rides on the
        constructor unchanged for every run.
        '''
        mde = MDE( self.dataFrame,
                   slopeMatrix = self.slopeMatrix,
                   config      = self._ConfigFor( task ) )
        mde.Run()
        return mde.MDEOut

    #-------------------------------------------------------------------
    def _ConfigFor( self, task ):
        '''Per-run MDEConfig for task.

        Root ( task.parent is None ) : the base config unchanged, so
        the mode-1 root honors the user's own verbose and removeColumns.
        Child : replace( base, target, removeColumns, plot = False ),
        and verbose = False when quietChildren suppresses child
        narration.

        Each MDE run is an independent manifold that excludes only
        itself ( the target ) and the caller's structural removeColumns.
        The parent is NOT excluded: MDE links are directional ( drivers
        into the target ), so parent -> child and child -> parent are
        distinct edges, and excluding the parent would suppress a
        legitimate reverse-direction driver. Termination is guaranteed
        by the Visited set ( each node expands at most once ), not by
        parent exclusion, so unidirectional links cannot cycle.

        removeColumns is the caller's list unioned with the target
        ( order-preserving, de-duplicated ), honored on every run.
        '''
        # Output destinations belong to the reverse layer, not to each
        # MDE run: blank outFile / outCSV on every launched run ( root
        # included ) so no child clobbers the caller's file with a
        # per-run MDE pickle. ReverseMDE.Output() is the sole writer.
        if task.parent is None :
            return replace( self.baseConfig, outFile = None, outCSV = None )

        childVerbose = self.baseConfig.verbose and not self.quietChildren
        remove       = list( self.baseConfig.removeColumns )
        if task.target not in remove :
            remove.append( task.target )

        return replace( self.baseConfig,
                        target        = task.target,
                        removeColumns = remove,
                        plot          = False,
                        verbose       = childVerbose,
                        outFile       = None,
                        outCSV        = None )

    #-------------------------------------------------------------------
    def _Drivers( self, MDEOut ):
        '''Driver names from an MDEOut DataFrame ( variables column ).'''
        if 'variables' in MDEOut.columns :
            return list( MDEOut['variables'] )
        return list( MDEOut.index )

    #-------------------------------------------------------------------
    def _AsMDEOut( self, reverseVariables ):
        '''Wrap a plain list of driver names as a minimal MDEOut
        frame, so a supplied-list root reads like a computed one.'''
        return DataFrame( { 'variables' : list( reverseVariables ) } )
