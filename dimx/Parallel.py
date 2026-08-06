
'''Multiprocessing infrastructure for dimx MDE cross-map sweeps.

This module owns the *plumbing* around the cross-map step:
  - start-method selection that never uses the deprecated 'fork' context
  - numeric-only data preparation (the leading time column is not needed)
  - data delivery to workers via initargs (small data) or a single
    shared-memory block (large data), gated by a byte threshold
  - a run-scoped worker pool (CrossMapPool) reused across all dimensions
  - per-dimension progress reporting via imap_unordered

The embedded pyEDM Simplex call inside SimplexWorker is left unchanged;
only how the data and pool reach it are managed here.  Workers always run
against a numeric-only frame, so pyEDM noTime is forced True in the worker
regardless of the user's noTime setting (the time column, if any, was
dropped during frame preparation).
'''

import os, uuid
from multiprocessing import ( get_context, get_start_method,
                              get_all_start_methods, shared_memory )

import numpy as np
from pandas import DataFrame
from pyEDM  import Simplex, ComputeError

# Native-math thread env vars pinned to 1 in workers to avoid CPU
# oversubscription (pool workers x internal BLAS/OpenMP threads).
_THREAD_VARS = ( 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS' )

# Worker-side globals, populated once per worker by InitWorker().
_WORKER = { 'df' : None, 'args' : None, 'shm' : None }

#----------------------------------------------------------------------------
def ResolveStartMethod( requested = None ):
    '''Choose a multiprocessing start method, never 'fork'.

       - An explicit, non-fork request is honored.
       - Otherwise use the system default if it is not 'fork'.
       - If the default is 'fork', use 'forkserver' when available, else
         'spawn'.
    '''
    if requested and requested != 'fork' :
        return requested

    default = get_start_method( allow_none = False )
    if default != 'fork' :
        return default

    if 'forkserver' in get_all_start_methods() :
        return 'forkserver'
    return 'spawn'

#----------------------------------------------------------------------------
def DataFrameBytes( df ):
    '''User-expected DataFrame size in bytes (compared against
       sharedMem, which is given in decimal MB).'''
    return int( df.memory_usage( deep = True ).sum() )

#----------------------------------------------------------------------------
def PrepareNumericFrame( df, noTime, verbose = False, logMsg = None ):
    '''Return a contiguous float64, numeric-only DataFrame for cross mapping.

       If noTime is False the first column is the time vector; it is dropped
       since it is not used by MDE.  Mixed / non-float64 numeric dtypes are
       upcast to float64 (a single warning is issued: results are then
       numerically equivalent to tolerance rather than bit-identical).

       Returns ( numericDF, upcast ).  numericDF carries a default RangeIndex;
       pyEDM lib/pred are 1-offset row positions, so the index is irrelevant.
    '''
    numeric = df.iloc[ :, 1: ] if not noTime else df

    dtypes = set( str( t ) for t in numeric.dtypes )
    upcast = len( dtypes ) > 1 or dtypes != { 'float64' }

    if verbose and upcast and logMsg is not None :
        logMsg( 'PrepareNumericFrame(): mixed/non-float64 numeric dtypes '
                'upcast to float64; results numerically equivalent to '
                'tolerance.' )

    arr     = np.ascontiguousarray( numeric.to_numpy( dtype = 'float64' ) )
    numeric = DataFrame( arr, columns = list( numeric.columns ), copy = False )
    return numeric, upcast

#----------------------------------------------------------------------------
def _MakeSharedBlock( arr ):
    '''Create a per-run uniquely named SharedMemory block holding arr.
       Name is greppable (prefix), attributable (pid), collision-resistant
       (uuid4 fragment), and short enough for platform name limits.'''
    name = f'mde_{os.getpid()}_{uuid.uuid4().hex[:12]}'
    shm  = shared_memory.SharedMemory( create = True, size = arr.nbytes,
                                       name = name )
    view = np.ndarray( arr.shape, dtype = arr.dtype, buffer = shm.buf )
    view[ : ] = arr[ : ]
    return shm, name

def _AttachShared( name, shape, columns ):
    '''Attach (zero-copy) to a shared block by name and wrap as a DataFrame.'''
    shm  = shared_memory.SharedMemory( name = name )
    view = np.ndarray( tuple( shape ), dtype = np.float64, buffer = shm.buf )
    df   = DataFrame( view, columns = list( columns ), copy = False )
    return shm, df

#----------------------------------------------------------------------------
def InitWorker( argsD, transport ):
    '''Pool initializer: establish worker-global data + run-constant args.

       transport is one of:
         { 'mode':'initargs', 'frame': <DataFrame> }
         { 'mode':'shared', 'name':str, 'shape':tuple, 'columns':list }
    '''
    # Best-effort thread pinning (effective when also set in the parent
    # before pool creation, which CrossMapPool does).
    for var in _THREAD_VARS :
        os.environ.setdefault( var, '1' )

    _WORKER['args'] = argsD

    if transport['mode'] == 'shared' :
        shm, df = _AttachShared( transport['name'],
                                 transport['shape'],
                                 transport['columns'] )
        _WORKER['shm'] = shm
        _WORKER['df']  = df
    else :
        _WORKER['df']  = transport['frame']

#----------------------------------------------------------------------------
def SimplexWorker( columns ):
    '''Cross-map rho for one candidate column list.

       Data and run-constant args come from worker globals; only the small
       column list crosses the task boundary.  The embedded pyEDM Simplex
       call is unchanged except noTime is forced True (the worker frame is
       numeric-only, with any time column already removed).
    '''
    df   = _WORKER['df']
    a    = _WORKER['args']
    cols = list( columns )

    # Pool parallelizes across candidates, so each KDTree query is single
    # threaded (kdWorkers=1) to avoid nWorkers x allcores oversubscription.
    sdf = Simplex( dataFrame       = df,
                   columns         = cols,
                   target          = a['target'],
                   lib             = a['lib'],
                   pred            = a['pred'],
                   E               = a['E'],
                   embedded        = a['embedded'],
                   exclusionRadius = a['exclusionRadius'],
                   Tp              = a['Tp'],
                   tau             = a['tau'],
                   noTime          = True,
                   kdWorkers       = a.get( 'kdWorkers', 1 ) )

    rho = ComputeError( sdf['Observations'], sdf['Predictions'] )['rho']
    return rho, cols

#----------------------------------------------------------------------------
class CrossMapPool :
    '''Run-scoped worker pool for cross-map sweeps.

       Created once per MDE run and reused across every dimension, which
       amortizes pool startup / module re-import (significant under the
       spawn / forkserver contexts this code requires).  Owns the optional
       shared-memory block and guarantees idempotent teardown.

       The pool serves the cross-map sweep only.  EmbedDimension / CCM
       validation runs in the parent process (daemonic pool workers cannot
       spawn pyEDM's own child processes), with this pool idle.
    '''

    def __init__( self, numericDF, argsD, crossMapCores = None, mpMethod = None,
                  sharedMem = 0.1, maxTasks = None, logMsg = None ):

        self.logMsg   = logMsg
        self.target   = argsD['target']
        self.shm      = None
        self.shm_name = None
        self.pool     = None
        self._closed  = False

        self.method = ResolveStartMethod( mpMethod )
        ctx         = get_context( self.method )

        # Contiguous float64 backing array shared by both transports.
        arr = np.ascontiguousarray( numericDF.to_numpy( dtype = 'float64' ) )
        columns = list( numericDF.columns )

        # sharedMem is in decimal MB (1e6 bytes); 0 (or falsy) forces initargs.
        self.useShared = bool( sharedMem ) and sharedMem > 0 and \
                         DataFrameBytes( numericDF ) >= sharedMem * 1_000_000

        if self.useShared :
            self.shm, self.shm_name = _MakeSharedBlock( arr )
            transport = { 'mode'    : 'shared',
                          'name'    : self.shm_name,
                          'shape'   : arr.shape,
                          'columns' : columns }
        else :
            transport = { 'mode'  : 'initargs',
                          'frame' : DataFrame( arr, columns = columns,
                                               copy = True ) }

        # Size for the widest sweep; idle workers at later dimensions are free.
        # crossMapCores is the sweep-pool cap: None means use the machine,
        # otherwise it is an upper bound.  Resolve None before the min().
        cpu      = os.cpu_count() or 1
        cap      = crossMapCores if crossMapCores is not None else cpu
        upper    = maxTasks if maxTasks else cap
        nWorkers = max( 1, min( cap, upper, cpu ) )
        self.nWorkers = nWorkers

        # Pin native math threads in children (inherited at pool creation),
        # then restore the parent's environment so validation phase pyEDM
        # parallelism is unaffected.
        saved = { k : os.environ.get( k ) for k in _THREAD_VARS }
        for k in _THREAD_VARS :
            os.environ[ k ] = '1'
        try :
            self.pool = ctx.Pool( processes   = nWorkers,
                                  initializer = InitWorker,
                                  initargs    = ( argsD, transport ) )
        finally :
            for k, v in saved.items() :
                if v is None :
                    os.environ.pop( k, None )
                else :
                    os.environ[ k ] = v

        if self.logMsg is not None :
            self.logMsg( f'\tCrossMapPool start={self.method} '
                      f'workers={nWorkers} '
                      f'transport={"shared" if self.useShared else "initargs"}' )

    #------------------------------------------------------------------------
    def CrossMap( self, candidateColumns, dimension = 1,
                  logPct = 0, verbose = False ):
        '''Cross-map each candidate column list to the target.

           Returns { 'c0,c1,...:target' : (rho, [cols]) }.  Results are
           assembled from each task's returned column list (imap_unordered
           is not order-preserving), so the dict is identical regardless of
           completion order.

           Progress (verbose only): logPct is a band width in percent.
           Only the highest band crossed is emitted, reset each dimension.
           logPct unset / 0 -> silent; logPct >= 100 -> one message at
           completion.
        '''
        total  = len( candidateColumns )
        result = {}

        bandMode     = ( verbose and self.logMsg is not None
                         and logPct and 0 < logPct < 100 )
        completeMode = ( verbose and self.logMsg is not None
                         and logPct and logPct >= 100 )
        lastBand = 0

        done = 0
        for rho, cols in self.pool.imap_unordered( SimplexWorker,
                                                   candidateColumns,
                                                   chunksize = 1 ) :
            key = ','.join( cols )
            result[ f'{key}:{self.target}' ] = ( rho, cols )
            done += 1

            if bandMode :
                pct  = 100.0 * done / total
                band = int( pct // logPct )
                if band > lastBand :
                    lastBand = band
                    self.logMsg( f'\t{dimension}-D cross map '
                                 f'{done}/{total} ({pct:.0f}%)' )

        if completeMode :
            self.logMsg( f'\t{dimension}-D cross map {total}/{total} (100%)' )

        return result

    #------------------------------------------------------------------------
    def close( self ):
        '''Idempotently tear down the pool and unlink the shared block.
           Safe to call multiple times and after partial construction.'''
        if self._closed :
            return
        self._closed = True

        if self.pool is not None :
            try :
                self.pool.close()
                self.pool.join()
            except Exception :
                try :
                    self.pool.terminate()
                except Exception :
                    pass
            finally :
                self.pool = None

        if self.shm is not None :
            try :
                self.shm.close()
            except Exception :
                pass
            try :
                self.shm.unlink()          # exactly-once, by the creator
            except FileNotFoundError :
                pass                       # already gone
            except Exception :
                pass
            finally :
                self.shm = None

    # Context-manager sugar so Run() can bracket the whole pooled section.
    def __enter__( self ):
        return self

    def __exit__( self, exc_type, exc, tb ):
        self.close()
        return False
