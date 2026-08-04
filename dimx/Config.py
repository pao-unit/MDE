# Python distribution modules
from dataclasses import dataclass, field
from typing      import Optional, List

#-----------------------------------------------------------------------
@dataclass
class MDEConfig:
    '''Configuration parameters for the MDE class.

    Single source of truth for every MDE parameter and its default.
    Both entry points converge here:
    
        - CLI_Parser.ParseCmdLine() returns an argparse Namespace whose
          dest names match these field names, mapped on by MDE.__init__.
        - The programmatic MDE(...) constructor builds an instance
          directly from keyword overrides or an explicit MDEConfig.

    Runtime data objects (dataFrame, slopeMatrix) are deliberately NOT
    configuration: they are passed to the MDE constructor and held as
    instance attributes, and are excluded from the pickled output.

    This is a passive container: no validation lives here. All checks
    remain in MDE.Validate(). The dataclass is intentionally mutable
    (not frozen) because Validate() / CreateOutDir() resolve some fields
    in place (empty lib / pred, outDir).

    List defaults use default_factory, which also removes the shared
    mutable-default-argument hazard the old constructor signature had
    (e.g. an in-place initDataColumns.reverse() in ReadData()).
    '''

    # --- Input / data selection -----------------------------------
    dataFile:        Optional[str] = None   # DataFrame source file
    slopeMatrixFile: Optional[str] = None   # CCM slope matrix .csv / .feather
    dataName:        Optional[str] = None   # dataName in .npz archive
    removeTime:      bool          = False  # drop dataFrame first column
    noTime:          bool          = False  # first dataFrame column is data
    columnNames:     List[str] = field( default_factory = list ) # partial-match
    initDataColumns: List[str] = field( default_factory = list ) # .npy/.npz column
    removeColumns:   List[str] = field( default_factory = list ) # columns to remove

    # --- MDE expansion --------------------------------------------
    D:      int           = 3     # MDE max dimension
    target: Optional[str] = None  # target variable to predict
    lib:    List[int] = field( default_factory = list ) # EDM lib  [start, stop]
    pred:   List[int] = field( default_factory = list ) # EDM pred [start, stop]
    Tp:     int = 1                              # prediction interval
    tau:    int = -1                             # CCM embedding delay

    # --- CCM / causal inference -----------------------------------
    exclusionRadius: int  = 0      # exclusion radius: CCM, CrossMap
    sample:          int  = 20     # CCM random sample
    # CCM libSizes (programmatic only; no CLI flag)
    libSizes:        List[int] = field( default_factory = list )
    # CCM libSizes percentiles
    pLibSizes:       List[int] = field( default_factory = lambda: [10,15,85,90] )
    noCCM:           bool      = False  # do not validate with CCM
    ccmSlope:        float     = 0.01   # CCM convergence criteria
    ccmSeed:         Optional[int] = None # CCM random seed
    E:               int           = 0    # static E for all CCM (0 = auto)

    # --- Embedding dimension / thresholds -------------------------
    crossMapRhoMin: float = 0.5    # threshold for L_rhoD in Run()
    embedDimRhoMin: float = 0.5    # maxRhoEDim threshold in Run()
    maxE:           int   = 15     # maximum embedding dim for CCM
    firstEMax:      bool  = False  # use first local peak for E-dim
    timeDelay:      int   = 0      # number of time delays to add

    # --- Parallelism / execution ----------------------------------
    crossMapCores: Optional[int] = None # cross-map core cap; None = all cores
    mpMethod:      Optional[str] = None # multiprocessing start context
    chunksize:     int           = 1    # multiprocessing chunksize
    sharedMem:     float         = 0.1  # shared-mem threshold (decimal MB)
    logPct:        float         = 0    # cross-map progress band
    kdWorkers:     int           = 1    # KDTree.query workers in Simplex

    # --- Output limits / destinations -----------------------------
    maxLenRhoD:     Optional[int] = None # Output() cap on rhoD per dim
    maxLenRhoD_CCM: Optional[int] = None # Output() cap on rhoD_CCM per dim
    outDir:  str           = './'        # output directory (use pathlib for windog)
    outFile: Optional[str] = None        # MDE object -> .pkl or .pkl.gz
    outCSV:  Optional[str] = None        # MDEOut -> .csv
    logFile: Optional[str] = None        # log file

    # --- Logging / diagnostics ------------------------------------
    consoleOut: bool = True   # LogMsg() prints to console
    verbose:    bool = False
    debug:      bool = False
    plot:       bool = False
