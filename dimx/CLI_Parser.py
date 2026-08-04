# Python distribution modules
from argparse import ArgumentParser

# Local modules
from .Config import MDEConfig

#--------------------------------------------------------------
#--------------------------------------------------------------
def ParseCmdLine( argv = None ):

    # Defaults come from MDEConfig, not from literals restated here.
    # A fresh instance per call preserves the "fresh list per parse"
    # behaviour of the original literal defaults. libSizes has no CLI
    # flag: it is a programmatic-only parameter and stays at its
    # MDEConfig default on the CLI path.
    _cfg = MDEConfig()

    parser = ArgumentParser(
        description = 'Manifold Dimensional Expansion' )

    parser.add_argument('-d', '--dataFile',
                        dest = 'dataFile', type = str,
                        action = 'store',
                        default = _cfg.dataFile,
                        help = 'Input data file.')

    parser.add_argument('-dN', '--dataName',
                        dest = 'dataName', type = str,
                        action = 'store',
                        default = _cfg.dataName,
                        help = 'Input .npz data name.')

    parser.add_argument('-rT', '--removeTime',
                        dest = 'removeTime',
                        action = 'store_true',
                        default = _cfg.removeTime,
                        help = 'removeTime.')

    parser.add_argument('-nT', '--noTime',
                        dest = 'noTime',
                        action = 'store_true',
                        default = _cfg.noTime,
                        help = 'noTime.')

    parser.add_argument('-cn', '--columnNames', nargs = '*',
                        dest = 'columnNames', type = str,
                        action = 'store',
                        default = _cfg.columnNames,
                        help = 'Data column names (partial match).')

    parser.add_argument('-di', '--initDataColumns', nargs = '*',
                        dest = 'initDataColumns', type = str,
                        action = 'store',
                        default = _cfg.initDataColumns,
                        help = 'Initial .npy / npz column names.')

    parser.add_argument('-rc', '--removeColumns', nargs = '*',
                        dest = 'removeColumns', type = str,
                        action = 'store',
                        default = _cfg.removeColumns,
                        help = 'data columns to remove.')

    parser.add_argument('-D', '--D',
                        dest = 'D', type = int,
                        action = 'store',
                        default = _cfg.D,
                        help = 'MDE maximum dimension.')

    parser.add_argument('-t', '--target',
                        dest = 'target', type = str,
                        action = 'store',
                        default = _cfg.target,
                        help = 'Data target name.')

    parser.add_argument('-l', '--lib', nargs = '*',
                        dest = 'lib', type = int,
                        action = 'store',
                        default = _cfg.lib,
                        help = 'library indices.')

    parser.add_argument('-p', '--pred', nargs = '*',
                        dest = 'pred', type = int,
                        action = 'store',
                        default = _cfg.pred,
                        help = 'prediction indices.')

    parser.add_argument('-T', '--Tp',
                        dest = 'Tp', type = int,
                        action = 'store',
                        default = _cfg.Tp,
                        help = 'Tp.')

    parser.add_argument('-tau', '--tau',
                        dest = 'tau', type = int,
                        action = 'store',
                        default = _cfg.tau,
                        help = 'tau.')

    parser.add_argument('-xr', '--exclusionRadius',
                        dest = 'exclusionRadius', type = int,
                        action = 'store',
                        default = _cfg.exclusionRadius,
                        help = 'Exclusion Radius.')

    parser.add_argument('-s', '--sample',
                        dest = 'sample', type = int,
                        action = 'store',
                        default = _cfg.sample,
                        help = 'CCM sample.')

    parser.add_argument('-L', '--pLibSizes', nargs = '*',
                        dest = 'pLibSizes', type = int,
                        action = 'store',
                        default = _cfg.pLibSizes,
                        help = 'CCM pLibSizes.')

    parser.add_argument('-nC', '--noCCM',
                        dest = 'noCCM',
                        action = 'store_true',
                        default = _cfg.noCCM,
                        help = 'no CCM.')

    parser.add_argument('-ccs', '--ccmSlope',
                        dest = 'ccmSlope', type = float,
                        action = 'store',
                        default = _cfg.ccmSlope,
                        help = 'CCM slope threshold.')

    parser.add_argument('-seed', '--ccmSeed',
                        dest = 'ccmSeed', type = int,
                        action = 'store',
                        default = _cfg.ccmSeed,
                        help = 'CCM seed.')

    parser.add_argument('-smf', '--slopeMatrixFile',
                        dest = 'slopeMatrixFile', type = str,
                        action = 'store',
                        default = _cfg.slopeMatrixFile,
                        help = 'CCM slope matrix file (.csv or .feather) '
                               'as returned by pyEDM CCM_Matrix. Square, '
                               'with identical .index (source) and '
                               '.columns (predicted) labels. If provided '
                               '(and not noCCM), CCM slopes are read from '
                               'it and EmbedDimension/CCM are skipped.')

    parser.add_argument('-E', '--E',
                        dest = 'E', type = int,
                        action = 'store',
                        default = _cfg.E,
                        help = 'Takens embedding dimension.')

    parser.add_argument('-cmin', '--crossMapRhoMin',
                        dest = 'crossMapRhoMin', type = float,
                        action = 'store',
                        default = _cfg.crossMapRhoMin,
                        help = 'crossMapRhoMin threshold.')

    parser.add_argument('-emin', '--embedDimRhoMin',
                        dest = 'embedDimRhoMin', type = float,
                        action = 'store',
                        default = _cfg.embedDimRhoMin,
                        help = 'embedDimRhoMin threshold.')

    parser.add_argument('-mE', '--maxE',
                        dest = 'maxE', type = int,
                        action = 'store',
                        default = _cfg.maxE,
                        help = 'EmbeddingDimension maxE.')

    parser.add_argument('-fE', '--firstEMax',
                        dest = 'firstEMax',
                        action = 'store_true',
                        default = _cfg.firstEMax,
                        help = 'EmbeddingDimension firstEMax.')

    parser.add_argument('-tD', '--timeDelay',
                        dest = 'timeDelay', type = int,
                        action = 'store',
                        default = _cfg.timeDelay,
                        help = 'Number of time delays to add.')

    parser.add_argument('-C', '--crossMapCores',
                        dest = 'crossMapCores', type = int,
                        action = 'store',
                        default = _cfg.crossMapCores,
                        help = 'Cross-map core cap; all cores if unset, '
                               'otherwise upper bound on CrossMapColumns '
                               'sweep pool.')

    parser.add_argument('-mp', '--mpMethod',
                        dest = 'mpMethod', type = str,
                        action = 'store',
                        default = _cfg.mpMethod,
                        help = 'Multiprocessing start method')

    parser.add_argument('-cz', '--chunksize',
                        dest = 'chunksize', type = int,
                        action = 'store', default = _cfg.chunksize,
                        help = 'ProcessPool chunksize')

    parser.add_argument('-sM', '--sharedMem',
                        dest = 'sharedMem', type = float,
                        action = 'store', default = _cfg.sharedMem,
                        help = 'Shared-memory threshold in decimal MB '
                               '(1e6 bytes), compared to the DataFrame size; '
                               '0 forces initargs.')

    parser.add_argument('-lp', '--logPct',
                        dest = 'logPct', type = float,
                        action = 'store', default = _cfg.logPct,
                        help = 'Cross-map progress band width (percent); '
                               'requires verbose. 0 = silent, >=100 = single '
                               'completion message.')

    parser.add_argument('-kw', '--kdWorkers',
                        dest = 'kdWorkers', type = int,
                        action = 'store', default = _cfg.kdWorkers,
                        help = 'KDTree.query workers in sweep Simplex. '
                               '1 (default) since the pool parallelizes across '
                               'candidates; -1 uses all cores per query.')

    parser.add_argument('-mrd', '--maxLenRhoD',
                        dest = 'maxLenRhoD', type = int,
                        action = 'store', default = _cfg.maxLenRhoD,
                        help = 'Output() cap on number of rhoD entries stored '
                               'per dimension (ranked by decreasing rho). ')

    parser.add_argument('-mrc', '--maxLenRhoD_CCM',
                        dest = 'maxLenRhoD_CCM', type = int,
                        action = 'store', default = _cfg.maxLenRhoD_CCM,
                        help = 'Output() cap on number of rhoD_CCM entries '
                               'stored per dimension. rhoD_CCM is the subset '
                               'of rhoD passing CCM convergence, populated '
                               'when a slopeMatrix is supplied.')

    parser.add_argument('-od', '--outDir',
                        dest = 'outDir', type = str,
                        action = 'store',
                        default = _cfg.outDir,
                        help = 'Output directory.')

    parser.add_argument('-o', '--outFile',
                        dest = 'outFile', type = str,
                        action = 'store',
                        default = _cfg.outFile,
                        help = 'Output MDE class pickle file.')

    parser.add_argument('-oc', '--outCSV',
                        dest = 'outCSV', type = str,
                        action = 'store',
                        default = _cfg.outCSV,
                        help = 'Output csv file.')

    parser.add_argument('-lf', '--logFile',
                        dest = 'logFile', type = str,
                        action = 'store',
                        default = _cfg.logFile,
                        help = 'Output log file.')

    parser.add_argument('-n', '--noConsoleOut',
                        dest = 'consoleOut',
                        action = 'store_false',
                        default = _cfg.consoleOut,
                        help = 'Do not LogMsg to console.')

    parser.add_argument('-v', '--verbose',
                        dest = 'verbose',
                        action = 'store_true',
                        default = _cfg.verbose,
                        help = 'verbose.')

    parser.add_argument('-g', '--debug',
                        dest = 'debug',
                        action = 'store_true',
                        default = _cfg.debug,
                        help = 'debug.')

    parser.add_argument('-P', '--plot',
                        dest = 'plot',
                        action = 'store_true',
                        default = _cfg.plot,
                        help = 'plot.')

    args = parser.parse_args( argv ) # if argv is None : default = sys.argv[1:]

    return args
