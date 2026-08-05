# Python distribution modules
from argparse import ArgumentParser, Namespace

# Local modules
from .CLI_Parser import ParseCmdLine

#--------------------------------------------------------------
#--------------------------------------------------------------
def ParseReverseCmdLine( argv = None ):
    '''Parse ReverseMDE arguments, delegating MDE arguments to
    CLI_Parser.ParseCmdLine so the ~40 MDE flags are defined once.

    Reverse-only flags are pulled off first with parse_known_args;
    the remaining argv ( all MDE flags ) is handed to ParseCmdLine.
    The two Namespaces are merged and returned as one. Every reverse
    flag's long name equals its Namespace dest equals its
    ReverseMDE.__init__ keyword ( maxDepth, MDEOut, reverseVariables,
    logProgress, logEvery, logEveryPct, quietChildren ), so
    ReverseMDE( args = args ) needs no per-field remapping.
    '''
    parser = ArgumentParser( add_help = False,
        description = 'Reverse Manifold Dimensional Expansion' )

    parser.add_argument('-md', '--maxDepth',
                        dest = 'maxDepth', type = int,
                        action = 'store',
                        default = None,
                        help = 'Levels below the root to expand; '
                               'unset => bounded only by the visited set.')

    parser.add_argument('-mo', '--MDEOut',
                        dest = 'MDEOut', type = str,
                        action = 'store',
                        default = None,
                        help = 'Path to a pre-existing MDEOut (.csv or '
                               '.feather) whose drivers seed the root; '
                               'the root is not re-run.')

    parser.add_argument('-rv', '--reverseVariables', nargs = '*',
                        dest = 'reverseVariables', type = str,
                        action = 'store',
                        default = None,
                        help = 'Explicit list of root driver variables '
                               '(list form of the supplied-root mode).')

    parser.add_argument('-pg', '--logProgress',
                        dest = 'logProgress',
                        action = 'store_true',
                        default = False,
                        help = 'Log a combined count + ETA progress line '
                               'per expansion.')

    parser.add_argument('-le', '--logEvery',
                        dest = 'logEvery', type = int,
                        action = 'store',
                        default = 1,
                        help = 'Emit a progress line every n completed '
                               'expansions ( default 1 ).')

    parser.add_argument('-lp', '--logEveryPct',
                        dest = 'logEveryPct', type = float,
                        action = 'store',
                        default = None,
                        help = 'Emit a progress line every p percent of N '
                               '( wins over logEvery when set ); float, so '
                               '0.5 suits very large N.')

    parser.add_argument('-qc', '--quietChildren',
                        dest = 'quietChildren',
                        action = 'store_true',
                        default = False,
                        help = 'Derive child MDE runs with verbose = False '
                               'so only reverse-level messages appear.')

    reverseArgs, mdeArgv = parser.parse_known_args( argv )

    # Delegate the remaining argv to the MDE parser ( defaults still
    # sourced from a fresh MDEConfig, CLI_Parser untouched ).
    mdeArgs = ParseCmdLine( argv = mdeArgv )

    # Merge : MDE fields plus the four reverse fields, one Namespace.
    return Namespace( **{ **vars( mdeArgs ), **vars( reverseArgs ) } )
