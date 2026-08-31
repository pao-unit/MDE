'''Smoke tests for the public command-line parsers.'''
from dimx.CLI_Parser        import ParseCmdLine
from dimx.ReverseCLI_Parser import ParseReverseCmdLine


#------------------------------------------------------------
def test_mde_parser_maps_common_flags():
    args = ParseCmdLine( ['--dataFile', 'data.csv',
                          '--target', 'response',
                          '--D', '5',
                          '--lib', '1', '20',
                          '--pred', '21', '40',
                          '--noCCM',
                          '--noConsoleOut'] )

    assert args.dataFile == 'data.csv'
    assert args.target == 'response'
    assert args.D == 5
    assert args.lib == [1, 20]
    assert args.pred == [21, 40]
    assert args.noCCM is True
    assert args.consoleOut is False


#------------------------------------------------------------
def test_parser_list_defaults_are_fresh():
    first  = ParseCmdLine( [] )
    second = ParseCmdLine( [] )

    first.removeColumns.append( 'changed' )
    first.pLibSizes.append( 100 )

    assert second.removeColumns == []
    assert second.pLibSizes == [10, 15, 85, 90]


#------------------------------------------------------------
def test_reverse_parser_merges_reverse_and_mde_flags():
    args = ParseReverseCmdLine( ['--maxDepth', '2',
                                 '--logEveryPct', '25',
                                 '--reverseVariables', 'a', 'b',
                                 '--target', 'root',
                                 '--D', '4'] )

    assert args.maxDepth == 2
    assert args.logEveryPct == 25
    assert args.reverseVariables == ['a', 'b']
    assert args.target == 'root'
    assert args.D == 4
