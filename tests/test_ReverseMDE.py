'''Fast orchestration tests for ReverseMDE.'''
import gzip
from pickle import load

from pandas import DataFrame, read_csv
from pandas.testing import assert_frame_equal

from dimx.Config     import MDEConfig
from dimx.ReverseMDE import ReverseMDE


#------------------------------------------------------------
def _Data( *columns ):
    values = [ float(i) for i in range( 10 ) ]
    return DataFrame( { column : values for column in columns } )


#------------------------------------------------------------
def _Config( **overrides ):
    kwargs = dict( target = 'root', lib = [1, 5], pred = [6, 10],
                   consoleOut = False )
    kwargs.update( overrides )
    return MDEConfig( **kwargs )


#------------------------------------------------------------
def _MDEOut( variables ):
    return DataFrame( { 'variables' : list( variables ) } )


#------------------------------------------------------------
def _StubGraph( monkeypatch, graph, calls = None ):
    def RunMDE( self, task ):
        if calls is not None :
            calls.append( task.target )
        return _MDEOut( graph.get( task.target, [] ) )

    monkeypatch.setattr( ReverseMDE, '_RunMDE', RunMDE )


#------------------------------------------------------------
def test_run_walks_graph_breadth_first( monkeypatch ):
    graph = { 'root' : ['a', 'b'],
              'a'    : ['c'],
              'b'    : ['d'],
              'c'    : [],
              'd'    : [] }
    _StubGraph( monkeypatch, graph )
    reverse = ReverseMDE( _Data( 'root', 'a', 'b', 'c', 'd' ),
                          config = _Config() )

    result = reverse.Run()

    assert result is None
    assert reverse.Order == ['root', 'a', 'b', 'c', 'd']
    assert list( reverse.GraphOut ) == reverse.Order
    assert reverse.Visited == set( reverse.Order )


#------------------------------------------------------------
def test_run_expands_each_target_once_when_graph_cycles( monkeypatch ):
    graph = { 'root' : ['a'],
              'a'    : ['root', 'b'],
              'b'    : ['a'] }
    calls = []
    _StubGraph( monkeypatch, graph, calls )
    reverse = ReverseMDE( _Data( 'root', 'a', 'b' ), config = _Config() )

    reverse.Run()

    assert calls == ['root', 'a', 'b']
    assert reverse.Order == calls


#------------------------------------------------------------
def test_max_depth_bounds_expansion( monkeypatch ):
    graph = { 'root' : ['a', 'b'],
              'a'    : ['c'],
              'b'    : ['d'] }
    calls = []
    _StubGraph( monkeypatch, graph, calls )
    reverse = ReverseMDE( _Data( 'root', 'a', 'b', 'c', 'd' ),
                          config = _Config(), maxDepth = 1 )

    reverse.Run()

    assert calls == ['root', 'a', 'b']
    assert reverse.Order == calls
    assert set( reverse.GraphOut ) == {'root', 'a', 'b'}


#------------------------------------------------------------
def test_supplied_root_seeds_children_without_recomputing_root( monkeypatch ):
    calls = []
    _StubGraph( monkeypatch, {}, calls )
    reverse = ReverseMDE( _Data( 'root', 'a', 'b' ), config = _Config(),
                          reverseVariables = ['a', 'root', 'b'] )

    reverse.Run()

    assert calls == ['a', 'b']
    assert reverse.Order == ['root', 'a', 'b']
    assert list( reverse.GraphOut['root']['variables'] ) == \
           ['a', 'root', 'b']


#------------------------------------------------------------
def test_empty_ranges_resolve_without_mutating_supplied_config():
    source = MDEConfig( target = 'root', consoleOut = False )

    reverse = ReverseMDE( _Data( 'root' ), config = source,
                          reverseVariables = [] )

    assert source.lib == []
    assert source.pred == []
    assert reverse.baseConfig.lib == [1, 5]
    assert reverse.baseConfig.pred == [6, 10]


#------------------------------------------------------------
def test_per_run_config_preserves_root_and_quiets_children( monkeypatch ):
    captured = {}
    graph = { 'root' : ['child'], 'child' : [] }

    def RunMDE( self, task ):
        captured[task.target] = self._ConfigFor( task )
        return _MDEOut( graph[task.target] )

    monkeypatch.setattr( ReverseMDE, '_RunMDE', RunMDE )
    source = _Config( removeColumns = ['structural'], plot = True,
                      verbose = True )
    reverse = ReverseMDE( _Data( 'root', 'child' ), config = source,
                          quietChildren = True )

    reverse.Run()

    root  = captured['root']
    child = captured['child']
    assert root.target == 'root'
    assert root.removeColumns == ['structural']
    assert root.plot is True
    assert child.target == 'child'
    assert child.removeColumns == ['structural', 'child']
    assert child.plot is False
    assert child.verbose is False
    assert root.outFile is None and child.outFile is None
    assert root.outCSV is None and child.outCSV is None
    assert source.removeColumns == ['structural']


#------------------------------------------------------------
def test_output_writes_per_target_csv_and_compressed_graph( monkeypatch,
                                                            tmp_path ):
    _StubGraph( monkeypatch, { 'child' : [] } )
    config = _Config( outDir = str( tmp_path ), outCSV = 'graph.csv',
                      outFile = 'graph.pkl.gz' )
    reverse = ReverseMDE( _Data( 'root', 'child' ), config = config,
                          reverseVariables = ['child'] )

    reverse.Run()

    rootCSV  = tmp_path / 'graph_root.csv'
    childCSV = tmp_path / 'graph_child.csv'
    graphPKL = tmp_path / 'graph.pkl.gz'
    assert rootCSV.is_file()
    assert childCSV.is_file()
    assert graphPKL.is_file()
    assert_frame_equal( read_csv( rootCSV ), reverse.GraphOut['root'] )
    with gzip.open( graphPKL, 'rb' ) as file :
        restored = load( file )
    assert list( restored ) == ['root', 'child']
    for target in restored :
        assert_frame_equal( restored[target], reverse.GraphOut[target] )
