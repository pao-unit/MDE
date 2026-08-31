'''Unit tests for the pinned external-suite compatibility hook.'''
from types import SimpleNamespace

from ci import external_validation_adapter as adapter


#------------------------------------------------------------
def test_known_golden_drift_is_strict_and_assertion_only():
    class Item:
        def __init__( self, nodeID ):
            self.nodeid  = f'external/EDM_MDE_validation/{nodeID}'
            self.markers = []

        def add_marker( self, marker ):
            self.markers.append( marker )

    items = [ Item(nodeID) for nodeID in adapter._KNOWN_GOLDEN_DRIFT ]
    clean = Item( 'test_Simplex.py::test_simplex1' )
    items.append( clean )

    adapter.pytest_collection_modifyitems( items )

    for item in items[:-1] :
        assert len( item.markers ) == 1
        marker = item.markers[0]
        assert marker.name == 'xfail'
        assert marker.kwargs['strict'] is True
        assert marker.kwargs['raises'] is AssertionError
    assert clean.markers == []


#------------------------------------------------------------
def test_legacy_mde_keywords_are_translated():
    module = SimpleNamespace( MDEArgs = { 'cores' : 5,
                                         'title' : 'legacy',
                                         'D' : 4 } )
    item = SimpleNamespace( path = SimpleNamespace( name = 'test_MDE.py' ),
                            module = module )

    adapter.pytest_runtest_setup( item )

    assert module.MDEArgs == { 'crossMapCores' : 5, 'D' : 4 }


#------------------------------------------------------------
def test_non_mde_module_is_not_adapted():
    args = { 'cores' : 5, 'title' : 'legacy' }
    item = SimpleNamespace( path = SimpleNamespace( name = 'test_CCM.py' ),
                            module = SimpleNamespace( MDEArgs = args ) )

    adapter.pytest_runtest_setup( item )

    assert args == { 'cores' : 5, 'title' : 'legacy' }
