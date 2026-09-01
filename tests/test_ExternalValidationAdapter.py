'''Unit tests for the pinned external-suite compatibility hook.'''
from types import SimpleNamespace

import pytest
from pandas.testing import assert_frame_equal
from pyEDM import sampleData

from ci import external_validation_adapter as adapter


#------------------------------------------------------------
def test_historical_mismatch_is_strict_and_assertion_only():
    class Item:
        def __init__( self, nodeID ):
            self.nodeid  = f'external/EDM_MDE_validation/{nodeID}'
            self.markers = []

        def add_marker( self, marker ):
            self.markers.append( marker )

    mismatches = [ Item(nodeID) for nodeID in
                   adapter._KNOWN_HISTORICAL_MISMATCHES ]
    clean = [ Item( nodeID ) for nodeID in [
        'test_SMap.py::test_smap4',
        'test_CCM.py::test_ccm5',
        'test_EDim.py::test_edim1',
        'test_EDim.py::test_edim6',
        'test_EDim.py::test_edim7',
        'test_Simplex.py::test_simplex1',
    ] ]
    items = mismatches + clean

    adapter.pytest_collection_modifyitems( items )

    for item in mismatches :
        assert len( item.markers ) == 1
        marker = item.markers[0]
        assert marker.name == 'xfail'
        assert marker.kwargs['strict'] is True
        assert marker.kwargs['raises'] is AssertionError
    for item in clean :
        assert item.markers == []


#------------------------------------------------------------
def test_shared_lorenz_sample_is_restored():
    baseline  = sampleData['Lorenz5D'].copy( deep = True )
    isolation = adapter._IsolateLorenzSampleData.__wrapped__()

    next( isolation )
    try :
        sampleData['Lorenz5D'].iloc[0, 1] = float('nan')
    finally :
        with pytest.raises( StopIteration ) :
            next( isolation )

    assert_frame_equal( sampleData['Lorenz5D'], baseline )


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
