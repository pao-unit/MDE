'''Fast unit tests for MDE range resolution and validation.'''
import pytest
from pandas import DataFrame

from dimx.MDE import MDE


#------------------------------------------------------------
def _Data( N = 100 ):
    values = [ float(i) for i in range( N ) ]
    return DataFrame( { 'Time'   : values,
                        'target' : values,
                        'driver' : values } )


#------------------------------------------------------------
@pytest.mark.parametrize( 'N, expectedLib, expectedPred', [
    ( 10, [1, 5], [6, 10] ),
    (  9, [1, 4], [5,  9] ),
] )
def test_resolve_lib_pred( N, expectedLib, expectedPred ):
    lib, pred = MDE.ResolveLibPred( N )

    assert lib == expectedLib
    assert pred == expectedPred


#------------------------------------------------------------
@pytest.mark.parametrize( 'lib, pred, expectedLib, expectedPred', [
    ( [],      [70, 90], [1, 50], [70, 90]  ),
    ( [5, 40], [],       [5, 40], [51, 100] ),
] )
def test_validate_resolves_only_empty_range( lib, pred,
                                             expectedLib, expectedPred ):
    mde = MDE( _Data(), target = 'target', lib = lib, pred = pred,
               consoleOut = False )

    mde.Validate()

    assert mde.args.lib == expectedLib
    assert mde.args.pred == expectedPred


#------------------------------------------------------------
def test_validate_requires_target():
    mde = MDE( _Data(), consoleOut = False )

    with pytest.raises( RuntimeError, match = 'target required' ) :
        mde.Validate()


#------------------------------------------------------------
def test_validate_requires_data_source():
    mde = MDE( target = 'target', consoleOut = False )

    with pytest.raises( RuntimeError, match = 'dataFrame or dataFile required' ) :
        mde.Validate()


#------------------------------------------------------------
@pytest.mark.parametrize( 'name, value', [
    ( 'removeColumns',   'driver' ),
    ( 'columnNames',     'driver' ),
    ( 'initDataColumns', 'Time'   ),
    ( 'lib',             (1, 50)  ),
    ( 'pred',            (51, 100) ),
] )
def test_validate_requires_list_parameters( name, value ):
    kwargs = { 'target' : 'target', name : value, 'consoleOut' : False }
    mde = MDE( _Data(), **kwargs )

    with pytest.raises( RuntimeError, match = f'{name} must be list' ) :
        mde.Validate()


#------------------------------------------------------------
def test_validate_requires_slope_matrix_dataframe():
    mde = MDE( _Data(), slopeMatrix = [[1.0]], target = 'target',
               consoleOut = False )

    with pytest.raises( RuntimeError, match = 'pandas DataFrame' ) :
        mde.Validate()


#------------------------------------------------------------
def test_validate_requires_matching_slope_matrix_labels():
    slopeMatrix = DataFrame( [[1.0, 0.0], [0.0, 1.0]],
                             index   = ['target', 'driver'],
                             columns = ['driver', 'target'] )
    mde = MDE( _Data(), slopeMatrix = slopeMatrix, target = 'target',
               consoleOut = False )

    with pytest.raises( RuntimeError, match = 'index and .columns' ) :
        mde.Validate()


#------------------------------------------------------------
def test_validate_requires_target_in_slope_matrix():
    labels = ['driver', 'other']
    slopeMatrix = DataFrame( [[1.0, 0.0], [0.0, 1.0]],
                             index = labels, columns = labels )
    mde = MDE( _Data(), slopeMatrix = slopeMatrix, target = 'target',
               consoleOut = False )

    with pytest.raises( RuntimeError, match = 'does not contain target' ) :
        mde.Validate()


#------------------------------------------------------------
def test_run_rejects_slope_matrix_missing_candidate():
    data = _Data().assign( uncovered = 1.0 )
    labels = ['target', 'driver']
    slopeMatrix = DataFrame( [[1.0, 0.0], [0.0, 1.0]],
                             index = labels, columns = labels )
    mde = MDE( data, slopeMatrix = slopeMatrix, target = 'target',
               removeColumns = ['target'], consoleOut = False )

    with pytest.raises( RuntimeError,
                        match = 'slope matrix missing candidate columns' ) :
        mde.Run()
