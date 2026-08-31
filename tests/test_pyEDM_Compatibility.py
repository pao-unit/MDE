'''Fast contract tests for the pyEDM interfaces used by dimx.'''
from inspect import signature

from pyEDM import CCM, EmbedDimension, Simplex


#------------------------------------------------------------
def test_simplex_supports_kd_workers():
    '''CrossMapPool workers pass kdWorkers to pyEDM.Simplex().'''
    assert 'kdWorkers' in signature( Simplex ).parameters


#------------------------------------------------------------
def test_embed_dimension_supports_parallel_controls():
    '''MDE.Run() bounds EmbedDimension processes and KDTree workers.'''
    parameters = signature( EmbedDimension ).parameters

    assert 'mpMethod' in parameters
    assert 'numProcess' in parameters
    assert 'kdWorkers' in parameters


#------------------------------------------------------------
def test_ccm_supports_multiprocessing_method():
    '''MDE.Run() propagates its resolved multiprocessing method to CCM.'''
    assert 'mpMethod' in signature( CCM ).parameters
