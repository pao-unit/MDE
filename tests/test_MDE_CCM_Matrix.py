'''Validation tests for dimx MDE with CCM slopeMatrix'''
import pytest
import dimx as dx
from   pandas.testing import assert_frame_equal

from conftest import MDEArgs, EvalArgs, ValidData, LoadData

#====================================================================
def test_MDE_CCM_Matrix():
    '''Compare MDE with internal CCM and CCM_Matrix on Fly 20 subset
       test_CCM_Matrix.py finds small CCM Matrix differences due to
       different handling of seed'''

    data        = LoadData('Fly20_norm_1061.csv')
    slopeMatrix = ValidData('Fly20_Slope_Matrix_E7_Tp1_smp20_seed777.feather')

    # CCM_Matrix parameters used to create slopeMatrix
    L      = [100,150,950,1000]
    E      = 7
    Tp     = 1
    tau    = -1
    sample = 20
    seed   = 7777

    # MDE with internal EmbedDimension -> CCM
    mde = dx.MDE( data,
                  slopeMatrix    = None,
                  removeColumns  = ['index','FWD','Left_Right'],
                  D              = 8,
                  target         = 'FWD',
                  lib            = [1,300],
                  pred           = [301,600],
                  Tp             = Tp,
                  tau            = tau,
                  sample         = sample,
                  libSizes       = L,
                  ccmSlope       = 0.01,
                  ccmSeed        = seed,
                  crossMapRhoMin = 0.2,
                  embedDimRhoMin = 0.2,
                  sharedMem      = 0.0001 )

    mde.Run()

    # MDE with precomputed CCM slope
    mdeSlope = dx.MDE( data,
                       slopeMatrix    = slopeMatrix,
                       removeColumns  = ['index','FWD','Left_Right'],
                       D              = 8,
                       target         = 'FWD',
                       lib            = [1,300],
                       pred           = [301,600],
                       Tp             = Tp,
                       tau            = tau,
                       ccmSlope       = 0.01,
                       crossMapRhoMin = 0.2,
                       sharedMem      = 0.0001 )
    mdeSlope.Run()

    # Compare first 3 dimensions
    assert_frame_equal( mde.MDEOut.iloc[:2,:],
                        mdeSlope.MDEOut.iloc[:2,:], rtol = 0, atol = 1E-5 )
