'''Validation tests for dimx MDE'''
import pytest
import dimx as dx
from   pyEDM import sampleData

from conftest import MDEArgs, EvalArgs, ValidData, MDE_FlyData

#------------------------------------------------------------
def test_mde1():
    '''Lorenz5D : D=4'''
    data = sampleData["Lorenz5D"]
    kwargs = MDEArgs.copy()
    kwargs.update( dict(removeTime      = True,
                        removeColumns   = ['V5'],
                        D               = 4,
                        target          = 'V5',
                        tau             = -5,
                        exclusionRadius = 10,
                        crossMapRhoMin  = 0.2,
                        embedDimRhoMin  = 0.2,
                        firstEMax       = True) )

    mde = dx.MDE(data, **kwargs)
    mde.Run()

    df  = mde.MDEOut
    dfv = ValidData("MDE_Lorenz5D_1_Valid.csv")

    mdeOut = round(  df.iloc[:,1:], 3 )
    valid  = round( dfv.iloc[:,1:], 3 )
    assert mdeOut.equals( valid )

#------------------------------------------------------------
def test_mde2():
    '''Fly FWD'''
    data = MDE_FlyData()
    kwargs = MDEArgs.copy()
    kwargs.update( dict(removeColumns   = ['index','FWD','Left_Right'],
                        D               = 8,
                        target          = 'FWD',
                        lib             = [1,300],
                        pred            = [301,600],
                        crossMapRhoMin  = 0.2,
                        embedDimRhoMin  = 0.65,
                        ccmSlope        = 0.01,
                        ccmSeed         = 7777) )

    mde = dx.MDE(data, **kwargs)
    mde.Run()

    df  = mde.MDEOut
    dfv = ValidData("MDE_Fly_2_Valid.csv")

    mdeOut = round(  df.iloc[:,1:], 3 )
    valid  = round( dfv.iloc[:,1:], 3 )
    assert mdeOut.equals( valid )

#------------------------------------------------------------
def test_mde3():
    '''   '''
    data = MDE_FlyData()
    kwargs = MDEArgs.copy()
    kwargs.update( dict(removeColumns   = ['index','FWD','Left_Right'],
                        D               = 10,
                        target          = 'Left_Right',
                        lib             = [1,600],
                        pred            = [801,1000],
                        crossMapRhoMin  = 0.05,
                        embedDimRhoMin  = 0.2,
                        ccmSlope        = 0.01,
                        ccmSeed         = 7777) )

    mde = dx.MDE(data, **kwargs)
    mde.Run()

    df  = mde.MDEOut
    dfv = ValidData("MDE_Fly_3_Valid.csv")

    mdeOut = round(  df.iloc[:,1:], 3 )
    valid  = round( dfv.iloc[:,1:], 3 )
    assert mdeOut.equals( valid )

#------------------------------------------------------------
def test_evaluate4():
    '''   '''
    data = MDE_FlyData()
    kwargs = EvalArgs.copy()
    kwargs.update( dict(mde_columns   = ['TS33','TS4','TS8','TS9','TS32'],
                        columns_range = [1,81],
                        predictVar    = 'FWD',
                        library       = [1,300],
                        prediction    = [301,600],
                        Tp            = 1,
                        components    = 5,
                        dmap_k        = 15) )

    ev = dx.Evaluate(data, **kwargs)
    ev.Run()

    df  = ev.mde # MDE simplex predictions
    dfv = ValidData("Evaluate_Fly_FWD_4_Valid.csv")

    evOut = round(  df.iloc[:,1:], 5 )
    valid  = round( dfv.iloc[:,1:], 5 )
    assert evOut.equals( valid )
