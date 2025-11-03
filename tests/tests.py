# Python distribution modules
from os import remove
import sys
import unittest
from pickle import load

# Community modules
from pandas import read_csv, DataFrame

if not "../dimx" in sys.path :
    sys.path.append( "../dimx" ) # To load Evaluate app

from warnings import filterwarnings # py 3.13 multiprocessing multi-threaded
filterwarnings( "ignore", category = DeprecationWarning )

#----------------------------------------------------------------
# Suite of tests
#----------------------------------------------------------------
class test_MDE( unittest.TestCase ):
    '''NOTE: Bizarre default of unittest class presumes
             methods names to be run begin with "test_" 
    '''
    # JP How to pass command line arg to class? verbose = False
    def __init__( self, *args, **kwargs):
        super( test_MDE, self ).__init__( *args, **kwargs )

    #------------------------------------------------------------
    # 
    #------------------------------------------------------------
    @classmethod
    def setUpClass( self ):
        self.verbose = False
        # self.GetValidFiles() # Not needed yet

    #------------------------------------------------------------
    # 
    #------------------------------------------------------------
    @classmethod
    def tearDown( self ):
        '''
        '''
        try:
            remove( 'MDE_API_2.csv' )
            remove( 'MDE_API_2.pkl' )
        except :
            pass

    #------------------------------------------------------------
    # MDE class API 1
    #------------------------------------------------------------
    def test_MDE_API_1( self ):
        '''MDE class API 1'''
        if self.verbose : print ( " --- MDE API 1 ---" )

        from dimx import MDE

        df = read_csv( '../data/Fly80XY_norm_1061.csv' )

        mde = MDE( df, target = 'FWD',
                   removeColumns = ['index','FWD','Left_Right'],
                   D = 5, lib = [1,300], pred = [301,600] )
        mde.Run()

        self.assertTrue( isinstance( mde.MDEOut, DataFrame ) )
        self.assertTrue( 'variables' in mde.MDEOut.columns )
        self.assertTrue( 'rho'       in mde.MDEOut.columns )

    #------------------------------------------------------------
    # MDE class API 2
    #------------------------------------------------------------
    def test_MDE_API_2( self ):
        '''MDE class API 2'''
        if self.verbose : print ( " --- MDE API 2 ---" )

        from dimx import MDE

        df = read_csv( '../data/Fly80XY_norm_1061.csv' )

        mde = MDE( df, target = 'FWD',
                   removeColumns = ['index','FWD','Left_Right'],
                   D = 5, lib = [1,300], pred = [301,600],
                   embedDimRhoMin = 0.4, ccmSlope = 0.01, ccmSeed = 7777,
                   outCSV = 'MDE_API_2.csv', outFile = 'MDE_API_2.pkl' )
        mde.Run()

        # Validate outCSV equals MDEOut
        mdeOut_ = read_csv( 'MDE_API_2.csv' )
        self.assertTrue( mde.MDEOut.equals( mdeOut_ ) )

        # Validate the MDE pickle file
        with open('MDE_API_2.pkl','rb') as f:
            mde_ = load(f)

        self.assertTrue( mde.MDEOut.equals( mde_.MDEOut ) )

    #------------------------------------------------------------
    # Evaluate class API
    #------------------------------------------------------------
    def test_Evaluate_API_1( self ):
        '''Evaluate class API 1'''
        if self.verbose : print ( " --- Evaluate API 1 ---" )

        from Evaluate import Evaluate

        df = read_csv( '../data/Fly80XY_norm_1061.csv' )

        ev = Evaluate( df, mde_columns = ['TS33','TS4','TS8','TS9','TS32'],
                       columns_range = [1,81], predictVar = 'FWD',
                       library = [1,300], prediction = [301,600], Tp = 1,
                       components = 5, dmap_k = 40 )
        ev.Run()

        self.assertTrue( isinstance( ev.mde, DataFrame ) )
        self.assertTrue( 'Predictions' in ev.mde.columns )
        self.assertTrue( ev.mde.shape == (301,4) )

#------------------------------------------------------------
#
#------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()
