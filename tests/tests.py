# Python distribution modules
import sys
import unittest

# Community modules
from pandas import read_csv

# Monkey patch sys.path for MDE app
if not '../src' in sys.path :
    sys.path.insert( 0, '../src' )

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
        pass

    #------------------------------------------------------------
    # 
    #------------------------------------------------------------
    def GetValidFiles( self ):
        '''Create dictionary of DataFrame values from file name keys'''
        self.ValidFiles = {}

        validFiles = [ '1_valid.csv',
                       '2_valid.csv',
                       '3_valid.csv' ]

        # Create map of module validFiles pathnames in validFiles
        for file in validFiles:
            filename = "validation/" + file
            self.ValidFiles[ file ] = read_csv( filename )

    #------------------------------------------------------------
    # MDE class API
    #------------------------------------------------------------
    def test_MDE_API_1( self ):
        '''MDE class API 1'''
        if self.verbose : print ( " --- MDE API 1 ---" )

        from MDE import MDE

        df = read_csv( '../data/Fly80XY_norm_1061.csv' )

        mde = MDE( df, target = 'FWD', 
                   removeColumns = ['index','FWD','Left_Right'],
                   D = 5, lib = [1,300], pred = [301,600] )
        mde.Run()

        self.assertTrue( 'variables' in mde.MDEOut.columns )
        self.assertTrue( 'rho'       in mde.MDEOut.columns )

    #------------------------------------------------------------
    # Evaluate class API
    #------------------------------------------------------------
    def test_API_2( self ):
        '''Evaluate class API 1'''
        if self.verbose : print ( " --- Evaluate API 1 ---" )

        from Evaluate import Evaluate

        df = read_csv( '../data/Fly80XY_norm_1061.csv' )

        ev = Evaluate( df, mde_columns = ['TS33','TS4','TS8','TS9','TS32'],
                       columns_range = [1,81], predictVar = 'FWD',
                       library = [1,300], prediction = [301,600], Tp = 1,
                       components = 5, dmap_k = 40 )
        ev.Run()

        self.assertTrue( 'Predictions' in ev.mde.columns )
        self.assertTrue( ev.mde.shape == (301,4) )

#------------------------------------------------------------
#
#------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()
