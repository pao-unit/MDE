
import sys
import unittest

from pandas import read_csv

# Monkey patch sys.path for MDE app
if not '../src' in sys.path :
    sys.path.insert( 0, '../src' )

#----------------------------------------------------------------
# Suite of tests
#----------------------------------------------------------------
class test_CC( unittest.TestCase ):
    '''NOTE: Bizarre default of unittest class presumes
             methods names to be run begin with "test_" 
    '''
    # JP How to pass command line arg to class? verbose = False
    def __init__( self, *args, **kwargs):
        super( test_CC, self ).__init__( *args, **kwargs )

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
    # API
    #------------------------------------------------------------
    def test_API_1( self ):
        '''API 1'''
        if self.verbose : print ( " --- API 1 ---" )

        from ManifoldDimExpand import ManifoldDimExpand

        df = read_csv( '../data/Fly80XY_norm_1061.csv' )

        MDE_df = ManifoldDimExpand( df, target = 'FWD', 
                                    removeColumns = ['index','FWD','Left_Right'],
                                    D = 5, lib = [1,300], pred = [301,600] )
        self.assertTrue( 'variables' in MDE_df.columns )
        self.assertTrue( 'rho'       in MDE_df.columns )

#------------------------------------------------------------
#
#------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()
