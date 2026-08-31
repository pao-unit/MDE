'''Unit tests for MDE configuration resolution.'''
from dataclasses import fields
from types       import SimpleNamespace

import pytest
from pandas import DataFrame

from dimx.Config import MDEConfig
from dimx.MDE    import MDE


#------------------------------------------------------------
def test_list_defaults_are_independent():
    '''Every mutable MDEConfig default belongs to its own instance.'''
    first  = MDEConfig()
    second = MDEConfig()

    listFields = [ field.name for field in fields( MDEConfig )
                   if isinstance( getattr( first, field.name ), list ) ]

    for name in listFields :
        firstList  = getattr( first, name )
        secondList = getattr( second, name )
        assert firstList is not secondList
        firstList.append( 'changed' )
        assert 'changed' not in secondList


#------------------------------------------------------------
def test_config_override_does_not_mutate_source():
    '''Keyword overrides take precedence over an explicit config.'''
    source = MDEConfig( target = 'source', D = 4,
                        removeColumns = ['structural'] )
    mde = MDE( DataFrame(), config = source, target = 'override', D = 7 )

    assert mde.args.target == 'override'
    assert mde.args.D == 7
    assert mde.args.removeColumns == ['structural']
    assert source.target == 'source'
    assert source.D == 4


#------------------------------------------------------------
def test_cli_args_take_constructor_precedence():
    '''The CLI Namespace wins and unknown Namespace keys are ignored.'''
    source = MDEConfig( target = 'config', D = 4 )
    args   = SimpleNamespace( target = 'cli', D = 6,
                              reverseOnly = 'ignored' )

    mde = MDE( DataFrame(), config = source, args = args,
               target = 'keyword', D = 8 )

    assert mde.args.target == 'cli'
    assert mde.args.D == 6
    assert not hasattr( mde.args, 'reverseOnly' )
    assert mde.args.removeColumns == []


#------------------------------------------------------------
@pytest.mark.parametrize( 'kwargs', [
    { 'notAnMDEParameter' : True },
    { 'config' : MDEConfig(), 'notAnMDEParameter' : True },
] )
def test_unknown_constructor_keyword_fails_fast( kwargs ):
    with pytest.raises( TypeError, match = 'notAnMDEParameter' ) :
        MDE( DataFrame(), **kwargs )
