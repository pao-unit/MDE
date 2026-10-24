# Community modules
import numpy as np
from matplotlib  import pyplot as plt
from scipy.stats import gaussian_kde

#-------------------------------------------------------------------
def Plot( self, title = '', table_xy = (0.6, 0.85),
          maxTable = None, fontsize = 12, figsize = (6,5) ):
    '''Plot an MDEOut DataFrame from MDE.Run()'''
    df = self.MDEOut.copy()
    D  = [d+1 for d in range(df.shape[0])]
    df.insert(0,"D",D)

    if maxTable is None:
        maxTable = df.shape[0]

    df_string = df.iloc[:maxTable,:].round(3).to_string(index=False)

    ax = df.plot( 'D', 'rho', lw = 4, title = title, figsize = figsize )
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)
    ax.set_ylabel('MDE ρ', fontsize = fontsize)
    ax.annotate( df_string, 
                 xy = table_xy, xycoords = 'axes fraction',
                 annotation_clip = False, fontsize = 11,
                 verticalalignment = 'top', wrap = True,
                 fontproperties = 'monospace' )
    plt.show()

#-------------------------------------------------------------------
def PlotRhoD( self,
              outputFile     = None,
              rhoType        = 'rhoD',
              rho            = False,
              bwMethod       = 1,
              dims           = None,
              Title          = None,
              lineWidth      = 3,
              figsize        = ( 6, 4.5 ),
              fontSize       = 12,
              legendFontSize = 11,
              xLabel         = None,
              yLabel         = None,
              labelFontSize  = 12,
              xlimit         = None,
              ylimit         = None,
              marker         = True,
              markerSize     = 12,
              full           = False,
              show           = True ):
    '''Plot distributions of MDE candidate rho at each dimension d.

    Data from MDE.rhoD or MDE.rhoD_CCM (selected by rhoType) : dict of
    { d : DataFrame } where key d is the dimension index from d = 1...D,
    values are DataFrame with index the list of MDE ID, column 'rho' the
    ranked simplex cross map correlation. MDE.rhoD_CCM is a subset of
    MDE.rhoD with items that passed CCM convergence.

    All dimensions are overlaid on one figure. The default is a kde pdf
    estimate of rho at each d. With rho = True the ranked rho values are
    plotted as stored: x = rank, y = rho.

    In the kde plot, if marker is True, a closed circle on the x axis
    marks the rho reported in MDE.MDEOut at each d, in the color of the
    kde trace. The kde is fit to the candidate vectors; the marker is the
    realized MDE value at that dimension.

    Each kde trace is truncated on the upper tail at the MDE.MDEOut rho
    of its dimension. With full = True the entire kde fit is plotted.

    xlimit, ylimit are ( min, max ) axis limits. The default xlimit of
    the kde plot is ( max( 0, lowest x ), min( 1, highest x ) ) of the
    plotted kde traces and markers, plus a small margin; otherwise
    (ranked plot x = rank, ylimit) the axis autoscales.

    The figure is saved to outputFile if given, and displayed if show.
    '''

    # MDE.rhoD or MDE.rhoD_CCM : { d : DataFrame }
    rhoDict = _SelectDims( getattr( self, rhoType ), dims )

    fig, ax = plt.subplots( figsize = tuple( figsize ) )

    if rho :
        _PlotRanked( rhoDict, ax, lineWidth )
    else :
        _PlotKDE( rhoDict, ax, self.MDEOut, bwMethod, lineWidth,
                  marker, markerSize, full )

    _Format( ax, rho, rhoType, Title, xLabel, yLabel,
             fontSize, labelFontSize, legendFontSize, xlimit, ylimit )

    _Output( fig, outputFile, show )

#-------------------------------------------------------------------
def _SelectDims( rhoDict, dims = None ):
    '''Return a new { d : DataFrame } ordered by d.

       Empty dimensions are already absent: rhoD_to_DF() and
       rhoD_CCM_to_DF() skip them. If dims is not None only those keys
       are kept.
    '''

    # Sort so that the color cycle runs from low to high dimension
    selected = { d : rhoDict[ d ] for d in sorted( rhoDict ) }

    if dims is not None :
        # Report requested dimensions that are not available
        missing = [ d for d in dims if d not in selected ]
        if missing :
            print( f'PlotRhoD: dimensions not found : {missing}' )
        selected = { d : df for d, df in selected.items() if d in dims }

    if not selected :
        raise RuntimeError( 'PlotRhoD: no DataFrame to plot' )

    return selected

#-------------------------------------------------------------------
def _PlotRanked( rhoDict, ax, lineWidth ):
    '''Plot rho as stored (ranked) vs rank, one line per dimension.

       Colors are explicit, C0, C1... in order of plotting, the same as
       the matplotlib default cycle. No MDE rho marker is drawn.
    '''

    # Color index advances only for a trace actually drawn
    nPlotted = 0

    for d, df in rhoDict.items() :
        values = df['rho'].dropna().to_numpy()
        # Ranks start at 1
        ranks  = np.arange( 1, len( values ) + 1 )
        ax.plot( ranks, values, color = f'C{nPlotted}', lw = lineWidth,
                 label = str( d ) )
        nPlotted += 1

#-------------------------------------------------------------------
def _PlotKDE( rhoDict, ax, mdeOut, bwMethod, lineWidth,
              marker = True, markerSize = 8, full = False ):
    '''Plot gaussian kde pdf estimate of rho, one line per dimension.

       The upper tail is truncated at the MDE.MDEOut rho of dimension d
       (row d - 1 of mdeOut) unless full : then the entire fit is drawn.

       If marker : a closed circle on the x axis at that MDE rho, in the
       color of the kde trace.
    '''

    # Color index advances only for a trace actually drawn, so a skipped
    # dimension leaves no gap in the color sequence
    nPlotted = 0

    for d, df in rhoDict.items() :
        values = df['rho'].dropna().to_numpy()

        # gaussian_kde fails on a singular covariance: need spread
        if len( values ) < 2 or values.std() == 0 :
            print( f'PlotRhoD: d = {d} skipped, kde needs >= 2 '
                   f'distinct values' )
            continue

        # MDEOut has one row per dimension, row 0 is d = 1. Key by d, not
        # by position: dims may select a subset of dimensions.
        mdeRho = None
        if d <= len( mdeOut ) :
            mdeRho = mdeOut['rho'].iloc[ d - 1 ]
        else :
            print( f'PlotRhoD: d = {d} has no MDEOut row, '
                   f'no truncation, no marker' )

        # Fit to all candidates, evaluate on a grid padded half a range
        # either side of the data so the tails are visible
        kde = gaussian_kde( values, bw_method = bwMethod )
        pad = 0.5 * ( values.max() - values.min() )
        x   = np.linspace( values.min() - pad, values.max() + pad, 1000 )

        # Truncate after the fit so the curve below the cut is unchanged
        # (not renormalized). Only a cut inside the grid counts: if the
        # MDE rho lies beyond the grid there is no tail to drop.
        truncated = ( mdeRho is not None and not full
                      and mdeRho < x[-1] )
        if truncated :
            # End the curve exactly at mdeRho, not at the last grid
            # point below it, so the vertical line meets the curve
            x = np.append( x[ x < mdeRho ], mdeRho )

        y     = kde( x )
        color = f'C{nPlotted}'
        ax.plot( x, y, color = color, lw = lineWidth, label = str( d ) )
        nPlotted += 1

        # Drop from the end of the truncated curve to the x axis
        if truncated :
            ax.plot( [ mdeRho, mdeRho ], [ 0, y[-1] ], color = color,
                     lw = lineWidth, label = '_nolegend_' )

        if marker and mdeRho is not None :
            # clip_on = False keeps the circle whole where it straddles
            # the axis frame at y = 0
            ax.plot( [ mdeRho ], [ 0 ], 'o', color = color,
                     ms = markerSize, clip_on = False, zorder = 3,
                     label = '_nolegend_' )

#-------------------------------------------------------------------
def _Format( ax, rho, rhoType, Title, xLabel, yLabel,
             fontSize, labelFontSize, legendFontSize,
             xlimit = None, ylimit = None ):
    '''Apply labels, font sizes, legend and title to ax.'''

    # Axis label defaults depend on the plot type
    if xLabel is None :
        xLabel = 'Rank'      if rho else 'MDE ρ'
    if yLabel is None :
        yLabel = 'MDE ρ'     if rho else 'Density'

    # kde x is rho : the kde tail can run outside [ 0, 1 ], so clip to it,
    # but never wider than the plotted data: max( 0, lowest x ) and
    # min( 1, highest x ), plus a small margin. Lines on ax include the
    # MDE rho markers so they stay in view. Ranked x is rank: no default.
    if xlimit is None and not rho :
        xData  = [ x for line in ax.lines for x in line.get_xdata() ]
        margin = 0.02
        xlimit = ( max( 0, min( xData ) ) - margin,
                   min( 1, max( xData ) ) + margin )

    title = Title if Title is not None else rhoType

    ax.set_xlabel( xLabel, fontsize = labelFontSize )
    ax.set_ylabel( yLabel, fontsize = labelFontSize )
    ax.tick_params( axis = 'both', labelsize = fontSize )
    ax.set_title( title )
    if xlimit is not None :
        ax.set_xlim( tuple( xlimit ) )
    if ylimit is not None :
        ax.set_ylim( tuple( ylimit ) )
    # Ranked rho decays from upper left to lower right, so the lower right
    # is the empty corner; kde peaks sit at high rho so upper left is clear
    legendLoc = 'lower right' if rho else 'upper left'
    ax.legend( title = 'D', fontsize = legendFontSize,
               title_fontsize = legendFontSize, loc = legendLoc )
    ax.figure.tight_layout()

#-------------------------------------------------------------------
def _Output( fig, outputFile = None, show = True ):
    '''Save fig to outputFile if given; display it if show.'''

    if outputFile :
        fig.savefig( outputFile )

    if show :
        plt.show()
