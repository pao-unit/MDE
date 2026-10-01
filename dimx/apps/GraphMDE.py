#! /usr/bin/env python3
"""GraphMDE: build a networkx DiGraph from a dict of pandas DataFrames."""

import os
import sys
import json
import pickle
import argparse

from datetime import datetime

import numpy    as np
import pandas   as pd

# networkx is an optional extra; fail with the install hint
try:
    import networkx as nx
except ImportError as err:
    raise ImportError('GraphMDE requires networkx: '
                      'pip install dimx[graph]') from err


class GraphMDE:
    """Directed-graph builder over a dict of node DataFrames.

    Keys are node names. Each value's `variables` column lists nodes
    that connect *to* that key, so every used row yields an edge
    variable -> key. The first key is the network root (a sink).
    """

    varColumn  = 'variables'
    outFormats = ('.json', '.pkl')

    def __init__(self, nodeFrames, N=None, outFile=None,
                 verbose=False, logN=None, logPercent=None):
        # nodeFrames must be a non-empty dict (keys are node names)
        if not isinstance(nodeFrames, dict) or not nodeFrames:
            raise ValueError('nodeFrames must be a non-empty dict')

        # N override: None means all rows; else non-neg int / numpy int
        if N is not None:
            if isinstance(N, bool) or \
               not isinstance(N, (int, np.integer)):
                raise TypeError('N must be an int or numpy int')
            if N < 0:
                raise ValueError('N must be >= 0')

        # Logging cadence flags: validate shape/range up front
        if logN is not None:
            if isinstance(logN, bool) or \
               not isinstance(logN, (int, np.integer)):
                raise TypeError('logN must be an int or numpy int')
            if logN < 1:
                raise ValueError('logN must be >= 1')

        if logPercent is not None:
            if isinstance(logPercent, bool) or \
               not isinstance(logPercent, (int, np.integer)):
                raise TypeError('logPercent must be an int or numpy int')
            if logPercent < 1 or logPercent > 100:
                raise ValueError('logPercent must be in 1..100')

        # Root is first key; eagerly validate it (non-root frames
        # are trusted until BuildGraph accesses them)
        rootNode  = next(iter(nodeFrames))
        rootFrame = nodeFrames[rootNode]
        if not isinstance(rootFrame, pd.DataFrame):
            raise ValueError('root node has no DataFrame')
        if self.varColumn not in rootFrame.columns:
            raise ValueError(
                "root DataFrame lacks '%s' column" % self.varColumn
            )
        # Empty root would silently resolve N=None to 0 -> disallow
        if rootFrame.empty:
            raise ValueError('root DataFrame is empty')

        # Resolve N=None to the row count (root non-empty guarantees >=1)
        if N is None:
            N = len(rootFrame)

        # Output format is selected later by extension; reject it now
        if outFile is not None:
            ext = os.path.splitext(outFile)[1].lower()
            if ext not in self.outFormats:
                raise ValueError('outFile must end in .json or .pkl')

        # Collapse cadence to one stride: logPercent > logN > 20%.
        # max(1,..) guards a small dict rounding the product to 0.
        totalKeys = len(nodeFrames)
        if logPercent is not None:
            logStride = max(1, round(totalKeys * logPercent / 100))
        elif logN is not None:
            logStride = logN
        else:
            logStride = max(1, round(totalKeys * 20 / 100))

        # Persist resolved state; graph is built lazily by BuildGraph
        self.nodeFrames = nodeFrames
        self.N          = int(N)
        self.outFile    = outFile
        self.verbose    = bool(verbose)
        self.totalKeys  = totalKeys
        self.logStride  = logStride
        self.rootNode   = rootNode
        self.graph      = None

    def BuildGraph(self):
        # Fresh graph each call -> idempotent, non-accumulating
        self.graph = nx.DiGraph()
        graph      = self.graph

        # Opening timestamp (denominator = total keys)
        if self.verbose:
            print('%s  start  0/%d' % (self._Now(), self.totalKeys))

        # Process each key in insertion order (order is authoritative)
        counter = 0
        for targetKey, nodeFrame in self.nodeFrames.items():
            # Truncate to N rows; pull sources + float32 weights
            usedRows = nodeFrame.head(self.N)
            sources  = usedRows[self.varColumn]
            # to_numpy() keeps np.float32 scalars; iterating the
            # Series would coerce each element back to Python float
            # A frame without rho (no skill measured) weights NaN
            if 'rho' in usedRows.columns:
                rhos = usedRows['rho'].astype(np.float32).to_numpy()
            else:
                rhos = np.full(len(usedRows), np.nan, dtype=np.float32)

            # Add edge sourceVar -> targetKey unless it closes a cycle
            for sourceVar, edgeRho in zip(sources, rhos):
                if self._WouldCycle(graph, sourceVar, targetKey):
                    continue
                graph.add_edge(sourceVar, targetKey, rho=edgeRho)

            # Periodic progress stamp every logStride keys
            counter += 1
            if self.verbose and counter % self.logStride == 0:
                print('%s  %d/%d' %
                      (self._Now(), counter, self.totalKeys))

        # Closing timestamp
        if self.verbose:
            print('%s  done  %d/%d' %
                  (self._Now(), self.totalKeys, self.totalKeys))

        # Serialize by extension: .pkl native, .json via node-link
        if self.outFile is not None:
            ext = os.path.splitext(self.outFile)[1].lower()
            if ext == '.pkl':
                with open(self.outFile, 'wb') as fh:
                    pickle.dump(self.graph, fh)
            else:
                # _JsonValue coerces np.float32 rho for JSON, NaN to
                # null; allow_nan=False rejects any NaN that slips by
                data = nx.node_link_data(self.graph)
                with open(self.outFile, 'w') as fh:
                    json.dump(data, fh, default=self._JsonValue,
                              allow_nan=False)

        return self.graph

    @staticmethod
    def _WouldCycle(graph, sourceVar, targetKey):
        # Absent endpoint -> no path possible -> edge is safe
        if sourceVar not in graph or targetKey not in graph:
            return False
        # Edge closes a loop iff target already reaches source
        return nx.has_path(graph, targetKey, sourceVar)

    @staticmethod
    def _JsonValue(value):
        # NaN rho is not valid JSON: write it as null
        value = float(value)
        return None if np.isnan(value) else value

    @staticmethod
    def _Now():
        # Single-second-precision stamp for all log lines
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def ParseArguments(argv=None):
    # Define the command-line surface
    parser = argparse.ArgumentParser(
        description='Build a networkx DiGraph from a pickled dict '
                    'of pandas DataFrames.'
    )
    parser.add_argument('-gf', '--graphFile', required=True,
                        help='path to pickled dict of DataFrames')
    parser.add_argument('-of', '--outFile', default=None,
                        help='output path; .pkl or .json selects format')
    parser.add_argument('-N', '--max-nodes', dest='N',
                        type=int, default=None,
                        help='rows per node to use (default: all)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        default=False, help='print timestamped progress')
    # logPercent wins over logN when both are given (see GraphMDE)
    parser.add_argument('-ln', '--logN', type=int, default=None,
                        help='log every N keys (verbose only)')
    parser.add_argument('-lp', '--logPercent', type=int, default=None,
                        help='log every P%% of keys (verbose only)')
    return parser.parse_args(argv)


def Main(argv=None):
    # Parse the command line
    args = ParseArguments(argv)

    # Load the pickled dict (trusts the input file by design)
    try:
        with open(args.graphFile, 'rb') as fh:
            nodeFrames = pickle.load(fh)
    except Exception as err:
        print('error: cannot read %s: %s' % (args.graphFile, err),
              file=sys.stderr)
        return 1

    # Construct + build; class raises become clean CLI errors
    try:
        gm    = GraphMDE(nodeFrames, N=args.N, outFile=args.outFile,
                         verbose=args.verbose, logN=args.logN,
                         logPercent=args.logPercent)
        gm.BuildGraph()
    except (TypeError, ValueError) as err:
        print('error: %s' % err, file=sys.stderr)
        return 1
    except OSError as err:
        print('error: cannot write %s: %s' % (args.outFile, err),
              file=sys.stderr)
        return 1

    return 0


# Run as a command-line application
if __name__ == '__main__':
    sys.exit(Main())
