#!/usr/bin/env python3

import argparse
import json
import math
import os
import subprocess
import sys
from copy import deepcopy

import numpy as np
from scipy.signal import argrelextrema


def _as_list(value):
    if value is None:
        return []
    return list(value)


def _float_or_none(value):
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def _linear_slope(x_vals, y_vals):
    x = np.asarray(x_vals, dtype=float).reshape(-1)
    y = np.asarray(y_vals, dtype=float).reshape(-1)
    if x.size != y.size or x.size < 2:
        return None
    y = np.nan_to_num(y)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.square(x - x_mean).sum()
    if denom == 0:
        return 0.0
    slope = ((x - x_mean) * (y - y_mean)).sum() / denom
    return round(float(slope), 5)


def _select_embed_dimension(embed_trace, first_emax):
    e_vals = np.asarray(embed_trace['E'], dtype=int)
    rho_vals = np.asarray(embed_trace['rho'], dtype=float)
    if not len(e_vals):
        return {'maxE': None, 'maxRho': None}

    if first_emax:
        maxima = argrelextrema(rho_vals, np.greater)[0]
        idx = int(maxima[0]) if len(maxima) else len(e_vals) - 1
    else:
        idx = int(np.nanargmax(np.round(rho_vals, 4)))

    return {
        'maxE': int(e_vals[idx]),
        'maxRho': round(float(rho_vals[idx]), 4),
    }


def _make_parser():
    parser = argparse.ArgumentParser(description='Compare MDE decisions between two repo trees.')
    parser.add_argument('--repo-a', type=str, help='Path to first repo tree.')
    parser.add_argument('--repo-b', type=str, help='Path to second repo tree.')
    parser.add_argument('--label-a', type=str, default='current')
    parser.add_argument('--label-b', type=str, default='original')
    parser.add_argument('--data', type=str, required=True, help='CSV data file.')
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--remove-columns', nargs='*', default=[])
    parser.add_argument('--D', type=int, default=12)
    parser.add_argument('--lib', nargs=2, type=int, required=True)
    parser.add_argument('--pred', nargs=2, type=int, required=True)
    parser.add_argument('--Tp', type=int, default=1)
    parser.add_argument('--tau', type=int, default=-1)
    parser.add_argument('--exclusion-radius', type=int, default=0)
    parser.add_argument('--sample', type=int, default=20)
    parser.add_argument('--p-lib-sizes', nargs='*', type=int, default=[10, 15, 85, 90])
    parser.add_argument('--no-ccm', action='store_true')
    parser.add_argument('--ccm-slope', type=float, default=0.01)
    parser.add_argument('--ccm-seed', type=int, default=None)
    parser.add_argument('--E', type=int, default=0)
    parser.add_argument('--cross-map-rho-min', type=float, default=0.5)
    parser.add_argument('--embed-dim-rho-min', type=float, default=0.5)
    parser.add_argument('--maxE', type=int, default=15)
    parser.add_argument('--first-emax', action='store_true')
    parser.add_argument('--time-delay', type=int, default=0)
    parser.add_argument('--trace-one', action='store_true', help=argparse.SUPPRESS)
    return parser


def _serialize_cross_map(rho_d):
    rows = []
    for rho, columns in rho_d.values():
        rows.append({
            'rho': _float_or_none(rho),
            'columns': list(columns),
        })
    return rows


def _trace_one(args):
    repo_path = os.path.abspath(args.repo_a)
    sys.path.insert(0, repo_path)

    trace = {
        'repo_path': repo_path,
        'dimensions': [],
    }

    import pandas as pd
    import dimx

    run_mod = sys.modules['dimx.Run']
    mde_cls = dimx.MDE

    state = {
        'cross_map_calls': 0,
    }

    def current_dim():
        if not trace['dimensions']:
            raise RuntimeError('Trace state lost: no current dimension.')
        return trace['dimensions'][-1]

    if hasattr(run_mod, 'CrossMapColumns'):
        original_cross_map = run_mod.CrossMapColumns

        def wrapped_cross_map(*wrapper_args, **wrapper_kwargs):
            rho_d = original_cross_map(*wrapper_args, **wrapper_kwargs)
            state['cross_map_calls'] += 1
            trace['dimensions'].append({
                'd': state['cross_map_calls'],
                'cross_map': _serialize_cross_map(rho_d),
                'embed': [],
                'ccm': [],
            })
            return rho_d

        run_mod.CrossMapColumns = wrapped_cross_map

    if hasattr(run_mod, 'ComputeCrossMapColumns'):
        original_compute_cross_map = run_mod.ComputeCrossMapColumns

        def wrapped_compute_cross_map(*wrapper_args, **wrapper_kwargs):
            rho_d = original_compute_cross_map(*wrapper_args, **wrapper_kwargs)
            state['cross_map_calls'] += 1
            trace['dimensions'].append({
                'd': state['cross_map_calls'],
                'cross_map': _serialize_cross_map(rho_d),
                'embed': [],
                'ccm': [],
            })
            return rho_d

        run_mod.ComputeCrossMapColumns = wrapped_compute_cross_map

    if hasattr(run_mod, 'EmbedDimension'):
        original_embed = run_mod.EmbedDimension

        def wrapped_embed(*wrapper_args, **wrapper_kwargs):
            df = original_embed(*wrapper_args, **wrapper_kwargs)
            current_dim()['embed'].append({
                'column': wrapper_kwargs.get('columns'),
                'backend': 'pyEDM',
                'E': [int(v) for v in df['E'].tolist()],
                'rho': [_float_or_none(v) for v in df['rho'].tolist()],
            })
            return df

        run_mod.EmbedDimension = wrapped_embed

    if hasattr(run_mod, 'ComputeEmbedDimension'):
        original_compute_embed = run_mod.ComputeEmbedDimension

        def wrapped_compute_embed(*wrapper_args, **wrapper_kwargs):
            df, backend = original_compute_embed(*wrapper_args, **wrapper_kwargs)
            current_dim()['embed'].append({
                'column': wrapper_kwargs.get('columns'),
                'backend': backend,
                'E': [int(v) for v in df['E'].tolist()],
                'rho': [_float_or_none(v) for v in df['rho'].tolist()],
            })
            return df, backend

        run_mod.ComputeEmbedDimension = wrapped_compute_embed

    if hasattr(run_mod, 'CCM'):
        original_ccm = run_mod.CCM

        def wrapped_ccm(*wrapper_args, **wrapper_kwargs):
            ccm_df = original_ccm(*wrapper_args, **wrapper_kwargs)
            column = wrapper_kwargs.get('columns')
            curves = {}
            for key in ccm_df.columns:
                if ':' in key:
                    curves[key] = [_float_or_none(v) for v in ccm_df[key].tolist()]
            current_dim()['ccm'].append({
                'backend': 'pyEDM',
                'column': column,
                'libSizes': [int(v) for v in ccm_df['LibSize'].tolist()],
                'curves': curves,
            })
            return ccm_df

        run_mod.CCM = wrapped_ccm

    if hasattr(run_mod, 'ComputeCCMCurves'):
        original_compute_ccm = run_mod.ComputeCCMCurves

        def wrapped_compute_ccm(*wrapper_args, **wrapper_kwargs):
            curves, backend = original_compute_ccm(*wrapper_args, **wrapper_kwargs)
            current_dim()['ccm'].append({
                'backend': backend,
                'columns': list(wrapper_kwargs.get('columns', [])),
                'libSizes': [int(v) for v in wrapper_kwargs.get('libSizes', [])],
                'curves': {
                    key: [_float_or_none(v) for v in values.tolist()]
                    for key, values in curves.items()
                },
                'E_by_column': {
                    key: int(value)
                    for key, value in wrapper_kwargs.get('E_by_column', {}).items()
                },
            })
            return curves, backend

        run_mod.ComputeCCMCurves = wrapped_compute_ccm

    data_path = os.path.abspath(args.data)
    df = pd.read_csv(data_path)
    mde = mde_cls(
        df,
        target=args.target,
        removeColumns=args.remove_columns,
        D=args.D,
        lib=args.lib,
        pred=args.pred,
        Tp=args.Tp,
        tau=args.tau,
        exclusionRadius=args.exclusion_radius,
        sample=args.sample,
        pLibSizes=args.p_lib_sizes,
        noCCM=args.no_ccm,
        ccmSlope=args.ccm_slope,
        ccmSeed=args.ccm_seed,
        E=args.E,
        crossMapRhoMin=args.cross_map_rho_min,
        embedDimRhoMin=args.embed_dim_rho_min,
        maxE=args.maxE,
        firstEMax=args.first_emax,
        timeDelay=args.time_delay,
        consoleOut=False,
        verbose=False,
        debug=False,
        plot=False,
    )
    mde.Run()

    trace['final'] = {
        'MDEcolumns': list(mde.MDEcolumns),
        'MDErho': [_float_or_none(v) for v in mde.MDErho.tolist()],
        'libSizes': [int(v) for v in mde.libSizes],
    }
    print(json.dumps(trace))


def _rank_cross_map_rows(rows, cross_map_rho_min):
    ranked = sorted(rows, key=lambda row: row['rho'], reverse=True)
    return [deepcopy(row) for row in ranked if row['rho'] > cross_map_rho_min]


def _summarize_trace(trace, args):
    lib_sizes = trace['final']['libSizes']
    lib_sizes_vec = np.asarray(lib_sizes, dtype=float).reshape(-1, 1)
    lib_sizes_vec = lib_sizes_vec / lib_sizes_vec[-1]

    summary = {
        'final_columns': trace['final']['MDEcolumns'],
        'final_rho': trace['final']['MDErho'],
        'dimensions': [],
    }

    for dim_trace in trace['dimensions']:
        ranked = _rank_cross_map_rows(dim_trace['cross_map'], args.cross_map_rho_min)
        embed_by_column = {
            row['column']: row
            for row in dim_trace['embed']
        }
        ccm_entries = dim_trace['ccm']
        chosen = None
        rows = []

        batched_ccm = None
        if ccm_entries and 'columns' in ccm_entries[-1]:
            batched_ccm = ccm_entries[-1]

        for row in ranked:
            new_column = row['columns'][0]
            decision = {
                'column': new_column,
                'cross_map_rho': row['rho'],
                'embed_maxE': None,
                'embed_maxRho': None,
                'embed_pass': False,
                'ccm_slope': None,
                'ccm_pass': False,
            }

            if args.E > 0:
                decision['embed_maxE'] = int(args.E)
                decision['embed_maxRho'] = round(row['rho'], 4)
            elif new_column in embed_by_column:
                embed_pick = _select_embed_dimension(embed_by_column[new_column], args.first_emax)
                decision['embed_maxE'] = embed_pick['maxE']
                decision['embed_maxRho'] = embed_pick['maxRho']

            if decision['embed_maxRho'] is not None and decision['embed_maxRho'] >= args.embed_dim_rho_min:
                decision['embed_pass'] = True

            if decision['embed_pass']:
                if args.no_ccm:
                    decision['ccm_pass'] = True
                elif batched_ccm is not None:
                    curve = batched_ccm['curves'].get(new_column)
                    if curve is not None:
                        slope = _linear_slope(lib_sizes_vec, curve)
                        decision['ccm_slope'] = slope
                        decision['ccm_pass'] = slope is not None and slope > args.ccm_slope
                else:
                    for ccm_entry in ccm_entries:
                        if ccm_entry.get('column') != new_column:
                            continue
                        curve = ccm_entry['curves'].get(f'{args.target}:{new_column}')
                        slope = _linear_slope(lib_sizes_vec, curve)
                        decision['ccm_slope'] = slope
                        decision['ccm_pass'] = slope is not None and slope > args.ccm_slope
                        break

            rows.append(decision)
            if chosen is None:
                if args.no_ccm:
                    chosen = new_column
                elif decision['embed_pass'] and decision['ccm_pass']:
                    chosen = new_column

        summary['dimensions'].append({
            'd': dim_trace['d'],
            'candidate_count': len(dim_trace['cross_map']),
            'ranked_count': len(ranked),
            'chosen': chosen,
            'rows': rows,
        })

    return summary


def _run_trace_subprocess(script_path, repo_path, common_argv):
    cmd = [
        sys.executable,
        script_path,
        '--trace-one',
        '--repo-a', repo_path,
        *common_argv,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _first_divergence(summary_a, summary_b):
    max_dims = max(len(summary_a['dimensions']), len(summary_b['dimensions']))
    for idx in range(max_dims):
        if idx >= len(summary_a['dimensions']) or idx >= len(summary_b['dimensions']):
            return idx + 1, 'dimension count differs'
        dim_a = summary_a['dimensions'][idx]
        dim_b = summary_b['dimensions'][idx]
        if dim_a['chosen'] != dim_b['chosen']:
            return idx + 1, f"chosen differs: {dim_a['chosen']} vs {dim_b['chosen']}"
        rows_a = dim_a['rows']
        rows_b = dim_b['rows']
        limit = min(len(rows_a), len(rows_b))
        for row_i in range(limit):
            if rows_a[row_i]['column'] != rows_b[row_i]['column']:
                return idx + 1, (
                    f"ranking differs at rank {row_i + 1}: "
                    f"{rows_a[row_i]['column']} vs {rows_b[row_i]['column']}"
                )
        if len(rows_a) != len(rows_b):
            return idx + 1, 'ranked candidate count differs'
    return None, 'no divergence found'


def _common_argv_from_args(args):
    argv = [
        '--data', os.path.abspath(args.data),
        '--target', args.target,
        '--D', str(args.D),
        '--lib', str(args.lib[0]), str(args.lib[1]),
        '--pred', str(args.pred[0]), str(args.pred[1]),
        '--Tp', str(args.Tp),
        '--tau', str(args.tau),
        '--exclusion-radius', str(args.exclusion_radius),
        '--sample', str(args.sample),
        '--ccm-slope', str(args.ccm_slope),
        '--cross-map-rho-min', str(args.cross_map_rho_min),
        '--embed-dim-rho-min', str(args.embed_dim_rho_min),
        '--maxE', str(args.maxE),
    ]
    if args.remove_columns:
        argv.extend(['--remove-columns', *args.remove_columns])
    if args.p_lib_sizes:
        argv.extend(['--p-lib-sizes', *[str(v) for v in args.p_lib_sizes]])
    if args.no_ccm:
        argv.append('--no-ccm')
    if args.ccm_seed is not None:
        argv.extend(['--ccm-seed', str(args.ccm_seed)])
    if args.E:
        argv.extend(['--E', str(args.E)])
    if args.first_emax:
        argv.append('--first-emax')
    if args.time_delay:
        argv.extend(['--time-delay', str(args.time_delay)])
    return argv


def main():
    parser = _make_parser()
    args = parser.parse_args()

    if args.trace_one:
        _trace_one(args)
        return

    script_path = os.path.abspath(__file__)
    common_argv = _common_argv_from_args(args)
    trace_a = _run_trace_subprocess(script_path, os.path.abspath(args.repo_a), common_argv)
    trace_b = _run_trace_subprocess(script_path, os.path.abspath(args.repo_b), common_argv)

    summary_a = _summarize_trace(trace_a, args)
    summary_b = _summarize_trace(trace_b, args)
    div_d, div_reason = _first_divergence(summary_a, summary_b)

    report = {
        args.label_a: summary_a,
        args.label_b: summary_b,
        'first_divergence': {
            'dimension': div_d,
            'reason': div_reason,
        }
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
