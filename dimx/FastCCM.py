import numpy as np
from pandas import DataFrame

FASTCCM_IMPORT_ERROR = None

try:
    from fastccm import PairwiseCCM, utils as fastccm_utils
    HAS_FASTCCM = True
except Exception as err:
    PairwiseCCM = None
    fastccm_utils = None
    HAS_FASTCCM = False
    FASTCCM_IMPORT_ERROR = err


def _require_fastccm():
    if not HAS_FASTCCM:
        msg = 'FastCCM is required but unavailable.'
        if FASTCCM_IMPORT_ERROR is not None:
            msg += f' Import error: {FASTCCM_IMPORT_ERROR!r}'
        raise RuntimeError(msg)


def _slice_pred_end(pred_end, Tp):
    if int(Tp) >= 0:
        return int(pred_end) - int(Tp)
    return int(pred_end)


def _fastccm_ccm_curves(dataFrame, columns, target, libSizes, sample,
                        E_by_column, Tp, tau, exclusionRadius, seed):
    _require_fastccm()

    tau_ = abs(int(tau))
    if tau_ < 1:
        raise ValueError('FastCCM requires |tau| >= 1.')

    libSizes = [int(libSize) for libSize in libSizes]
    trials   = max(1, int(sample))
    curves   = {column: np.full(len(libSizes), np.nan, dtype=float)
                for column in columns}

    target_vec = dataFrame[target].to_numpy()[:, None]
    unique_E   = sorted({int(E_by_column[column]) for column in columns})
    E_to_src_i = {E: i for i, E in enumerate(unique_E)}
    X_emb      = [fastccm_utils.embed(target_vec, E=E, tau=tau_)[0]
                  for E in unique_E]
    Y_emb      = [fastccm_utils.embed(
                     dataFrame[column].to_numpy()[:, None], E=int(E_by_column[column]), tau=tau_
                  )[0]
                  for column in columns]
    nbrs_num   = [E + 1 for E in unique_E]
    ccm = PairwiseCCM(device='cpu')

    for lib_i, libSize in enumerate(libSizes):
        trial_scores = {column: [] for column in columns}

        for trial in range(trials):
            trial_seed = None if seed is None else int(seed) + trial

            score = ccm.score_matrix(X_emb=X_emb,
                                     Y_emb=Y_emb,
                                     library_size=libSize,
                                     sample_size=libSize,
                                     exclusion_window=exclusionRadius,
                                     tp=Tp,
                                     method='simplex',
                                     seed=trial_seed,
                                     nbrs_num=nbrs_num,
                                     clean_after=False)

            for column_i, column in enumerate(columns):
                E_col = int(E_by_column[column])
                trial_scores[column].append(
                    score[E_col - 1, column_i, E_to_src_i[E_col]]
                )

        for column in columns:
            curves[column][lib_i] = np.nanmean(trial_scores[column])

    return curves


def _fast_simplex_projection_rho(dataFrame, columns, target, lib, pred,
                                 Tp, exclusionRadius):
    _require_fastccm()

    lib_start, lib_end   = _as_range(lib)
    pred_start, pred_end = _as_range(pred)
    pred_stop            = _slice_pred_end(pred_end, Tp)

    if pred_stop <= pred_start - 1:
        raise ValueError('Prediction interval leaves no prediction samples.')

    columns = list(columns)
    if not len(columns):
        return {}

    dim_groups = {}
    for column_list in columns:
        dim_groups.setdefault(len(column_list), []).append(column_list)

    y = np.array(dataFrame[target].to_numpy(), copy=True)[:, None]
    rhoD = {}
    simplex = PairwiseCCM(device='cpu')

    for dim, group in dim_groups.items():
        X_lib = []
        X_pred = []
        for column_list in group:
            X = np.array(dataFrame.loc[:, column_list].to_numpy(), copy=True)
            X_lib.append(X[lib_start - 1:lib_end])
            X_pred.append(X[pred_start - 1:pred_stop])

        Y_lib = [y[lib_start - 1:lib_end]]
        pred_vals = simplex.predict_matrix(
            X_lib_emb=X_lib,
            Y_lib_emb=Y_lib,
            X_pred_emb=X_pred,
            library_size=lib_end - lib_start + 1,
            exclusion_window=exclusionRadius,
            tp=Tp,
            method='simplex',
            nbrs_num=dim + 1,
            clean_after=False,
        )

        y_true = y[pred_start - 1 + int(Tp):pred_end, 0]

        for i, column_list in enumerate(group):
            rho = _corrcoef_safe(pred_vals[:, 0, 0, i], y_true)
            rhoD[f"{','.join(column_list)}:{target}"] = (rho, list(column_list))

    return rhoD


def ComputeCrossMapColumns(dataFrame, columns, target, lib, pred, Tp = 1,
                           exclusionRadius = 0, embedded = True):
    '''Return CrossMapColumns-style rho dict using FastCCM simplex only.'''

    if not embedded:
        raise NotImplementedError(
            'FastCCM CrossMapColumns currently requires embedded=True.')

    return _fast_simplex_projection_rho(
        dataFrame=dataFrame,
        columns=columns,
        target=target,
        lib=lib,
        pred=pred,
        Tp=Tp,
        exclusionRadius=exclusionRadius)


def ComputeCCMCurves(dataFrame, columns, target, libSizes, sample,
                     E_by_column, Tp, tau, exclusionRadius, seed = None,
                     noTime = False):
    '''Return {column : rho(libSizes)} using FastCCM only.'''

    columns = list(columns)
    if not len(columns):
        return {}, 'none'

    curves = _fastccm_ccm_curves(dataFrame=dataFrame,
                                 columns=columns,
                                 target=target,
                                 libSizes=libSizes,
                                 sample=sample,
                                 E_by_column=E_by_column,
                                 Tp=Tp,
                                 tau=tau,
                                 exclusionRadius=exclusionRadius,
                                 seed=seed)

    return curves, 'fastccm'


def _as_range(range_):
    if isinstance(range_, str):
        vals = [int(v) for v in range_.split()]
    else:
        vals = [int(v) for v in range_]

    if len(vals) != 2:
        raise ValueError('range must contain exactly two indices.')

    return vals[0], vals[1]


def _corrcoef_safe(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape or x.size < 2:
        return np.nan
    if np.isnan(x).any() or np.isnan(y).any():
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def _fast_simplex_embed_dimension(dataFrame, columns, target, maxE, lib, pred,
                                  Tp, tau, exclusionRadius):
    _require_fastccm()

    lib_start, lib_end   = _as_range(lib)
    pred_start, pred_end = _as_range(pred)

    if pred_start <= lib_end + int(exclusionRadius):
        raise ValueError('Fast EmbedDimension requires disjoint lib/pred ranges.')

    tau_ = abs(int(tau))
    if tau_ < 1:
        raise ValueError('Fast EmbedDimension requires |tau| >= 1.')

    x = dataFrame[columns].to_numpy()[:, None]
    y = dataFrame[target].to_numpy()
    simplex = PairwiseCCM(device='cpu')
    rho_list = []

    for E in range(1, int(maxE) + 1):
        X_emb = fastccm_utils.embed(x, E=E, tau=tau_)
        emb_len = X_emb.shape[1]
        Y_emb = y[None, -emb_len:, None]

        offset = (E - 1) * tau_
        lib_s  = max(0, lib_start  - 1 - offset)
        lib_e  = max(lib_s, lib_end - offset)
        pred_s = max(0, pred_start - 1 - offset)
        pred_e = max(pred_s, pred_end - offset - max(0, int(Tp)))

        if lib_e <= lib_s or pred_e <= pred_s:
            rho_list.append(np.nan)
            continue

        pred_vals = simplex.predict_matrix(
            X_lib_emb=X_emb[:, lib_s:lib_e],
            Y_lib_emb=Y_emb[:, lib_s:lib_e],
            X_pred_emb=X_emb[:, pred_s:pred_e],
            library_size=lib_end - lib_start + 1,
            exclusion_window=None,
            tp=Tp,
            method='simplex',
            nbrs_num=E + 1,
            clean_after=False,
        )[:, 0, 0, 0]

        y_true = y[pred_s + offset + int(Tp): pred_e + offset + int(Tp)]
        rho_list.append(_corrcoef_safe(pred_vals, y_true))

    return DataFrame({'E': list(range(1, int(maxE) + 1)), 'rho': rho_list})


def ComputeEmbedDimension(dataFrame, columns, target, maxE = 10, lib = "",
                          pred = "", Tp = 1, tau = -1, exclusionRadius = 0,
                          embedded = False, validLib = [], noTime = False,
                          ignoreNan = True, verbose = False, numProcess = 4,
                          mpMethod = None, chunksize = 1, showPlot = False):
    '''Return pyEDM-style EmbedDimension DataFrame using FastCCM only.'''

    if embedded:
        raise NotImplementedError(
            'FastCCM EmbedDimension does not support embedded=True.')

    if not isinstance(columns, str) or len(columns.split()) != 1:
        raise NotImplementedError(
            'FastCCM EmbedDimension currently supports exactly one source column.')

    df = _fast_simplex_embed_dimension(
        dataFrame=dataFrame,
        columns=columns,
        target=target,
        maxE=maxE,
        lib=lib,
        pred=pred,
        Tp=Tp,
        tau=tau,
        exclusionRadius=exclusionRadius)

    return df, 'fastccm'


def FastCCMStatus():
    '''Return diagnostic info about FastCCM availability in this Python env.'''
    return {
        'available': HAS_FASTCCM,
        'import_error': None if FASTCCM_IMPORT_ERROR is None
                        else repr(FASTCCM_IMPORT_ERROR)
    }
