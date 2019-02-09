import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .ssa_ import ssa, SSA


__all__ = ['sliding_ssa', 'overlap_ssa', 'ms_ssa']


def sliding_ssa(ts, lag, n_components, window_size, step=None, window_ends=None):
    """Rolling Singular Spectrum Analysis.

    Parameters
    ----------
    ts : pd.Series or DataFrame
        Time-series with DatetimeIndex
    window_ends : list of DateTime, optional
        Time-intervals
    window_size : int
        Size of the windows for rolling SSA.
    step : int, optional
        Size of the step to skip when rolling the SSA window. Alternative to specifying
        window_ends.
    lag : int
        lag a.k.a. embedding dimension. Maximum should be half the length of input time series
    n_components : int
        number of components to use when reconstructing the time series

    Returns
    -------
    out_series : pd.Series
        The Series objects have a right-labeled timestamp index, where each row contains
        the reconstructed time-series from window_series
        For given index t, it holds a time series reconstructed from the 1st eigenvalue up to and including t.
        I.e. it is a Series of lists.
    """
    if window_ends is None:
        if step is None:
            step = 1
        window_ends = ts.index[window_size::step]

    # Start looping through the SSAs
    model = SSA(lag, n_components)
    ssa_results = []
    for window_end in window_ends:
        assert window_end in ts.index, "Window end %s not present in ts index" % (window_end)

        window_end_idx = ts.index.get_loc(window_end)
        window_start_idx = window_end_idx - window_size # (+1)???
        if window_start_idx < 0:
            raise ValueError("The lowest window end must be > index start + window size.")
        window_ts = ts.iloc[window_start_idx:window_end_idx]
        assert len(window_ts) == window_size

        # window_ssa is the reconstructed time-series.
        window_ssa = model.transform(window_ts)
        ssa_results.append(window_ssa)

    out_series = pd.Series(ssa_results, index=window_ends)
    return out_series


def overlap_ssa(ts, n_windows, embedding_dimension=None, n_components=2,  big_window_ratio=3, n_jobs=1):
    """Overlap-ssa from paper: (2018) A new algorithm in singular spectrum analysis framework:The Overlap-SSA (ov-SSA),
    M.C.R. Leles, J.P.H. Sansão, L.A. Mozelli, H.N. Guimarães,

    Parameters
    ----------
    ts : pd.Series
        input time series
    n_windows : int
        number of windows that will overlap
    embedding_dimension : int, optional
        lag used in SSA. Default: see SSA implementation
    n_components : int, optional
        number of components obtained from SSA to reconstruct the original ts
    big_window_ratio : float, optional
        how much bigger should the big window be bigger than the small one
    n_jobs : int, optionl
        number of parallel jobs

    Returns
    -------
    ts : pd.Series
        reconstruction of the input ts
    """
    if isinstance(ts, pd.Series):
        idx = ts.index
        ts = ts.values
    else:
        idx = None
    ts = np.array(ts)
    if len(ts.shape) != 1:
        raise ValueError('Only 1-D arrays accepted.')
    n = len(ts)
    s_win_size = n // n_windows
    if s_win_size < 1:
        raise ValueError('n_windows: {} is too big for ts of len {}'.format(n_windows, n))
    b_win_size = big_window_ratio * s_win_size
    m = 2 * s_win_size  # overlap size
    print('s_win_size: {}, b_win_size: {} m: {}'.format(s_win_size, b_win_size, m))
    chunks = []
    for i in range(0, n, b_win_size - m):
        ch = ts[i:i + b_win_size]
        if len(ch) < s_win_size:
            break
        chunks.append(ch)
    #chunks = [np.arange(len(ts))[i:i + b_win_size] for i in range(0, n, b_win_size - m)]

    res = Parallel(n_jobs=n_jobs)(delayed(ssa)(ch, embedding_dimension, n_components, verbose=False) for ch in chunks)

    final_ts = res[0][:2*s_win_size]
    for i in range(1, len(res)):
        r = res[i]
        max_id = min(s_win_size*2, len(r))
        final_ts = np.concatenate((final_ts, r[s_win_size:max_id]))

    if idx is not None:
        final_ts = pd.Series(final_ts, index=idx)
    return final_ts


def ms_ssa(ts, window_sizes=None, alpha=3, steps=None, return_bases=2, verbose=False):
    """
    Multi-scale singular-spectrum-analysis introduced by Yiou, Pascal & Sornette, Didier & D Accepted, Physica. (2000).
    Data-Adaptive Wavelets and Multi-Scale SSA.
    :param ts:
    :param window_sizes: Different lengths of the moving window
    :param alpha: factor used to calculate the lag for each window size. Paper suggests 3.
    :param steps: steps[i] is number of steps to skip when moving window os size window_sizes[i].
    :param return_bases: 2 means it returns set of EOF-2 vectors
    :return:
    """

    n = len(ts)
    if window_sizes is None:
        window_sizes = np.array([30, 60, 120, 240])
        window_sizes = window_sizes[window_sizes <= n]
        print('MS-SSA picked the windows of size ', window_sizes)
    if steps is None:
        steps = np.array([10 * (2**i) for i in range(len(window_sizes))])
        print('MS-SSA picked the steps for the window sizes ', steps)

    #window_sizes = window_sizes[1:]
    #steps = steps[1:]

    if verbose:
        print('window_sizes: ', window_sizes)
        print('steps: ', steps)

    if len(window_sizes) != len(steps):
        raise ValueError('{} != {}'.format(len(window_sizes), len(steps)))

    bases_set = []

    for iw, w in enumerate(window_sizes):
        step_size = steps[iw]
        win_starts = range(0, n, step_size)
        m = int(w / alpha)
        for start in win_starts:
            end = start + w
            if end > n:
                break
            if verbose:
                print('start:end = {}:{}'.format(start, end))
            window = ts[start:end]

            reconstructed, bases = ssa(window, lag=m, n_components=3, return_eofs=True, verbose=verbose)
            bases_set.append(bases[:, return_bases])

        # start = int(n / 2 - w / 2)
        # end = start + w
        # window = ts[start:end]
        # reconstructed = ssa(window, embedding_dimension=m, n_components=3, verbose=True, plot=True)
    return bases_set