import numpy as np
from joblib import Parallel, delayed

from .ssa_ import *


def overlap_ssa(ts, n_windows, embedding_dimension=None, n_components=2,  big_window_ratio=3, n_jobs=1):
    """
    Overlap-ssa from paper: (2018) A new algorithm in singular spectrum analysis framework:The Overlap-SSA (ov-SSA),
    M.C.R. Leles, J.P.H. Sansão, L.A. Mozelli, H.N. Guimarães,
    :param ts:
    :param n_windows:
    :param embedding_dimension:
    :param n_components:
    :param big_window_ratio:
    :param n_jobs:
    :return:
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
    res = [r[0] for r in res]

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

            reconstructed, bases = ssa(window, embedding_dimension=m, n_components=3, verbose=verbose, plot=False)
            bases_set.append(bases[:, return_bases])

        # start = int(n / 2 - w / 2)
        # end = start + w
        # window = ts[start:end]
        # reconstructed = ssa(window, embedding_dimension=m, n_components=3, verbose=True, plot=True)
    return bases_set