# https://github.com/aj-cloete/pySSA
import sys
import numpy as np
import pandas as pd
from scipy import linalg

import matplotlib.pylab as plt

plt.rcParams['figure.figsize'] = 11, 4


def _printer(name, *args):
    """
    Helper function to print messages neatly
    """
    print('-' * 40)
    print(name + ':')
    for msg in args:
        print(msg)


def view_s_contributions(s_contributions, adjust_scale=False, cumulative=False):
    """
    Plots the contribution to variance of each singular value's signal.
    :param adjust_scale:
    :param cumulative:
    :return:
    """
    contribs = s_contributions.copy()
    if cumulative:
        contribs['Contribution'] = contribs.Contribution.cumsum()
    if adjust_scale:
        contribs = (1 / contribs).max() * 1.1 - (1 / contribs)
    ax = contribs.plot.bar(legend=False)
    ax.set_xlabel("Singular_i")
    ax.set_title('Non-zero{} contribution of Singular_i {}'. \
                 format(' cumulative' if cumulative else '', '(scaled)' if adjust_scale else ''))
    if adjust_scale:
        ax.axes.get_yaxis().set_visible(False)
    else:
        ax.set_yscale('log')
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.0f}%'.format(x * 100) for x in vals])


def view_time_series(time_series, show=False):
    """
    Plot the input time series.
    :param time_series:
    :return:
    """
    plt.figure()
    plt.plot(time_series)
    plt.title('Original Time Series')
    if show:
        plt.show()


def embed(time_series, embedding_dimension=None, verbose=False):
    """
    Embed the time series with embedding_dimension window size.
    :param time_series: input time series
    :param embedding_dimension: L ( or M in paper). K (W in paper) is derived as N - L + 1.
    It should be it most half of the length of the time series and higher it is,
    better is the reconstruction of the original signal.
    :param verbose: if True, print some info.
    :return: Trajectory matrix of shape L * K that contains lagged copies of the time series.
    """

    ts_N = len(time_series)

    K = ts_N - embedding_dimension + 1
    X = linalg.hankel(time_series, np.zeros(embedding_dimension)).T[:, :K]
    if np.isnan(X).any():
        raise ValueError('NaN must not appear in the lag correlation matrix if there are no NaNs in the input '
                         'time series.')
    X = np.matrix(X)

    if verbose:
        print('L-trajectory matrix (L x K)\t: {}'.format(X.shape))

    return X


def decompose(X, verbose=False, plot_eigenvalues=False):
    """
    Performs the Singular Value Decomposition and identifies the rank of the embedding subspace
    Characteristic of projection: the proportion of variance captured in the subspace.
    :param X: hankel matrix of the time series
    :param verbose:
    :param plot_eigenvalues:
    :return:
    Xs : 3D matrix  of shape d * X.shape (d is rank of X) elementary matrices. Sum of all of them equals X.
    s_contributions: Pandas DataFrame that contains contributions for elementary matrices.
                     It's index might miss some values, i.e. for contributions smaller then 1.
    orthonormal_base:  U[:, :num_contribs], where num_contribs is number of positive contributions
    """
    # Build lag-correlation matrix (C in paper)
    S = X * X.T

    # Run SVD
    U, lambdas, V = linalg.svd(S)
    U, s, V = np.matrix(U), np.sqrt(lambdas), np.matrix(V)
    d = np.linalg.matrix_rank(X)

    if plot_eigenvalues:
        plt.scatter(range(len(s)), s)
        plt.yscale('log')
        plt.ylabel('Eigenvalues')
        plt.xlabel('Index of Eigenvalues')


    Xs = np.zeros((d, X.shape[0], X.shape[1]))
    for i in range(d):
        V_i = X.T * (U[:, i] / s[i])
        Xs[i, :, :] = s[i] * U[:, i] * np.matrix(V_i).T

    # Get contributions of elementary matrices
    s_contributions = get_contributions(X, s, False)
    num_contribs = len(s_contributions)
    r_characteristic = round((s[:num_contribs] ** 2).sum() / (s ** 2).sum(), 4)
    #orthonormal_base = U[:, :num_contribs] #{i: U[:, i] for i in range(num_contribs)}

    if not np.all(lambdas > 0):
        raise AssertionError(lambdas)
    orthonormal_base = U

    if verbose:
        msg1 = 'Rank of trajectory\t\t: {}'.format(d)
        msg2 = 'Dimension of projection space (num_contribs)\t: {}'.format(num_contribs)
        msg3 = 'Characteristic of projection\t: {}'.format(r_characteristic)

        if d < min(X.shape[0], X.shape[1]):
            _printer('DECOMPOSITION SUMMARY', msg1, msg2, msg3)
        else:
            _printer('DECOMPOSITION SUMMARY', msg2, msg3)

    return Xs, s_contributions, orthonormal_base


def get_reconstruction(*hankel, names=None, plot=True, symmetric_plots=False):
    """
    Visualise the reconstruction of the hankel matrix/matrices passed to *hankel
    :param hankel: elementary matrices that are going to be summed to get a hankel matrix used for reconstruction
    :param names: Names of the scales used for reconstruction
    :param plot:
    :param symmetric_plots: If True, plot is in the middle of y axis
    :return:
    """
    hankel_mat = None
    for han in hankel:
        if isinstance(hankel_mat, np.matrix):
            hankel_mat = hankel_mat + han
        else:
            hankel_mat = np.matrix(han)
    reconstructed_ts = diagonal_averaging(hankel_mat)
    title = 'Reconstruction of signal'
    if names or names == 0:
        if isinstance(names, range):
            names = list(names)
        title += ' associated with singular value{}: {}'
        title = title.format('' if len(str(names)) == 1 else 's', names)
    if plot:
        plt.figure()
        plt.plot(reconstructed_ts)
        plt.title(title)
        if symmetric_plots:
            velocity = np.abs(reconstructed_ts).max()
            plt.ylim(bottom=-velocity, top=velocity)

    return reconstructed_ts


def get_contributions(X, s, plot=True):
    """
    Calculate the relative contribution of each of the singular values
    :param X: Trajectory matrix of shape L * K that contains lagged copies of the time series
    :param s: Singular values of X
    :param plot: pd.DataFrame with > 0 contributions. Index is the id of eigenvalue and 'Contribution' is the result.
    :return:
    """
    # Eigenvalues
    lambdas = np.square(s)
    forb_norm = np.linalg.norm(X)
    ret = pd.DataFrame(lambdas / (forb_norm ** 2), columns=['Contribution'])
    ret['Contribution'] = ret.Contribution.round(4)

    if plot:
        ax = ret[ret.Contribution != 0].plot.bar(legend=False)
        ax.set_xlabel("Lambda_i")
        ax.set_title('Non-zero contributions of Lambda_i')
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:3.2f}%'.format(x * 100) for x in vals])
        return ax
    return ret[ret.Contribution > 0]


def diagonal_averaging(hankel_matrix):
    """
    Performs anti-diagonal averaging from given hankel matrix.
    :param hankel_matrix:
    :return: reconstructed series
    """
    mat = np.matrix(hankel_matrix)
    L, K = mat.shape
    L_star, K_star = min(L, K), max(L, K)
    if L > K:
        mat = mat.T
    ret = []

    # Diagonal Averaging
    for k in range(1 - K_star, L_star):
        mask = np.eye(K_star, k=k, dtype='bool')[::-1][:L_star, :]
        mask_n = sum(sum(mask))
        ma = np.ma.masked_array(mat.A, mask=1 - mask)
        ret += [ma.sum() / mask_n]

    return np.array(ret)


def generate_price(T, log_prices=True):
    ts = range(0, T)
    np.random.seed(None)
    rands = np.random.randn(T)
    Ws = [np.sum(rands[:t]) / np.sqrt(T) for t in ts]
    sig = np.exp(Ws)
    ys = sig * (np.random.choice(100) + 1)
    if log_prices:
        ys = np.log(ys)
    return ys


def ssa(time_series, embedding_dimension=None, n_components=2, verbose=True, plot=False, add_component=None):
    """
    Main function to run Singular-Spectrum-Analysis
    :param time_series:
    :param embedding_dimension:
    :param n_components:
    :param verbose:
    :param plot:
    :return:
    """
    if not isinstance(time_series, np.ndarray):
        if isinstance(time_series, pd.Series):
            time_series = time_series.values
        else:
            time_series = np.array(time_series)

    if time_series.ndim > 1 and time_series.shape[1] > 1:
        raise ValueError('Time series must be 1-D, but it has shape: ', time_series.shape)

    if not embedding_dimension:
        embedding_dimension = 2 * int(np.sqrt(len(time_series)))
        print('embedding_dimension is automatically set to ', embedding_dimension)

    # Get trajectory matrix
    X = embed(time_series, embedding_dimension=embedding_dimension, verbose=verbose)

    # Run SVD and get elementary matrices and orthonormal bases
    Xs, s_contributions, orthonormal_base = decompose(X, verbose=verbose, plot_eigenvalues=plot)

    if plot:
        view_s_contributions(s_contributions, adjust_scale=True)

    # Components of the Signals
    plot_components = False
    if plot_components and plot:
        plt.rcParams['figure.figsize'] = 11, 2
        for i in range(3):
            get_reconstruction(Xs[i], names=i, symmetric_plots=i != 0)
        plt.rcParams['figure.figsize'] = 11, 4

    result = get_reconstruction(*[Xs[i] for i in range(n_components)], names=range(n_components), plot=False)

    if plot:
        # plot EOFs
        plt.figure()
        plt.plot(orthonormal_base[:, 0], label='EOF 0')
        plt.plot(orthonormal_base[:, 1], label='EOF 1')
        plt.plot(orthonormal_base[:, 2], label='EOF 2')
        plt.legend()
        plt.title('EOFs')

        # plot reconstructed signal
        rec1 = get_reconstruction(*[Xs[i] for i in range(1)], names=range(2), plot=False)
        rec2 = get_reconstruction(*[Xs[i] for i in range(2)], names=range(2), plot=False)
        plt.figure()
        plt.plot(time_series, label='Original')
        plt.plot(rec1, label='From 1 component')
        plt.plot(rec2, label='From 2 components')
        if n_components not in [1,2]:
            plt.plot(result, label='Reconstruction from {} components'.format(n_components))
        plt.legend()
        #plt.xlabel('Time (Months)')
        #plt.ylabel('# Passengers (000)')
        plt.title('Reconstructed signal')
        # plt.show()

    if add_component is None:
        return result, orthonormal_base[:, :n_components]
    else:
        return result, orthonormal_base[:, :n_components], get_reconstruction(*[Xs[i] for i in [add_component]], names=range(2), plot=False)


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


if __name__ == '__main__':

    # Construct the data with gaps
    #ts = pd.read_csv('air_passengers.csv', parse_dates=True, index_col='Month')
    random_ts = generate_price(240)

    embedding_dim = 3 * int(np.sqrt(len(random_ts)))
    #res = ssa(ts, embedding_dimension=embedding_dim, n_components=5, verbose=True, plot=True)

    ms_ssa(random_ts)

