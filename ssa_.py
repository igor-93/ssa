import sys
import warnings
from time import time
import numpy as np
import pandas as pd
from scipy import linalg

import matplotlib.pylab as plt
from sklearn.utils.extmath import randomized_svd


__all__ = ['SSA', 'ssa']


class SSA(object):

    def __init__(self, lag, n_components, return_eofs=False):
        self.lag = lag
        self.n_components = n_components
        self.return_eofs = return_eofs

    def fit(self):
        return self

    def transform(self, time_series):
        res = ssa(time_series, lag=self.lag, n_components=self.n_components,
                  return_eofs=self.return_eofs, verbose=False)
        return res


def validate_lag(n, lag):
    if lag >= n:
        raise ValueError('lag {} is bigger than time series length {}.'.format(lag, n))
    elif lag > n // 2:
        lag = n // 2
        warnings.warn(
            'Lag is bigger than half of the time-series length, it will be set to {}'.format(lag), RuntimeWarning)
    elif lag < 2:
        raise ValueError('Lag {} is too small'.format(lag))
    return lag


def validate_input(time_series):
    if not isinstance(time_series, np.ndarray):
        if isinstance(time_series, pd.Series):
            time_series = time_series.values
        else:
            time_series = np.array(time_series)

    time_series = np.squeeze(time_series)

    if time_series.ndim > 1:
        raise ValueError('Time series must be 1-D, but it has shape: ', time_series.shape)

    return time_series


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

    if verbose:
        print('L-trajectory matrix (L x K)\t: {}'.format(X.shape))

    return X


def decompose(X, n_components, return_eofs=False, verbose=False, plot_eigenvalues=False):
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
    # Run SVD
    U, lambdas, V = randomized_svd(X, n_components=n_components)

    if n_components is None:
        d = np.linalg.matrix_rank(X)
    else:
        d = len(lambdas)

    if plot_eigenvalues:
        plt.scatter(range(len(s)), s)
        plt.yscale('log')
        plt.ylabel('Eigenvalues')
        plt.xlabel('Index of Eigenvalues')

    Ud = U[:, :n_components]
    Vd = V[:n_components, :]
    Xs = lambdas[:n_components, None, None] * np.einsum("ik,kj->kij", Ud, Vd)

    if not np.all(lambdas > 0):
        raise AssertionError(lambdas)

    if verbose:
        # Get contributions of elementary matrices
        s_contributions = get_contributions(X, s, False)
        num_contribs = len(s_contributions)
        r_characteristic = round((s[:num_contribs] ** 2).sum() / (s ** 2).sum(), 4)
        msg1 = 'Rank of trajectory\t\t: {}'.format(d)
        msg2 = 'Dimension of projection space (num_contribs)\t: {}'.format(num_contribs)
        msg3 = 'Characteristic of projection\t: {}'.format(r_characteristic)

        if d < min(X.shape[0], X.shape[1]):
            _printer('DECOMPOSITION SUMMARY', msg1, msg2, msg3)
        else:
            _printer('DECOMPOSITION SUMMARY', msg2, msg3)

    if return_eofs:
        return Xs, U
    else:
        return Xs


def get_reconstruction(*hankel, names=None, plot=False, symmetric_plots=False):
    """
    Visualise the reconstruction of the hankel matrix/matrices passed to *hankel
    :param hankel: elementary matrices that are going to be summed to get a hankel matrix used for reconstruction
    :param names: Names of the scales used for reconstruction
    :param plot:
    :param symmetric_plots: If True, plot is in the middle of y axis
    :return:
    """
    hankel_mat = None
    for i, han in enumerate(hankel):
        if i == 0:
            hankel_mat = han
        else:
            hankel_mat = hankel_mat + han

    reconstructed_ts = diagonal_averaging(hankel_mat)

    if plot:
        title = 'Reconstruction of signal'
        if names or names == 0:
            if isinstance(names, range):
                names = list(names)
            title += ' associated with singular value{}: {}'
            title = title.format('' if len(str(names)) == 1 else 's', names)
        plt.figure()
        plt.plot(reconstructed_ts)
        plt.title(title)
        if symmetric_plots:
            velocity = np.abs(reconstructed_ts).max()
            plt.ylim(bottom=-velocity, top=velocity)

    return reconstructed_ts


def get_contributions(X, s, plot=False):
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
    L, K = hankel_matrix.shape

    flipped = hankel_matrix[::-1, :]
    divisors = np.concatenate((np.arange(1, L), [L] * (K - L), np.arange(L, 0, -1)))
    x1d = np.array([np.trace(flipped, i) for i in range(-L + 1, K)])
    x1d = x1d / divisors
    return x1d


def ssa(time_series, lag, n_components, return_eofs=False, verbose=False):
    """Function to run Singular Spectrum Analysis

    Parameters
    ----------
    time_series : array-like
        1D array or pd.Series input to run SSA on.
    lag : int
        lag a.k.a. embedding dimension. Maximum should be half the length of input time series
    n_components : int
        number of components to use when reconstructing the time series
    return_eofs : bool, optional
        if True, the function also returns EOFs. Default False.
    verbose : bool
        if True, print some messages during execution of the algorithm

    Returns
    -------
    result: ndarray
        reconstruction of the input time series

    """
    time_series = validate_input(time_series)
    n = len(time_series)
    lag = validate_lag(n, lag)

    # Get trajectory matrix
    X = embed(time_series, embedding_dimension=lag, verbose=verbose)

    # Run SVD and get elementary matrices and orthonormal bases
    res = decompose(X, n_components=n_components, return_eofs=return_eofs, verbose=verbose)
    if return_eofs:
        Xs, U = res
    else:
        Xs = res

    result = get_reconstruction(*[Xs[i] for i in range(n_components)])

    if return_eofs:
        return result, U
    else:
        return result



