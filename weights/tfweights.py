from __future__ import division
from scipy import sparse
import numpy as np


def binary(t_mat):
    # type: (sparse.csc_matrix) -> sparse.csc_matrix
    """
    :param scipy.sparse.csc_matrix t_mat:
    :return scipy.sparse.csc_matrix :
    """
    mat = t_mat.copy()
    mat.data = np.ones_like(mat.data, dtype=np.float64)
    return mat


def raw_count(t_mat):
    # type: (sparse.csc_matrix) -> sparse.csc_matrix
    """
    :param scipy.sparse.csc_matrix t_mat:
    :return scipy.sparse.csc_matrix :
    """
    mat = t_mat.copy()
    return mat


def term_freq(t_mat):
    # type: (sparse.csc_matrix) -> sparse.csc_matrix
    """
    :param scipy.sparse.csc_matrix t_mat:
    :return scipy.sparse.csc_matrix :
    """
    term_mul = np.power(t_mat.sum(axis=0).astype(np.float64), -1)
    term_rep = t_mat.getnnz(axis=0)
    mul_mat = sparse.csc_matrix((np.repeat(term_mul, term_rep), t_mat.indices, t_mat.indptr), shape=t_mat.shape)
    return t_mat.multiply(mul_mat)


def log_norm(t_mat):
    # type: (sparse.csc_matrix) -> sparse.csc_matrix
    """
    :param scipy.sparse.csc_matrix t_mat:
    :return scipy.sparse.csc_matrix :
    """
    mat = t_mat.copy()
    mat.data = 1 + np.log(mat.data)
    return mat


def double_norm(t_mat):
    # type: (sparse.csc_matrix) -> sparse.csc_matrix
    """
    :param scipy.sparse.csc_matrix t_mat:
    :return scipy.sparse.csc_matrix :
    """
    row_max = 0.5 * np.array(t_mat.max(axis=0).todense(), dtype=np.float64)
    row_max[0 == row_max] = 1  # To avoid division by zero - if the max is zero all values are zeros
    term_mul = np.power(row_max, -1)
    term_rep = t_mat.getnnz(axis=0)
    mul_mat = sparse.csc_matrix((np.repeat(term_mul, term_rep), t_mat.indices, t_mat.indptr), shape=t_mat.shape)
    mat = t_mat.multiply(mul_mat)
    mat.data = 0.5 + mat.data
    return mat


tfmethods = {
    'binary': binary,
    'raw_count': raw_count,
    'term_freq': term_freq,
    'log_norm': log_norm,
    'double_norm': double_norm
}
