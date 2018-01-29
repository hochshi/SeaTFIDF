from __future__ import division
from scipy import sparse
import numpy as np
from collections import Counter
from cf_ingredient import imap


def unary(t_mat):
    # type: (sparse.csc_matrix) -> sparse.csc_matrix
    """
    :param scipy.sparse.csc_matrix t_mat:
    :return scipy.sparse.csc_matrix :
    """
    tfc = Counter(t_mat.indices)
    indices = tfc.keys()
    return sparse.csc_matrix((np.ones_like(indices), indices, [0, len(indices)]), shape=(imap.size, 1))


def idf(t_mat):
    # type: (sparse.csc_matrix) -> sparse.csc_matrix
    """
    :param scipy.sparse.csc_matrix t_mat:
    :return scipy.sparse.csc_matrix :
    """
    tfc = Counter(t_mat.indices)
    idf = []
    indices = []
    for key, val in tfc.iteritems():
        idf.append(np.log(t_mat.shape[1] / val))
        indices.append(key)
    return sparse.csc_matrix((idf, indices, [0, len(indices)]), shape=(imap.size, 1))


def idf_smooth(t_mat):
    # type: (sparse.csc_matrix) -> sparse.csc_matrix
    """
    :param scipy.sparse.csc_matrix t_mat:
    :return scipy.sparse.csc_matrix :
    """
    tfc = Counter(t_mat.indices)
    idf = []
    indices = []
    doc_no = t_mat.shape[1]
    for key, val in tfc.iteritems():
        idf.append(np.log(1 + doc_no / val))
        indices.append(key)
    return sparse.csc_matrix((idf, indices, [0, len(indices)]), shape=(imap.size, 1))


def idf_max(t_mat):
    # type: (sparse.csc_matrix) -> sparse.csc_matrix
    """
    :param scipy.sparse.csc_matrix t_mat:
    :return scipy.sparse.csc_matrix :
    """
    tfc = Counter(t_mat.indices)
    idf = []
    indices = []
    term_max = np.max(tfc.values())
    for key, val in tfc.iteritems():
        idf.append(np.log(term_max / (1 + val)))
        indices.append(key)
    return sparse.csc_matrix((idf, indices, [0, len(indices)]), shape=(imap.size, 1))


def prob_idf(t_mat):
    # type: (sparse.csc_matrix) -> sparse.csc_matrix
    """
    :param scipy.sparse.csc_matrix t_mat:
    :return scipy.sparse.csc_matrix :
    """
    tfc = Counter(t_mat.indices)
    idf = []
    indices = []
    doc_no = t_mat.shape[1]
    for key, val in tfc.iteritems():
        idf.append(np.log((doc_no - val) / val))
        indices.append(key)
    return sparse.csc_matrix((idf, indices, [0, len(indices)]), shape=(imap.size, 1))


idfmethods = {
    'unary': unary,
    'idf': idf,
    'idf_smooth': idf_smooth,
    'idf_max': idf_max,
    'prob_idf': prob_idf
}