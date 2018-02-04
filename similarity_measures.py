import numpy as np
from scipy import sparse
from cf_ingredient import compound_target_mat
from log_ingredient import log_np_data
import pandas as pd
from weights import tfmethods
from typing import Iterable


def cosine(mata, matb):
    # type: (sparse.csc_matrix, sparse.csc_matrix) -> np.ndarray
    """
    :param scipy.sparse.csc_matrix mata: (l features, m samples)
    :param scipy.sparse.csc_matrix matb: (l features, n samples)
    :return numpy.ndarray :
    """
    # return pairwise_distances(mata, matb, metric='cosine', n_jobs=-2)
    a_samples = mata.shape[1]
    b_samples = matb.shape[1]
    nom = (mata.transpose() * matb).todense()  # shape: MxN
    a_sums = np.array(mata.power(2).sum(axis=0), dtype=np.float64)  # m sums
    b_sums = np.array(matb.power(2).sum(axis=0), dtype=np.float64)  # n sums
    denom = np.multiply(np.repeat(a_sums, b_samples).reshape(a_samples, b_samples),
            np.tile(b_sums, a_samples).reshape(a_samples, b_samples))
    nom_zeros = (0 == nom)
    denom_zeros = (0 == denom )
    denom[denom_zeros] = 1
    denom = np.power(denom, -0.5)
    denom[denom_zeros] = 0
    return np.multiply(nom, denom)


def self_cosine(mata):
    # type: (sparse.csc_matrix) -> np.ndarray
    """
    :param scipy.sparse.csc_matrix mata:
    :return numpy.ndarray :
    """
    # return pairwise_distances(mata, metric='cosine', n_jobs=1)
    samples = mata.shape[1]
    nom = (mata.transpose() * mata).todense()
    samples_sum = np.array(mata.power(2).sum(axis=0), dtype=np.float64)
    denom = np.multiply(np.tile(samples_sum, samples).reshape(samples, samples),
            np.repeat(samples_sum, samples).reshape(samples, samples))
    denom = np.power(denom, -0.5)
    return np.multiply(nom, denom)


def dice(mata, matb):
    # type: (sparse.csc_matrix, sparse.csc_matrix) -> np.ndarray
    """
    :param scipy.sparse.csc_matrix mata: (l features, m samples)
    :param scipy.sparse.csc_matrix matb: (l features, n samples)
    :return numpy.ndarray :
    """
    a_samples = mata.shape[1]
    b_samples = matb.shape[1]
    nom = 2 * (mata.transpose() * matb).todense()  # shape: MxN
    a_sums = np.array(mata.power(2).sum(axis=0), dtype=np.float64)  # m sums
    b_sums = np.array(matb.power(2).sum(axis=0), dtype=np.float64)  # n sums
    denom = np.repeat(a_sums, b_samples).reshape(a_samples, b_samples) + \
            np.tile(b_sums, a_samples).reshape(a_samples, b_samples)
    denom = np.power(denom, -1)
    return np.multiply(nom, denom)


def self_dice(mata):
    # type: (sparse.csc_matrix) -> np.ndarray
    """
    :param scipy.sparse.csc_matrix mata: (l features, m samples)
    :return numpy.ndarray :
    """
    samples = mata.shape[1]
    nom = 2 * (mata.transpose() * mata).todense()
    samples_sum = np.array(mata.power(2).sum(axis=0), dtype=np.float64)
    denom = np.tile(samples_sum, samples).reshape(samples, samples) + \
            np.repeat(samples_sum, samples).reshape(samples, samples)
    denom = np.power(denom, -1)
    return np.multiply(nom, denom)


def self_similarity(mat):
    # type: (sparse.csc_matrix) -> (np.ndarray, np.ndarray)
    """
    :rtype: (numpy.ndarray, numpy.ndarray)
    :param scipy.sparse.csc_matrix mat:
    """
    csim = self_cosine(mat)
    dsim = self_dice(mat)
    return (csim, dsim)


def similarity(mata, matb):
    # type: (sparse.csc_matrix, sparse.csc_matrix) -> Iterable[(str, np.ndarray)]
    """
    :param scipy.sparse.csc_matrix mata:
    :param scipy.sparse.csc_matrix matb:
    :return (numpy.ndarray, numpy.ndarray) (csim, dim):
    """
    yield ('cosine', cosine(mata, matb))
    yield ('dice', dice(mata, matb))


def log_self_similarity(mat, artifact_name):
    # type: (sparse.csc_matrix, str) -> object
    """
    :param str artifact_name:
    :param scipy.sparse.csc_matrix mat:
    """
    "calc self similarity"
    csim, dsim = self_similarity(mat)
    "calc self similarity done"
    log_np_data({'cosine_similarity': csim, 'dice_similarity': dsim}, artifact_name)


def log_similarity(sim_mat, key_name, artifact_name):
    # type: (numpy.ndarray, str, str) -> None
    """
    :rtype: (numpy.ndarray, numpy.ndarray)
    :param str artifact_name:
    :param scipy.sparse.csc_matrix mata:
    :param scipy.sparse.csc_matrix matb:
    """
    log_np_data({key_name: sim_mat}, artifact_name)
    pass


def target_similarity_compounds(t_df, mol_map, extra_details=''):
    # type: (pandas.DataFrame, pandas.DataFrame) -> object
    """
    :param pandas.DataFrame t_df:
    :param pandas.DataFrame mol_map:
    :param str extra_details:
    """
    tc_mat = compound_target_mat(t_df, mol_map)
    log_self_similarity(tc_mat, '%s compound based target similarity' % extra_details)


def target_similarity_cf(t_mat, extra_details=''):
    # type: (sparse.csc_matrix) -> object
    """
    :param scipy.sparse.csc_matrix t_mat:
    """
    t_cf_mat = t_mat.copy()
    t_cf_mat.data = np.ones_like(t_cf_mat.data)
    log_self_similarity(t_cf_mat, '%s chemical features based target similarity' % extra_details)


def tm_df_to_mat(tm_df):
    # type: (pd.DataFrame) -> sparse.csc_matrix
    """
    :return sparse.csc_matrix mat: molecules X targets sparse matrix
    :param pandas.DataFrame tm_df: First index is target id, second is molecule id
    """
    imi = np.array([np.array(val) for val in tm_df.index.values])
    target_ids = np.unique(imi[:, 0])
    mol_ids = np.unique(imi[:, 1])
    target_map = pd.DataFrame(data=np.arange(len(target_ids)), columns=['pos'], index=target_ids)
    mol_map = pd.DataFrame(data=np.arange(len(mol_ids)), columns=['pos'], index=mol_ids)
    mat = sparse.lil_matrix((mol_map.shape[0], target_map.shape[0]), dtype=np.bool)
    mat[mol_map.loc[imi[:, 1]].values, target_map.loc[imi[:, 0]].values] = True
    return mat.tocsr()


def mol_target_sim_pos(sim_mat, check_tm):
    # type: (np.ndarray, pd.DataFrame, pd.DataFrame) -> np.ndarray
    """
    :param numpy.ndarry sim_mat: Similarity matrix of M molecules X N targets
    :param pandas.DataFrame known_tm: known molecules targets relations
    :param pandas.DataFrame check_tm: unknown molecules targets relations for which we need to find the positions
    :rtype numpy.ndarray
    """
    c_mat = tm_df_to_mat(check_tm)
    c_mat = c_mat.multiply(sim_mat).tocsr()

    acc = [c_mat.getnnz(axis=1).reshape(-1)]
    while 0 < c_mat.getnnz():
        max_vec = c_mat.max(axis=1).todense()
        max_vec[0 == max_vec] = np.NINF
        max_pos = c_mat.argmax(axis=1).reshape(-1)
        acc.append(np.array((sim_mat > max_vec).sum(axis=1)).reshape(-1))
        sim_mat[np.arange(len(max_pos)), max_pos] = 0
        c_mat[np.arange(max_pos.size), max_pos] = 0
        c_mat.eliminate_zeros()
    # return acc
    return np.array(acc).transpose()


def tfidt_sim(tfmethod_name, q_mat, idf_vec, doc_mat):
    # type: (str, sparse.csc_matrix, sparse.csc_matrix, sparse.csc_matrix) -> Iterable[(str, np.ndarray)]
    """
    :rtype (numpy.ndarray, numpy.ndarray)
    :param str desc:
    :param str tfmethod_name:
    :param scipy.sparse.csc_matrix q_mat:
    :param scipy.sparse.csc_matrix  idf_vec:
    :param scipy.sparse.csc_matrix  doc_mat:
    """
    q_mat = tfmethods[tfmethod_name](q_mat)
    q_mat = q_mat.multiply(idf_vec)
    return similarity(q_mat, doc_mat)