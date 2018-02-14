#!/usr/bin/env python
# How to run with settings override:
#
# from tfidf_experiment import ex
# filters = [
#   'filter_target_by_drug_num',
#   'keep_single_mapping',
#   'sanitize',
#   'smiles_largest_frag'
# ]
# config_updates = {
#         'filters.action_list': filters,
#         'filters.filter_target_by_drug_num': {'cutoff': 9, 'max_phase':4}
#     }
# r = ex.run(
#     config_updates = config_updates
# )


from sacred import Experiment
from dataset_ingredient import data_ingredient, load_data
from filter_ingredient import filter_ingredient, filter_data, sanitize_data
from cf_ingredient import cf_ingredient, gen_indices_map, imap, cf_df_to_sp_vec_parallel, create_target_sparse_vectors_parallel
from log_ingredient import log_ingredient, log_data_structure, log_np_dict
from CMerModel import CMerModel
from sacred.observers import MongoObserver
import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.model_selection import KFold
from similarity_measures import target_similarity_compounds, target_similarity_cf, tfidt_sim, mol_target_sim_pos
from weights import recommended_weighting_schemes, tfmethods, idfmethods
import sys
from typing import Iterable


class DataSourceHolder:

    name = ''
    mol_mat = None
    tm = None
    ut_selector = None

    def __init__(self, name, mol_mat, tm, ut_selector):
        # type: (str, sparse.csc_matrix, pd.DataFrame, Iterable[slice]) -> None
        self.name, self.mol_mat, self.tm, self.ut_selector = name, mol_mat, tm, ut_selector


ex = Experiment('tfidf_experiment', ingredients=[data_ingredient, filter_ingredient, cf_ingredient, log_ingredient], interactive=True)
ex.observers.append(MongoObserver.create())


@ex.config
def config():
    mol_id = 'MOLREGNO'
    target_id = 'TARGET_ID'
    kfcv = False


@ex.capture
def set_model_params(mol_id, target_id):
    CMerModel.mol_id = mol_id
    CMerModel.target_id = target_id

def mols_per_target(target_df, tm_df):
    # type: (pd.DataFrame, pd.DataFrame) -> pd.Series
    """
    :param pandas.DataFrame target_df:
    :param pandas.DataFrame  tm_df:
    :return pandas.Series :
    :rtype pandas.DataFrame
    """
    return target_df.index.get_level_values(CMerModel.target_id).map(
        lambda x: np.unique(tm_df.xs(x, level=CMerModel.target_id).index.values)
    )


def add_mol_sp_vec_col(target_df, mols_df, tm_df):
    # type: (pd.DataFrame, pd.DataFrame, pd.DataFrame) -> pd.DataFrame
    """
    :param pandas.DataFrame mols_df:
    :param pandas.DataFrame target_df:
    :param pandas.DataFrame  tm_df:
    :return pandas.DataFrame :
    :rtype: pandas.DataFrame
    """
    target_df.loc[:, CMerModel.target_mols] = mols_per_target(target_df, tm_df)
    target_df.loc[:, CMerModel.sp_col] = create_target_sparse_vectors_parallel(target_df[CMerModel.target_mols],
                                                                               mols_df)
    return target_df


def curate_data_set(mols, targets, tm, gen_map=False):
    mols, targets, tm = filter_data(mols, targets, tm)
    if gen_map:
        imap.reset()
        gen_indices_map(mols[CMerModel.cf_df].values)
    mols[CMerModel.sp_col] = cf_df_to_sp_vec_parallel(mols[CMerModel.cf_df])
    mols, targets, tm = sanitize_data(mols, targets, tm)
    return (mols, targets, tm)


def prepare_data(mols, targets, tm, fold=None):
    targets = add_mol_sp_vec_col(targets, mols, tm)
    t_mat = sparse.hstack(targets[CMerModel.sp_col].values)
    log_data_structure(mols, targets, tm, imap, "C17", fold)
    return (mols, targets, tm, t_mat)


def log_similarity(q_tf, q_mat, q_idf, doc_term, artifact_name, map_df=None, log_sim_pos=False):
    # type: (str, sparse.csc_matrix, sparse.csc_matrix, sparse.csc_matrix, str, pd.DataFrame, bool) -> None
    sim_gen = tfidt_sim(q_tf, q_mat, q_idf, doc_term)
    for sim_name, sim_mat in sim_gen:
        key_name = '%s_similarity_matrix' % sim_name
        log_np_dict({key_name: sim_mat}, sim_name + ' '+ artifact_name)
        if log_sim_pos:
            key_name = '%s_sim_pos' % sim_name
            log_np_dict({key_name: mol_target_sim_pos(sim_mat, map_df)}, sim_name + ' similarity positions ' + artifact_name)


def prep_rscheme_data(rscheme, t_mat):
    # type: (dict, sparse.csc_matrix) -> object
    doc_tf, doc_idf = tfmethods[rscheme['doc']['tf']](t_mat), idfmethods[rscheme['doc']['idf']](t_mat)
    q_idf = idfmethods[rscheme['query']['idf']](t_mat)
    t_doc = doc_tf.multiply(doc_idf)
    return (t_doc, q_idf)


def rscheme_similarity(rscheme, t_mat, data_sources):
    # type: (dict, sparse.csc_matrix, Iterable[DataSourceHolder]) -> None

    t_doc, q_idf = prep_rscheme_data(rscheme, t_mat)

    log_similarity(rscheme['query']['tf'], t_mat, q_idf, t_doc, "TFIDF Target similarity, doc:%s*%s, query:%s*%s" % (
        rscheme['doc']['tf'], rscheme['doc']['idf'],
        rscheme['query']['tf'], rscheme['query']['idf']))

    for ds in data_sources:
        log_similarity(rscheme['query']['tf'], ds.mol_mat, q_idf, t_doc[ds.ut_selector],
                       "%s TFIDF %d Targets %d Compound similarity, doc:%s*%s, query:%s*%s" % (ds.name,
                           t_doc.shape[1], ds.mol_mat.shape[1], rscheme['doc']['tf'], rscheme['doc']['idf'],
                           rscheme['query']['tf'],
                           rscheme['query']['idf']), ds.tm, True)

@ex.automain
def run(kfcv, _run, _rnd, _config):
    # type: (bool, sacred.run.Run, np.random.RandomState, dict) -> None
    """
    :param dict _config:
    :param bool kfcv: Injected by sacred, should we run k-fold cross validation
    :param sacred.run.Run _run: The run object inject by sacred
    :param numpy.random.RandomState _rnd: The random state created by sacred
    """
    sim_mats_keys = ['cosine_similarity', 'dice_similarity']
    set_model_params()
    c17mols, c17targets, c17tm = load_data(_config['dataset']['files'])
    c17mols, c17targets, c17tm = curate_data_set(c17mols, c17targets, c17tm, gen_map=True)

    if kfcv:
        kf = KFold(n_splits=10, shuffle=True, random_state=_rnd)
        mol_locs = np.arange(c17mols.shape[0])
        for fold_no, (train_index, test_index) in enumerate(kf.split(mol_locs)):
            train_mols, test_mols = c17mols.iloc[train_index, :], c17mols.iloc[test_index, :]
            train_mols, train_targets, train_tm = sanitize_data(train_mols, c17targets, c17tm)
            test_mols, test_targets, test_tm = sanitize_data(test_mols, c17targets, c17tm)
            train_mols, train_targets, train_tm, train_t_mat = prepare_data(train_mols, train_targets, train_tm, fold_no)

            log_data_structure(train_mols, train_targets, train_tm, imap, "Fold no.%d train" % fold_no, None)

            train_mol_map = pd.DataFrame(data=np.arange(train_mols.shape[0]), columns=['idx'], index=train_mols.index)
            target_similarity_compounds(train_targets, train_mol_map, "Fold no.%d train")
            target_similarity_cf(train_t_mat, "Fold no.%d train")
            train_m_mat = sparse.hstack(train_mols[CMerModel.sp_col].values)
            train_target_ids = pd.DataFrame(data=np.arange(len(train_targets.index.values)), columns=['idx'],
                                         index=train_targets.index.values)

            test_tm = test_tm.query("%s in @train_target_ids.index.values" % CMerModel.target_id)
            unknown_mappings = np.setdiff1d(test_tm.index.values, train_tm.index.values)
            test_tm = test_tm.loc[unknown_mappings, :]
            test_mols, test_targets, test_tm = curate_data_set(test_mols, test_targets, test_tm)
            log_data_structure(test_mols, test_targets, test_tm, imap, "Fold no.%d test" % fold_no , None)
            test_m_mat = sparse.hstack(test_mols[CMerModel.sp_col].values)

            for rscheme in recommended_weighting_schemes:
                doc_tf, doc_idf = tfmethods[rscheme['doc']['tf']](train_t_mat), idfmethods[rscheme['doc']['idf']](train_t_mat)
                q_idf = idfmethods[rscheme['query']['idf']](train_t_mat)
                t_doc = doc_tf.multiply(doc_idf)

                tfidt_sim(rscheme['query']['tf'], train_t_mat, q_idf, t_doc,
                          "TFIDF Train Fold %d Target similarity, doc:%s*%s, query:%s*%s" % ( fold_no,
                              rscheme['doc']['tf'], rscheme['doc']['idf'],
                              rscheme['query']['tf'], rscheme['query']['idf']))

                csim, dsim = tfidt_sim(rscheme['query']['tf'], train_m_mat, q_idf, t_doc,
                                       "C17 Train Fold %d TFIDF %d Targets %d Compound similarity, doc:%s*%s, query:%s*%s" % ( fold_no,
                                           t_doc.shape[1], train_m_mat.shape[1], rscheme['doc']['tf'],
                                           rscheme['doc']['idf'],
                                           rscheme['query']['tf'],
                                           rscheme['query']['idf']))
                csp = mol_target_sim_pos(csim, train_tm)
                dsp = mol_target_sim_pos(dsim, train_tm)
                log_np_dict({'cosine_sim_pos': csp, 'dice_sim_pos': dsp},
                            "C17 Train Fold %d TFIDF doc:%s*%s, query:%s*%s similarity positions" % (fold_no,
                                rscheme['doc']['tf'], rscheme['doc']['idf'],
                                rscheme['query']['tf'],
                                rscheme['query']['idf']))

                test_ut = train_target_ids.loc[test_targets.index.values].values.reshape(-1)

                csim, dsim = tfidt_sim(rscheme['query']['tf'], test_m_mat, q_idf, t_doc[:, test_ut],
                                       "C17 Test Fold %d TFIDF %d Targets %d Compound similarity, doc:%s*%s, query:%s*%s" % (fold_no,
                                           t_doc[:, test_ut].shape[1], test_m_mat.shape[1], rscheme['doc']['tf'],
                                           rscheme['doc']['idf'],
                                           rscheme['query']['tf'],
                                           rscheme['query']['idf']))
                csp = mol_target_sim_pos(csim, test_tm)
                dsp = mol_target_sim_pos(dsim, test_tm)
                log_np_dict({'cosine_sim_pos': csp, 'dice_sim_pos': dsp},
                            "C17 Test Fold %d TFIDF doc:%s*%s, query:%s*%s similarity positions" % (fold_no,
                                rscheme['doc']['tf'], rscheme['doc']['idf'],
                                rscheme['query']['tf'],
                                rscheme['query']['idf']))
    else:

        data_sources = []

        c17mols, c17targets, c17tm, c17t_mat = prepare_data(c17mols, c17targets, c17tm)
        mol_map = pd.DataFrame(data=np.arange(c17mols.shape[0]), columns=['idx'], index=c17mols.index)
        target_similarity_compounds(c17targets, mol_map)
        target_similarity_cf(c17t_mat)
        c17m_mat = sparse.hstack(c17mols[CMerModel.sp_col].values)
        c17target_ids = pd.DataFrame(data=np.arange(len(c17targets.index.values)), columns=['idx'], index=c17targets.index.values)

        data_sources.append(DataSourceHolder('C17', c17m_mat, c17tm, (slice(None), slice(None))))

        c20mols, c20targets, c20tm = load_data(_config['dataset']['c20files'])
        c20tm = c20tm.query("%s in @c17target_ids.index.values" %CMerModel.target_id)
        unknown_mappings = np.setdiff1d(c20tm.index.values, c17tm.index.values)
        c20tm = c20tm.loc[unknown_mappings, :]
        c20mols, c20targets, c20tm = curate_data_set(c20mols, c20targets, c20tm)
        log_data_structure(c20mols, c20targets, c20tm, imap, "C20", None)
        c20m_mat = sparse.hstack(c20mols[CMerModel.sp_col].values)

        c20ut = c17target_ids.loc[c20targets.index.values].values.reshape(-1)
        data_sources.append(DataSourceHolder('C20', c20m_mat, c20tm, (slice(None), c20ut)))

        c23mols, c23targets, c23tm = load_data(_config['dataset']['c20files'])
        c23tm = c23tm.query("%s in @c17target_ids.index.values" %CMerModel.target_id)
        unknown_mappings = np.setdiff1d(c23tm.index.values, c17tm.index.values)
        c23tm = c23tm.loc[unknown_mappings, :]
        c23mols, c23targets, c23tm = curate_data_set(c23mols, c23targets, c23tm)
        log_data_structure(c23mols, c23targets, c23tm, imap, "C23", None)
        c23m_mat = sparse.hstack(c23mols[CMerModel.sp_col].values)

        c23ut = c17target_ids.loc[c23targets.index.values].values.reshape(-1)
        data_sources.append(DataSourceHolder('C23', c23m_mat, c23tm, (slice(None), c23ut)))

        for rscheme in recommended_weighting_schemes:
            rscheme_similarity(rscheme, c17t_mat, data_sources)
    sys.exit(0)
