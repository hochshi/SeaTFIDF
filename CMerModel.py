from __future__ import division
from wtc_utilities import *
import collections
from multiprocessing import cpu_count
from pathos.pools import ProcessPool, ThreadPool
from itertools import product
import crcmod
import numpy as np
import pandas as pd
from itertools import combinations, combinations_with_replacement
from functools import reduce
from scipy import sparse


class FunctionHolder:
    
    func = None
    args = None
    
    def __init__(self, func, args):
        self.func = func
        self.args = args

class PandasParallelRunner:
    
    num_partitions = 10 #number of partitions to split dataframe
    num_cores = cpu_count() - 1 #number of cores on your machine
    pool = None
    
    def __init__(self):
        self.num_partitions = self.num_cores
        # self.pool = ThreadPool(self.num_cores)
        self.pool = ProcessPool(self.num_cores)
        
    def p_arr_run(self, tup):
        data, func_holder = tup
        for i, v in enumerate(data):
            data[i] = func_holder.func(v, *func_holder.args)
        return data
    
    def p_arr(self, arr, func_holder):
        arr_split = np.array_split(arr, self.num_partitions)
        arr = np.concatenate(self.pool.map(self.p_arr_run, product(arr_split, [func_holder])))
        return arr
        
    def p_df_run(self, tup):
        data, func_holder = tup
        return data.apply(func_holder.func, args=func_holder.args)

    def p_df(self, df, func_holder):
        df_split = np.array_split(df, self.num_partitions)
        df = pd.concat(self.pool.map(self.p_df_run, product(df_split, [func_holder])))
        return df


def ppr_factory(_singlton = PandasParallelRunner()):
    # type: (PandasParallelRunner) -> PandasParallelRunner
    return _singlton

class arrayHasher:
    
    size = 0
    tupToPos = {}
    posToTup = {}
    seen_pos = 0
    pos_pos = 1
    
    def __init__(self, start_pos=0):
        self.size = start_pos

    def reset(self):
        self.size = 0
        self.tupToPos = {}
        self.posToTup = {}
        
    def hashArray(self, arr, mers):
        for tup in combinations(arr, mers):
            try:
                val = self.tupToPos[tup]
            except KeyError:
                self.tupToPos[tup] = [1, self.size]
                self.posToTup[self.size] = tup
                self.size += 1
                continue
            val[self.seen_pos] += 1
    
    def hashArrayWithRep(self, arr, mers):
        for tup in combinations_with_replacement(arr, mers):
            try:
                val = self.tupToPos[tup]
            except KeyError:
                self.tupToPos[tup] = [1, self.size]
                self.posToTup[self.size] = tup
                self.size += 1
                continue
            val[self.seen_pos] += 1
    
    def __getitem__(self, key):
        try:
            return self.tupToPos[key][self.pos_pos]
        except KeyError:
            return self.posToTup[key]
        
    def get(self, key, dflt):
        try:
            return self.tupToPos[key][self.pos_pos]
        except KeyError:
            pass
            try:
                return self.posToTup[key]
            except KeyError:
                return dflt

class CMerModel:
    mers=2
    mers_upto=False
    frag_col = 'LARGEST_FRAGMENT'
    sp_col = 'SPARSE_VECTOR'
    mol_id = 'MOLECULE_CHEMBL_ID'
    target_id = 'TARGET_CHEMBL_ID'
    mer_key = 'CRC'
    mer_val = 'MER_POS'
    idf_col = 'IDF'
    res_cor = 'CORRECT'
    res_tot = 'TOTAL'
    target_mols = 'TARGET_MOLS'
    cf_df = 'FEATURE_DF'
    
    crcfunc = staticmethod(crcmod.predefined.mkPredefinedCrcFun('crc-64'))
    
    prunner = ppr_factory()
    
    imap = arrayHasher()
    
    @staticmethod
    def getLargestFragment(smiles):
        return sorted(smiles.split('.'),key=lambda x: len(x),reverse=True)[0]

    @classmethod
    def smiles_ecfc1_sparse_vector(cls, smiles, mer_pos_df):
        rd1_unique_indices = cls.smiles_ecfc1_unique_indices(smiles)
        if not np.any(np.isnan(rd1_unique_indices)) and len(rd1_unique_indices) > 0:
            indices = []
            for idx in cls.hashindices(rd1_unique_indices):
                try:
                    indices.append(mer_pos_df[idx])
                except KeyError:
                    continue
            return sparse.csc_matrix((np.ones_like(indices, dtype=np.int32), indices, [0, len(indices)]), shape=(len(mer_pos_df), 1))
        return np.nan

    @staticmethod
    def gen_mol(smiles):
        if smiles is None:
            return None
        return Chem.MolFromSmiles(smiles)

    @classmethod
    def smiles_ecfc_mat(cls, smiles, radii):
        mol = cls.gen_mol(smiles)
        if mol is None:
            return None

        rad_keys = sorted(radii.keys(), reverse=False)
        while (not radii[key]) and (0 != len(rad_keys)):
            key = rad_keys.pop()
        state = True
        rd = AllChem.GetMorganFingerprint(mol, key)
        in_indices = rd.GetNonzeroElements().keys()
        indices_w_count = pd.DataFrame.from_dict(rd.GetNonzeroElements(), orient='index') \
            .reset_index(level=0, inplace=False)
        indices_w_count.columns = ['feature', 'count']
        for key in rad_keys:
            if state != radii[key]:
                state = radii[key]
                if state:
                    in_indices = in_indices + AllChem.GetMorganFingerprint(mol, key).GetNonzeroElements().keys()
                else:
                    in_indices = np.setdiff1d(in_indices, AllChem.GetMorganFingerprint(mol, key).GetNonzeroElements().keys())
        return indices_w_count.query('feature in @in_indices')

    @staticmethod
    def smiles_ecfc1_unique_indices(smiles):
        rd1 = smile2rdkit(smiles, 1)
        if rd1 is not None:
            rd0 = smile2rdkit(smiles, 0)
            return np.setdiff1d(rd1.GetNonzeroElements().keys(), rd0.GetNonzeroElements().keys())
        return np.nan
    
    @classmethod
    def mols_df_indices(cls, mols_df):
        func_holder = FunctionHolder(CMerModel.smiles_ecfc1_unique_indices, ())
        indices = cls.prunner.p_df(mols_df[cls.frag_col], func_holder)
        return indices

    @classmethod
    def create_mer_pos_df(cls, mols_df):
        indices = cls.mols_df_indices(mols_df)
        cls.gen_indices_map(indices.dropna().values)
        return indices

    @classmethod
    def gen_indices_map(cls, indices, withRep=False):
        if withRep:
            for arr in indices:
                cls.imap.hashArrayWithRep(arr, cls.mers)
        else:
            for arr in indices:
                cls.imap.hashArray(arr, cls.mers)

    # create_sparse_vectors
    @classmethod
    def create_sparse_vectors(cls, mols_df):
        func_holder = FunctionHolder(cls.smiles_ecfc1_unique_indices, ())
        sp_indices = CMerModel.prunner.p_df(mols_df[CMerModel.frag_col], func_holder)
        return [cls.indices_to_sp_vec(val) for val in sp_indices.values]

    @classmethod
    def indices_to_sp_vec(cls, indices):
        if not np.any(np.isnan(indices)) and len(indices) > 0:
            hash_indices = []
            for tup in combinations(indices, cls.mers):
                try:
                    hash_indices.append(cls.imap[tup])
                except KeyError:
                    continue
            return sparse.csc_matrix((np.ones_like(hash_indices, dtype=np.int32), hash_indices, [0, len(hash_indices)]), shape=(cls.imap.size, 1))
        return np.nan

    @classmethod
    def indicesmat_to_sp_vec(cls, indices_mat):
        if not np.any(np.isnan(indices_mat)) and indices_mat.size > 0:
            comb = combinations(np.repeat(indices_mat[:, 0], indices_mat[:, 1]), cls.mers)
            hash_indices = np.array([cls.imap.get(key, np.nan) for key in comb])
            indices, data = np.unique(hash_indices[~np.isnan(hash_indices)], return_counts=True)
            return sparse.csc_matrix((data, indices, [0, len(indices)]), shape=(cls.imap.size, 1))
        return np.nan

    @classmethod
    def create_target_sparse_vectors(cls, target_mols, mols_df):
        query_string = "%s in @target_mols" % cls.mol_id
        mols_sp_vecs = mols_df.query(query_string)[cls.sp_col].values
        return reduce(sparse.csc_matrix.__add__, mols_sp_vecs)

    @classmethod
    def create_tm_idf_df(cls, targets_df, tf_func, idf_func, norm_func):
        # The following explanation is in order to relate this to document-corpus example.
        # In our case (protein targets and small molecules):
        #   Document = protein target.
        #   Words = chemical features extracted using ECFCX (where X denotes the radius parameter)
        #     
        # tm - Target sparse matrix
        # tfc - Target feature counter
        # tfd - Target feature dataframe
        
        tm = sparse.hstack(targets_df[cls.sp_col].values)
        # Set document term weight to 1+log(term_freq)
        # tm.data = 1 + np.log(tm.data)
        tf_func(tm)
        
        # Count the number of occurences of each term (term<->vector row index)
        # tfc = collections.Counter(tm.indices)
        
        # Set the query term weight to log(1+(number of documents)/(number of occurences of term t in the corpus))
        # In our case: 
        #   the query term weight is log(1+ (number of proteins)/(number of occurences of chemical feature t))
        # In vector format this is log(1+(number of columns)/(number of occurences of row index t))
        # tfd = {key: np.log(1 + tm.shape[1] / val) for key, val in
        #                        tfc.iteritems()}
        idf = idf_func
        idf_df = pd.DataFrame.from_dict(idf, orient='index')
        idf_df.columns = [cls.idf_col]

        # Normalize each column vector
        # for i in xrange(len(tm.indptr) - 1):
        #     tm.data[tm.indptr[i]:tm.indptr[i + 1]] = tm.data[tm.indptr[i]:
        #     tm.indptr[i + 1]] / np.linalg.norm(
        #         tm.data[tm.indptr[i]:tm.indptr[i + 1]])
        norm_func(tm)
        return (tm, idf_df)

    
    @classmethod
    def create_mol_idf_mat(cls, mols_df, indices_dict):
        idf_mat = sparse.hstack(mols_df[cls.sp_col].values)
        idf_mat.data = np.zeros_like(idf_mat.data, dtype=np.float32)
        for i in xrange(idf_mat.shape[1]):
            start = idf_mat.indptr[i]
            end = idf_mat.indptr[i + 1]
            col_idf = []
            for idx in idf_mat.indices[start:end]:
                try:
                    col_idf.append(indices_dict[idx])
                except KeyError:
                    col_idf.append(0)
            norm = np.linalg.norm(col_idf)
            if 0 != norm:
                idf_mat.data[start:end] = col_idf / np.linalg.norm(col_idf)
        return idf_mat.transpose().tocsr()

    @staticmethod
    def dm_calc_cosine(q_vec, doc_mat):
        return q_vec.dot(doc_mat).toarray()

    def calc_results_for_boundry(res_mat, boundry, mol2target_df, mols_df, targets_df, cons=True):
        total_targets = 0
        hit_correct = 0
        boundry_mat = np.argpartition(res_mat, -boundry, axis=1)[:,-boundry:]
        for pos in xrange(boundry_mat.shape[0]):
            mol_targets = mol2target_df.loc[mols_df.loc[(slice(None), pos),'MOLECULE_CHEMBL_ID'], 'TARGET_CHEMBL_ID'][0]
            predicted_targets = targets_df.loc[(slice(None), boundry_mat[pos,:]),].index.get_level_values(0).values
            if len(mol_targets) > boundry:
                if cons:
                    continue
                else:
                    total_targets = total_targets + boundry
            else:
                total_targets = total_targets + len(mol_targets)
            hit_correct = hit_correct + np.count_nonzero(np.isin(mol_targets, predicted_targets))
        return {'correct':hit_correct, 'total':total_targets}