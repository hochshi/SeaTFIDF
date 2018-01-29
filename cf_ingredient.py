from CMerModel import CMerModel, PandasParallerRunner, FunctionHolder, arrayHasher
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pandas as pd
from sacred import Ingredient
from itertools import combinations
from scipy import sparse

cf_ingredient = Ingredient('cf')
prunner = PandasParallerRunner()
imap = arrayHasher()


@cf_ingredient.config
def cfg():
    radii = {
        '0': True,
        '1': True,
        '2': True
    }
    use_counts = False
    mers = 1


def gen_mol(smiles):
    # type: (str) -> Chem.Mol
    if smiles is None:
        return None
    return Chem.MolFromSmiles(smiles)


def smiles_ecfc_mat(smiles, radii):
    # type: (str, list) -> pd.DataFrame
    mol = gen_mol(smiles)
    if mol is None:
        return np.nan

    rad_keys = sorted(radii.keys(), reverse=False)
    key = rad_keys.pop()
    while (not radii[key]) and (0 != len(rad_keys)):
        key = rad_keys.pop()
    state = True
    rd = AllChem.GetMorganFingerprint(mol, radius=int(key))
    in_indices = rd.GetNonzeroElements().keys()
    in_indices
    indices_w_count = pd.DataFrame.from_dict(rd.GetNonzeroElements(), orient='index') \
        .reset_index(level=0, inplace=False)
    indices_w_count.columns = ['feature', 'counts']
    for key in rad_keys:
        if state != radii[key]:
            state = radii[key]
            if state:
                in_indices = np.concatenate([in_indices, AllChem.GetMorganFingerprint(mol, radius=int(key)).GetNonzeroElements().keys()])
            else:
                in_indices = np.setdiff1d(in_indices,
                                          AllChem.GetMorganFingerprint(mol, radius=int(key)).GetNonzeroElements().keys())
    retval = pd.DataFrame(indices_w_count.query('feature in @in_indices'))
    return retval


@cf_ingredient.capture
def smiles_ecfc_mat_parallel(smiles_series, radii):
    # type: (pd.Series, list) -> pd.DataFrame
    func_holder = FunctionHolder(smiles_ecfc_mat, (radii,))
    return prunner.p_df(smiles_series, func_holder)


@cf_ingredient.capture
def gen_indices_map(cf_counts_dfs, use_counts, mers):
    # type: (list, bool, int) -> object
    if use_counts:
        for df in cf_counts_dfs:
            imap.hashArrayWithRep(df['feature'].values, mers)
    else:
        for df in cf_counts_dfs:
            imap.hashArray(df['feature'].values, mers)


def cf_to_sp_vec(indices, mers):
    # type: (list, int) -> sparse.csc_matrix
    if not np.any(np.isnan(indices)) and len(indices) > 0:
        hash_indices = []
        for tup in combinations(indices, mers):
            try:
                hash_indices.append(imap[tup])
            except KeyError:
                continue
        return sparse.csc_matrix((np.ones_like(hash_indices, dtype=np.int32), hash_indices, [0, len(hash_indices)]), shape=(imap.size, 1))
    return np.nan


def cf_mat_to_sp_vec(indices_mat, mers):
    # type: (np.array, int) -> sparse.csc_matrix
    if not np.any(np.isnan(indices_mat)) and indices_mat.size > 0:
        comb = combinations(np.repeat(indices_mat[:, 0], indices_mat[:, 1]), mers)
        hash_indices = np.array([imap.get(key, np.nan) for key in comb])
        indices, data = np.unique(hash_indices[~np.isnan(hash_indices)], return_counts=True)
        return sparse.csc_matrix((data, indices, [0, len(indices)]), shape=(imap.size, 1))
    return np.nan


@cf_ingredient.capture
def cf_df_to_sp_vec(cf_count_df, use_counts, mers):
    # type: (pd.DataFrame, bool, int) -> sparse.csc_matrix
    if np.any(pd.isna(cf_count_df)):
        return np.nan
    if use_counts:
        return cf_mat_to_sp_vec(cf_count_df.values, mers)
    return cf_to_sp_vec(cf_count_df['feature'].values, mers)


@cf_ingredient.capture
def cf_df_to_sp_vec_parallel(cf_count_series, use_counts, mers):
    func_holder = FunctionHolder(cf_df_to_sp_vec, (use_counts, mers, ))
    return prunner.p_df(cf_count_series, func_holder)


def create_target_sparse_vectors(target_mols, mols_df):
    # type: (list, pd.DataFrame) -> sparse.csc_matrix
    query_string = "%s in @target_mols" % CMerModel.mol_id
    mols_sp_vecs = mols_df.query(query_string)[CMerModel.sp_col].values
    return reduce(sparse.csc_matrix.__add__, mols_sp_vecs)


def create_target_sparse_vectors_parallel(target_mols_series, mols_df):
    func_holder = FunctionHolder(create_target_sparse_vectors, (mols_df,))
    return prunner.p_df(target_mols_series, func_holder)


def create_target_compound_vec(target_mols, mol_map):
    # type: (list, pd.DataFrame) -> sparse.csc_matrix
    return sparse.csc_matrix((np.ones_like(target_mols), mol_map.loc[target_mols].values.reshape(-1), [0, len(target_mols)]),
                             shape=(mol_map.shape[0], 1))


def create_target_compound_vec_parallel(target_mols_series, mol_map):
    func_holder = FunctionHolder(create_target_compound_vec, (mol_map,))
    return prunner.p_df(target_mols_series, func_holder)


def compound_target_mat(t_df, mol_map):
    """
    :rtype: scipy.sparse.csc_matrix
    :param pandas.DataFrame t_df:
    :param pandas.DataFrame mol_map:
    """
    # type: (pd.DataFrame, int) -> sparse.csc_matrix
    return sparse.hstack(create_target_compound_vec_parallel(t_df[CMerModel.target_mols], mol_map).values)
