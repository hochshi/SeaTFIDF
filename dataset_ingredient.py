from CMerModel import CMerModel, ppr_factory, FunctionHolder
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sacred import Ingredient

prunner = ppr_factory()
data_ingredient = Ingredient('dataset')


@data_ingredient.config
def cfg():
    files = {
        'mols': 'data/chembl_17_10uM_mol_data.csv',
        'targets': 'data/chembl_17_10uM_target_data.csv',
        'map': 'data/chembl_17_10uM_target_mol.csv'
    }
    c20files = {
        'mols': 'data/chembl_20_10uM_mol_data.csv',
        'targets': 'data/chembl_20_10uM_target_data.csv',
        'map': 'data/chembl_17_10uM_target_mol.csv'
    }
    c23files = {
        'mols': 'data/chembl_23_10uM_mol_data.csv',
        'targets': 'data/chembl_23_10uM_target_data.csv',
        'map': 'data/chembl_23_10uM_target_mol.csv'
    }
    max_radius = 2
    select_largest_frag = True


@data_ingredient.capture
def load_data(files, select_largest_frag):
    try:
        mol_data = pd.read_pickle(files['mols'].split('.')[0] + '.pd', compression='xz')
        target_data = pd.read_pickle(files['targets'].split('.')[0] + '.pd', compression='xz')
        tm_map = pd.read_pickle(files['map'].split('.')[0] + '.pd', compression='xz')
    except IOError:
        mol_data = pd.read_csv(files['mols'])
        target_data = pd.read_csv(files['targets'])
        tm_map = pd.read_csv(files['map'])

        mol_data[['MOLREGNO', 'MAX_PHASE', 'THERAPEUTIC']] = mol_data[
            ['MOLREGNO', 'MAX_PHASE', 'THERAPEUTIC']].apply(pd.to_numeric)
        target_data[['TARGET_ID']] = target_data[['TARGET_ID']].apply(pd.to_numeric)
        tm_map[['TARGET_ID', 'MOLREGNO', 'STANDARD_VALUE']] = tm_map[
            ['TARGET_ID', 'MOLREGNO', 'STANDARD_VALUE']].apply(pd.to_numeric)

        target_data = target_data.sort_values(['TARGET_ID']).set_index(['TARGET_ID'])
        target_data.index.names = ['TARGET_ID']

        mol_data = mol_data.sort_values(['MOLREGNO']).set_index(['MOLREGNO'])
        mol_data.index.names = ['MOLREGNO']
        if select_largest_frag:
            mol_data = get_largest_fragment(mol_data)
        mol_data[CMerModel.cf_df] = smiles_ecfc_mat_parallel(mol_data[CMerModel.frag_col])
        mol_data.dropna(axis=0, how='any', inplace=True)

        tm_map = tm_map.set_index(['TARGET_ID', 'MOLREGNO'])

    return (mol_data, target_data, tm_map)


def gen_mol(smiles):
    # type: (str) -> Chem.Mol
    if smiles is None:
        return None
    return Chem.MolFromSmiles(smiles)


def smiles_ecfc_mat(smiles, max_radius):
    # type: (str, list) -> pd.DataFrame
    mol = gen_mol(smiles)
    if mol is None:
        return np.nan

    rd = AllChem.GetMorganFingerprint(mol, radius=int(max_radius))
    feature_df = pd.DataFrame.from_dict(rd.GetNonzeroElements(), orient='index') \
        .reset_index(level=0, inplace=False)
    feature_df.columns = ['feature', 'counts']
    feature_df['radius'] = max_radius
    for radius in xrange(max_radius-1, -1, -1):
        indices = AllChem.GetMorganFingerprint(mol, radius=int(radius)).GetNonzeroElements().keys()
        rows = np.isin(feature_df['feature'].values, indices)
        feature_df.loc[rows, 'radius'] = radius
    return feature_df


@data_ingredient.capture
def smiles_ecfc_mat_parallel(smiles_series, max_radius):
    # type: (pd.Series, list) -> pd.DataFrame
    func_holder = FunctionHolder(smiles_ecfc_mat, (max_radius,))
    return prunner.p_df(smiles_series, func_holder)


def get_largest_fragment(mols_df):
    # type: (pd.DataFrame, pd.DataFrame, pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    mols_df[CMerModel.frag_col] = mols_df['CANONICAL_SMILES'].apply(get_smiles_largest_fragment)
    return mols_df


def get_smiles_largest_fragment(smiles):
    # type: (str) -> str
    return sorted(smiles.split('.'), key=lambda x: len(x), reverse=True)[0]


