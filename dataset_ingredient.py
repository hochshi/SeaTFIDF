import pandas as pd
import numpy as np
from sacred import Ingredient

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


@data_ingredient.capture
def load_data(files):
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

    tm_map = tm_map.set_index(['TARGET_ID', 'MOLREGNO'])

    return (mol_data, target_data, tm_map)