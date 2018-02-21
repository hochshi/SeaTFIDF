from CMerModel import CMerModel, FunctionHolder, ppr_factory
import numpy as np
import pandas as pd
from sacred import Ingredient

prunner = ppr_factory()
filter_ingredient = Ingredient('filters')


@filter_ingredient.config
def cfg():
    action_list = [
        'filter_mol_by_target_num'
        'keep_single_mapping',
        'sanitize',
        'filter_radii'
    ]
    radii = {
        '0': True,
        '1': True,
        '2': True
    }
    filter_mol_by_target_num = {
        'cutoff': 21
    }
    filter_target_by_mol_num = {
        'cutoff': 0
    }
    filter_target_by_drug_num = {
        'cutoff': 0,
        'max_phase': 0
    }
    filter_mols_by_phase = {
        'max_phase': 0
    }


# @filter_ingredient.capture
# def add_position(mols_df, targets_df, tm_map):
#     # type: (pd.DataFrame, pd.DataFrame, pd.DataFrame) -> tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame)
#     mols_df.set_index([np.arange(mols_df.shape[0]), mols_df.index.get_level_values(level='MOLREGNO')], inplace=True)
#     mols_df.index.names = ['POSITION', 'MOLREGNO']
#     targets_df.set_index([np.arange(targets_df.shape[0]), targets_df.index.get_level_values(level='TARGET_ID')], inplace=True)
#     targets_df.index.names = ['POSITION', 'TARGET_ID']
#     return (mols_df, targets_df, tm_map)

@filter_ingredient.capture
def keep_single_mapping(mols_df, targets_df, tm_map):
    # type: (pd.DataFrame, pd.DataFrame, pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    mols_df = mols_df[~mols_df.index.duplicated(keep='first')]
    targets_df = targets_df[~targets_df.index.duplicated(keep='first')]
    tm_map = tm_map[~tm_map.index.duplicated(keep='first')]
    return (mols_df, targets_df, tm_map)


@filter_ingredient.capture
def sanitize_data(mols_df, targets_df, tm_map):
    # type: (pd.DataFrame, pd.DataFrame, pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    mol_ids = mols_df.index.get_level_values(level=CMerModel.mol_id)
    target_ids = targets_df.index.get_level_values(level=CMerModel.target_id)
    tm_map = pd.DataFrame(tm_map.query('%s in @mol_ids' % CMerModel.mol_id).query('%s in @target_ids' % CMerModel.target_id))
    mol_ids = np.intersect1d(mol_ids, tm_map.index.get_level_values(level=CMerModel.mol_id))
    target_ids = np.intersect1d(target_ids, tm_map.index.get_level_values(level=CMerModel.target_id))
    mols_df = pd.DataFrame(mols_df.query('%s in @mol_ids' % CMerModel.mol_id))
    targets_df = pd.DataFrame(targets_df.query('%s in @target_ids' % CMerModel.target_id))
    return (mols_df, targets_df, tm_map)


@filter_ingredient.capture(prefix='filter_mol_by_target_num')
def filter_mol_by_target_num(mols_df, targets_df, tm_map, cutoff):
    # type: (pd.DataFrame, pd.DataFrame, pd.DataFrame, int) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    tm_map = pd.DataFrame(tm_map.groupby(level=CMerModel.mol_id).filter(
        lambda x: len(np.unique(x.index.get_level_values(level=CMerModel.target_id))) < cutoff
    ))
    return (mols_df, targets_df, tm_map)


@filter_ingredient.capture(prefix='filter_target_by_mol_num')
def filter_target_by_mol_num(mols_df, targets_df, tm_map, cutoff):
    # type: (pd.DataFrame, pd.DataFrame, pd.DataFrame, int) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    tm_map = pd.DataFrame(tm_map.groupby(level=CMerModel.target_id).filter(
        lambda x: len(np.unique(x.index.get_level_values(level=CMerModel.mol_id))) > cutoff
    ))
    return (mols_df, targets_df, tm_map)


@filter_ingredient.capture(prefix='filter_target_by_drug_num')
def filter_target_by_drug_num(mols_df, targets_df, tm_map, cutoff, max_phase):
    # type: (pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    tmp_mols = pd.DataFrame(mols_df['MAX_PHASE'])
    tm_map = tmp_mols.join(tm_map, how='inner')
    tm_map = pd.DataFrame(tm_map.query('MAX_PHASE >= @max_phase').groupby(level=CMerModel.target_id).filter(
        lambda x: len(np.unique(x.index.get_level_values(level=CMerModel.mol_id))) > cutoff
    ))
    tm_map.drop(columns=['MAX_PHASE'], inplace=True)
    return (mols_df, targets_df, tm_map)


@filter_ingredient.capture(prefix='filter_mols_by_phase')
def filter_mols_by_phase(mols_df, targets_df, tm_map, max_phase):
    # type: (pd.DataFrame, pd.DataFrame, pd.DataFrame, int) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    mols_df = pd.DataFrame(mols_df.query('MAX_PHASE >= @max_phase'))
    return (mols_df, targets_df, tm_map)


@filter_ingredient.capture
def filter_radii(feature_df, radii):
    # type: (pd.DataFrame, dict) -> pd.DataFrame
    keep_radii = []
    for radius, keep in radii.iteritems():
        if keep:
            keep_radii.append(radius)
    return pd.DataFrame(feature_df.query('radius in @keep_radii'))


@filter_ingredient.capture
def filter_radii_parallel(feature_df_series, radii):
    # type: (pd.Series, dict) -> pd.DataFrame
    func_holder = FunctionHolder(filter_radii, (radii,))
    return prunner.p_df(feature_df_series, func_holder)


methods = {
    # 'add_position': add_position,
    'filter_radii': filter_radii_parallel,
    'keep_single_mapping': keep_single_mapping,
    'sanitize': sanitize_data,
    'filter_mol_by_target_num': filter_mol_by_target_num,  # requires a cutoff keyword argument
    'filter_target_by_mol_num': filter_target_by_mol_num,  # requires a cutoff keyword argument
    'filter_target_by_drug_num': filter_target_by_drug_num,  # requires a cutoff and max_phase keyword argument
    'filter_mols_by_phase': filter_mols_by_phase  # requires a max_phase (between 0 and 4) keyword argument
}


@filter_ingredient.capture
def filter_data(mol_data, target_data, tm_map, action_list):
    # type: (pd.DataFrame, pd.DataFrame, pd.DataFrame, list) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    for func in action_list:
        # print "running %s" % (str(func))
        mol_data, target_data, tm_map = methods[func](mol_data, target_data, tm_map)

    return (mol_data, target_data, tm_map)
