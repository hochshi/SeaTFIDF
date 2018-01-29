from itertools import product
mers_dict = [{key:val} for key, val in list(product(['cf.mers'], [1,2]))]
counts_dict = [{key:val} for key, val in list(product(['cf.use_counts'], [True, False]))]

vals = [True, False]
acc = []
for i in vals:
    for j in vals:
        for k in vals:
            acc.append((i, j, k))

radii_dict = [ {key: val} for key, val in list(product(['cf.radii'], [{'0': val1, '1':val2, '2': val3} for val1, val2, val3 in acc]))]

target_num_cutoff = [{'filters.filter_mol_by_target_num': val} for val in [{key: val} for key, val in list(product(['cutoff'], [2, 3, 6, 11, 21]))]]
mol_num_cutoff = {'filter_target_by_mol_num': {'cutoff': 9}}

filters_action_list = [
        'filter_mol_by_target_num'
        'keep_single_mapping',
        'sanitize',
        'smiles_largest_frag'
    ]

target_filter_action_list = [
        'filter_mol_by_target_num',
        'filter_target_by_mol_num',
        'keep_single_mapping',
        'sanitize',
        'smiles_largest_frag'
    ]

config_updates = {
    'filters.action_list': filters_action_list,
    'filters.filter_mol_by_target_num': {'cutoff': 21}
}


def yield_config():
    gen = product([True, False], target_num_cutoff, radii_dict, counts_dict, mers_dict)
    for cfg_comb in gen:
        cfg = {}
        if cfg_comb[0]:
            cfg = {'filters.action_list': target_filter_action_list, 'filters.filter_target_by_mol_num': {'cutoff': 9}}
        for d in cfg_comb[1:]:
            cfg.update(d)
        yield cfg