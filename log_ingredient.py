from sacred import Ingredient
from tempfile import NamedTemporaryFile
import numpy as np
from CMerModel import CMerModel
from numpy import savez_compressed as savez

log_ingredient = Ingredient('log')


@log_ingredient.capture
def log_np_data(kwdict, artifact_name, _run):
    # type: (dict, str, sacred.run.Run) -> object
    """
    :param dict kwdict:
    :param str artifact_name:
    :param sacred.run.Run _run:
    """
    outfile = NamedTemporaryFile()
    savez(outfile, **kwdict)
    _run.add_artifact(outfile.name, artifact_name)
    outfile.close()


@log_ingredient.capture
def save_array_histogram(arr, artifact_name):
    """
    :param numpy.ndarray arr:
    :param str artifact_name:
    """
    density, density_edges = np.histogram(arr, density=True, bins='auto')
    count, count_edges = np.histogram(arr, density=False, bins='auto')
    log_np_data({'density': density, 'density_edges': density_edges, 'count': count,
             'count_edges': count_edges}, artifact_name)


@log_ingredient.capture
def log_data_structure(mols, targets, tm, imap, data_desc, fold, _run ):
    """
    :param str data_desc:
    :param int fold:
    :param pandas.DataFrame mols:
    :param pandas.DataFrame targets:
    :param pandas.DataFrame tm:
    :param sacred.run.Run _run:
    :param CMerModel.arrayHasher imap:

    """
    _run.log_scalar('%s%s Number of target compound mappings' % (data_desc, str(fold) if fold is not None else ''), tm.shape[0])
    _run.log_scalar('%s%s Number of targets' % (data_desc, str(fold) if fold is not None else ''), targets.shape[0])
    _run.log_scalar('%s%s Number of compounds' % (data_desc, str(fold) if fold is not None else ''), mols.shape[0])
    _run.log_scalar('%s%s Number of chemical features' % (data_desc, str(fold) if fold is not None else ''), imap.size)

    save_array_histogram(tm.STANDARD_RELATION.groupby(level=CMerModel.target_id).agg(['count']).values.reshape(-1),
                         "%s Target compound histogram %s" % (data_desc, str(fold) if fold is not None else ''))
    save_array_histogram(tm.STANDARD_RELATION.groupby(level=CMerModel.mol_id).agg(['count']).values.reshape(-1),
                         "%s Compound target histogram %s" % (data_desc, str(fold) if fold is not None else ''))
