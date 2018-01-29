import numpy as np
from scipy import sparse
import csv
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pandas as pd


def save_sparse_matrix(filename, x):
    row = x.row
    col = x.col
    data = x.data
    shape = x.shape
    np.savez(filename, row=row, col=col, data=data, shape=shape)

def check_sparse(name):
    if name in ["ap", "tt"] or name.startswith("ecfc") \
            or name.startswith("fcfc") or name.startswith("nc_"):
        return True
    return False

def load_sparse_matrix(fpName):
    return np.load(fpName)

def accToWeights(acc):
    return np.log1p(acc.sum()/acc)

def createweightVector(weightData, isSparse):
    if (isSparse):
        data = np.double(accToWeights(weightData['data']))
        weightVector = DataStructs.cDataStructs.MSVectorXdHelper.Create()
        DataStructs.cDataStructs.MSVectorXdHelper.Map(weightVector, np.int64(weightData['col']), data,
                                                      np.int64(len(data)), np.int64(np.max(weightData['col'])+1))
    else:
        data = sparse.coo_matrix((accToWeights(weightData['data']), (weightData['row'], weightData['col'])),
                                 shape=weightData['shape']).todense().tolist()[0]
        weightVector = DataStructs.cDataStructs.MVectorXdHelper.Create()
        DataStructs.cDataStructs.MVectorXdHelper.Map(weightVector, np.double(data), np.int64(len(data)))
    return weightVector

def loadWeights(fpName):
    weightVector = None
    try:
        weightVector = weightVectors[fpName]
    except KeyError as e:
        weightData = load_sparse_matrix(fpName)
        weightVector = createweightVector(weightData, check_sparse(fpName))
        weightVectors[fpName] = weightVector
    finally:
        return weightVector


def createweightVectorAsCSC(weightData):
    data = np.double(accToWeights(weightData['data']))
    return sparse.csr_matrix((data, (weightData['row'], weightData['col'])), dtype=np.double)

def loadWeightsAsCSC(fpName):
    weightVector = None
    try:
        weightVector = weightVectors[fpName]
    except KeyError as e:
        weightData = load_sparse_matrix(fpName)
        weightVector = createweightVectorAsCSC(weightData)
        weightVectors[fpName] = weightVector
    finally:
        return weightVector


def foldr(func, init, seq):
    if not seq:
        return init
    else:
        return func(seq[0], foldr(func, init, seq[1:]))


def foldl(func, init, seq):
    if seq is None or len(seq) == 0:
        return init
    else:
        return foldl(func, func(init, seq[0]), seq[1:])


def mol2smiles_mapping(fileName):
    reader = csv.DictReader(open(fileName, 'rb'))
    dict_list = {}
    for line in reader:
        dict_list[int(line['molregno'])] = line['canonical_smiles']
    dict_keys = dict_list.keys()
    dict_keys.sort()
    mol_list = [None] * (dict_keys[-1]+1)
    for key in dict_keys:
        mol_list[key] = dict_list[key]
    return np.array(mol_list)


def smile2rdkit(smi, radius=2, *args, **kwargs):
    if smi is None:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprint(mol, radius, *args, **kwargs)


def smile2ecfp(smi):
    if smi is None:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 16384)


def clean_mols(mols):
    mol_indices = []
    for key, val in enumerate(mols):
        if val is not None:
            mol_indices.append(key)
    return mols[mol_indices]


def group_target2_mol_mapping(fileName):
    return pd.read_csv(fileName)


def smile2eigen(smi, size):
    if smi is None:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprint(mol,2).convertToEigenVector(size)

weightVectors = {}