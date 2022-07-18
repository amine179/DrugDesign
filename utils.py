
# ------------------------------------------------------------------
# list of all characters used for molecules in the Chembl21 dataset
all_chars_list = ['<', 'a','b','c','e','g','i','l','n',
      'o','p','r','s','t','A','B','C','F','H','I','K','L','M','N',
      'O','P','R','S','T','V','X','Z','0','1','2','3','4','5','6','7',
       '8','9', '=','#','+','-','[',']','(',')','/','\\', '@','.','%', '>']

all_chars = ''.join(all_chars_list)
n_chars = len(all_chars)

# ------------------------------------------------------
# function to save a list to text file
def list2txt(filename, mylist):
    textfile = open(filename, "w")
    for element in mylist:
        textfile.write(element + "\n")
    textfile.close()

# ------------------------------------------------------------------
# function to convert a list of smiles to concatenated string contating all
# smiles, plus an added SOS '<' and EOS '>'
def listsmis2strsmis(list_smis): 
    str_smis = []
    for s in list_smis:
        s='<' + s + '>'
        str_smis.append(s)
    str_smis = ''.join(str_smis)
    return str_smis

# function to go back from the string above, to a list of smiles again 
def strsmis2listsmis(str_smis):
    str_smis = str_smis.replace("<", "") #remove sos
    list_smis = []
    smi = ''
    for c in str_smis:
        smi+=c
        if c == '>': #stop at eos and append
            smi = smi.replace(">", "")
            list_smis.append(smi) 
            smi = ''
    return list_smis

#---------------------------------------------------------------------
import numpy as np
import random
import rdkit as rd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity

#LogP
def logP(smiles):
    return(Descriptors.MolLogP(Chem.MolFromSmiles(smiles)))

#molecular weight
def molWt(smiles):
    return(Descriptors.MolWt(Chem.MolFromSmiles(smiles)))

#number hydrogen bond acceptors
def numAcc(smiles):
    return(Descriptors.NumHAcceptors(Chem.MolFromSmiles(smiles)))

#number hydrogen bond donors
def numDon(smiles):
    return(Descriptors.NumHDonors(Chem.MolFromSmiles(smiles)))

#number of rotatable bonds
def rolBon(smiles):
    return(Descriptors.NumRotatableBonds(Chem.MolFromSmiles(smiles)))

# -----------------------------------------------------------------------------
# function to check gramatical and chemical validity of a smiles
def check_validity(smi):
    m = Chem.MolFromSmiles(smi,sanitize=False)
    if m is None:
        v = 0
        #print('invalid SMILES')
    else:
        v = 1
        #print("valid smiles.")
        try:
            Chem.SanitizeMol(m)
        except:
            v = 0
            #print('invalid chemistry')
    return v

# function to check the novelty of a smiles with respect to a given list of smiles
def check_novelty(smi, smis_list):
    n = 1
    if smi in smis_list:
        n = 0
    return n
# --------------------------------------------------------


# function to compute the R-value in Fei Mao et al.
def compute_rvalue(mol):
    new_s = []
    for atom in mol.GetAtoms():
        asym = atom.GetSymbol()
        new_s.append(asym)
    # print(new_s)

    chars = set(new_s)
    # print(chars)

    r_dict = {}
    for c in chars:
        n = 0
        for s in new_s:
            if c == s:
                n = n+1
        r_dict[c] = n
    # print(r_dict)

    n_with_c = sum(list(r_dict.values()))
    # print(n_with_c)

    n_without_c = 0
    for k in list(r_dict.keys()):
        if k != 'C' and k != 'c':
            n_without_c += r_dict[k]
    # print(n_without_c)

    if n_with_c == 0:
        return 0
    else:
        rvalue = n_without_c/n_with_c
        return rvalue

# -------------------------------------------------------
# function to compute properties used for objectives discussed in the paper
import rdkit
def get_props(smi, c=0):
    if c==0: 
        if check_validity(smi):
            return logP(smi)
        else:
            return 99999                   # i.e., 'invalid smiles'

    if c==1:
        if check_validity(smi):
            return (logP(smi), molWt(smi), numAcc(smi), numDon(smi), rolBon(smi))
        else:
            return 9999, 99999, 99999, 99999, 999999


    if c==2:
        if check_validity(smi):
            mol = rdkit.Chem.MolFromSmiles(smi)
            
            arr  = rdkit.Chem.rdMolDescriptors.CalcNumAromaticRings(mol)
            alr  = rdkit.Chem.rdMolDescriptors.CalcNumAliphaticRings(mol)
            oh   = rdkit.Chem.Fragments.fr_Ar_OH(mol) +  rdkit.Chem.Fragments.fr_Al_OH(mol)
            cooh = rdkit.Chem.Fragments.fr_Al_COO(mol) + rdkit.Chem.Fragments.fr_Ar_COO(mol)
            coor = rdkit.Chem.Fragments.fr_ester(mol)
            nh2  = rdkit.Chem.Fragments.fr_NH2(mol)
            rval = compute_rvalue(mol)
            
            return (arr, alr, oh, cooh, coor, nh2, rval)
        else:
            return 9999, 99999, -99999, -99999, -99999, -99999, 99999


def computeUMAP(smis_list, seed=42):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from sklearn.manifold import TSNE
    import umap
    sns.set_context('poster')
    sns.set_style('white')
    sns.set_color_codes()
    plot_kwds = {'alpha':0.5, 's':80, 'linewidth':0}

    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import Draw
    from rdkit.Chem import DataStructs
    from rdkit.Chem.Draw import IPythonConsole

    # convert to rdkit mol objects from smiles
    mols = []
    for s in smis_list:
        mols.append(Chem.MolFromSmiles(s))
        
    # compute Morgant fingerprints :
    X = []
    for mol in mols:
        arr = np.zeros((0,))
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X.append(arr)
    
    # compute tanimoto distance (use numba for speed)
    import numba
    @numba.njit()
    def tanimoto_dist(a,b):
        dotprod = np.dot(a,b)
        tc = dotprod / (np.sum(a) + np.sum(b) - dotprod)
        return 1.0-tc


    import warnings
    warnings.filterwarnings('ignore')

    # compute the UMAP
    umap_X = umap.UMAP(n_neighbors=8, min_dist=0.3, metric=tanimoto_dist, random_state=seed).fit_transform(X)

    return umap_X



