
# ------------------------------------------------------------------

all_chars_list = ['<', 'a','b','c','e','g','i','l','n',
      'o','p','r','s','t','A','B','C','F','H','I','K','L','M','N',
      'O','P','R','S','T','V','X','Z','0','1','2','3','4','5','6','7',
       '8','9', '=','#','+','-','[',']','(',')','/','\\', '@','.','%', '>']

all_chars = ''.join(all_chars_list)
n_chars = len(all_chars)

# ------------------------------------------------------
def list2txt(filename, mylist):
    textfile = open(filename, "w")
    for element in mylist:
        textfile.write(element + "\n")
    textfile.close()

# ------------------------------------------------------------------

def listsmis2strsmis(list_smis): 
    str_smis = []
    for s in list_smis:
        s='<' + s + '>'
        str_smis.append(s)
    str_smis = ''.join(str_smis)
    return str_smis


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

# code below is from: https://github.com/jyasonik/MoleculeMO/blob/master/DataPostprocessing.ipynb
import numpy as np
import random
import rdkit as rd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity

random.seed(123)
np.random.seed(123)

#Determines LogP
def logP(smiles):
    return(Descriptors.MolLogP(Chem.MolFromSmiles(smiles)))

#Determines molecular weight
def molWt(smiles):
    return(Descriptors.MolWt(Chem.MolFromSmiles(smiles)))

#Determines number hydrogen bond acceptors
def numAcc(smiles):
    return(Descriptors.NumHAcceptors(Chem.MolFromSmiles(smiles)))

#Determines number hydrogen bond donors
def numDon(smiles):
    return(Descriptors.NumHDonors(Chem.MolFromSmiles(smiles)))

#Determines topological polar surface area
def polSur(smiles):
    return(Descriptors.TPSA(Chem.MolFromSmiles(smiles)))

#Determines number of rotatable bonds
def rolBon(smiles):
    return(Descriptors.NumRotatableBonds(Chem.MolFromSmiles(smiles)))

# -----------------------------------------------------------------------------
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

def check_novelty(smi, smis_list):
    n = 1
    for s in smis_list:
        if s == smi:
            n = 0
    return n
# --------------------------------------------------------


# R-value in Fei Mao et al.
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




