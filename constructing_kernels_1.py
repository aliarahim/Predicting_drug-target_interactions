import pandas as pd
import numpy as np
import os
from rdkit.Chem import PandasTools
from rdkit import DataStructs
from rdkit.Chem import AllChem
from mismatchkernel import mismatchkernel


drug_target_data=pd.read_csv(os.getcwd()+'/drug_target_data_Part_2.csv')
PandasTools.AddMoleculeColumnToFrame(drug_target_data,'SMILES','Molecule',includeFingerprints=True)
fingerprints = [AllChem.GetMorganFingerprintAsBitVect(x,2,1024) for x in drug_target_data['Molecule']]
drug_target_data=drug_target_data.assign(fingerprints= pd.Series(fingerprints))
drug_target_data_train=drug_target_data.sample(frac=0.8,random_state=200)
drug_target_data_test=drug_target_data.drop(drug_target_data_train.index)
drug_target_data_train.reset_index(inplace=True, drop=True)
drug_target_data_test.reset_index(inplace=True, drop=True)
drug_target_data_test.to_csv('drug_target_data_test.csv', encoding='utf-8', index=False)
drug_target_data_train.to_csv('drug_target_data_train.csv', encoding='utf-8', index=False)

tanimoto_kernel_train=np.empty([len(drug_target_data_train),len(drug_target_data_train)],dtype=float)
mismatch_kernel_train=np.empty([len(drug_target_data_train),len(drug_target_data_train)],dtype=float)
tanimoto_kernel_test_train=np.empty([len(drug_target_data_test),len(drug_target_data_train)],dtype=float)
mismatch_kernel_test_train=np.empty([len(drug_target_data_test),len(drug_target_data_train)],dtype=float)
tanimoto_kernel_test=np.empty([len(drug_target_data_test),len(drug_target_data_test)],dtype=float)
mismatch_kernel_test=np.empty([len(drug_target_data_test),len(drug_target_data_test)],dtype=float)
for i in range(len(drug_target_data_train)):
    for j in range(len(drug_target_data_train)):
        tanimoto_kernel_train[i][j]=DataStructs.FingerprintSimilarity(drug_target_data_train['fingerprints'].iloc[i],drug_target_data_train['fingerprints'].iloc[j])        
        mismatch_kernel_train[i][j]=mismatchkernel(drug_target_data_train['Sequence'].iloc[i],drug_target_data_train['Sequence'].iloc[j],5)
np.savetxt('tanimoto_kernel_train.npy', tanimoto_kernel_train)
np.savetxt('mismatch_kernel_train.npy', mismatch_kernel_train)


for i in range(len(drug_target_data_test)):
    for j in range(len(drug_target_data_train)):
        tanimoto_kernel_test_train[i][j]=DataStructs.FingerprintSimilarity(drug_target_data_test['fingerprints'].iloc[i],drug_target_data_train['fingerprints'].iloc[j])        
        mismatch_kernel_test_train[i][j]=mismatchkernel(drug_target_data_test['Sequence'].iloc[i],drug_target_data_train['Sequence'].iloc[j],5)
np.savetxt('tanimoto_kernel_test_train.npy', tanimoto_kernel_test_train)
np.savetxt('mismatch_kernel_test_train.npy', mismatch_kernel_test_train)

for i in range(len(drug_target_data_test)):
    for j in range(len(drug_target_data_test)):
        tanimoto_kernel_test[i][j]=DataStructs.FingerprintSimilarity(drug_target_data_test['fingerprints'].iloc[i],drug_target_data_test['fingerprints'].iloc[j])        
        mismatch_kernel_test[i][j]=mismatchkernel(drug_target_data_test['Sequence'].iloc[i],drug_target_data_test['Sequence'].iloc[j],5)
np.savetxt('tanimoto_kernel_test.npy', tanimoto_kernel_test)
np.savetxt('mismatch_kernel_test.npy', mismatch_kernel_test)
