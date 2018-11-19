import pandas as pd
import numpy as np
import os
drug_target_data_A=pd.read_csv(os.getcwd()+'/drug_target_data_Part_1.csv')
drug_target_data_A.drop_duplicates(inplace=True)
drug_target_data_A.reset_index(inplace=True, drop=True)
drug_data=drug_target_data_A[['DrugBank ID', 'Name', 'SMILES']]
target_data=drug_target_data_A[['UniProt ID', 'UniProt Name','Sequence']]
x=drug_data.sample(n=10000,replace=True,random_state=1)
x.reset_index(inplace=True, drop=True)
y=target_data.sample(n=10000,replace=True,random_state=3)
y.reset_index(inplace=True, drop=True)
pos_labels=pd.Series(np.ones((len(drug_target_data_A)),dtype=int))
drug_target_data_A=drug_target_data_A.assign(labels=pos_labels)
drug_target_data_B=pd.concat([x,y],axis=1)
drug_target_data_B.drop_duplicates(inplace=True)
drug_target_data_B.reset_index(inplace=True, drop=True)
drug_target_data_B=drug_target_data_B.assign(labels= pd.Series(np.full((len(drug_target_data_B)),-1,dtype=int), index=drug_target_data_B.index))
drug_target_data=pd.concat([drug_target_data_A,drug_target_data_B],axis=0,sort=False)
drug_target_data.drop_duplicates(subset=['DrugBank ID', 'Name', 'SMILES','UniProt ID', 'UniProt Name','Sequence'],inplace=True)
drug_target_data=drug_target_data.sample(frac=0.05,random_state=1)
drug_target_data.reset_index(inplace=True, drop=True)
drug_target_data.to_csv('drug_target_data_Part_2.csv', encoding='utf-8', index=False)

