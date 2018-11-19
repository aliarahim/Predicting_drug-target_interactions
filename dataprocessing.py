import pandas as pd
import os
import urllib
from pandas.compat import StringIO

drug_target_data_A=pd.read_csv(os.getcwd()+'/uniprot_links.csv')
drug_data=pd.read_csv(os.getcwd()+'/structure_links.csv')
drug_data=drug_data[['DrugBank ID','SMILES']]
drug_target_data_B= pd.merge(drug_data,drug_target_data_A,on='DrugBank ID')

uniprot_ids=drug_target_data_B['UniProt ID']
df=uniprot_ids.dropna()
uniprot=df.unique()
uniprots_id=uniprot.tolist()
uniprots=" ".join(uniprots_id)
url='https://www.uniprot.org/uploadlists/'
params= {
'from':'ACC+ID',
'to':'ACC',
'columns':'id,sequence',
'format':'tab',
'query':uniprots
}
data=urllib.parse.urlencode(params).encode("utf-8")
request=urllib.request.Request(url, data)
contact="alia_rahim@live.com"
request.add_header('User-Agent','Python %s')
response=urllib.request.urlopen(request)
page=response.read().decode("utf-8")
target_info=pd.read_csv(StringIO(page),sep="\t")
target_info.rename(columns={'Entry': 'UniProt ID'}, inplace=True)
target_info=target_info[['UniProt ID', 'Sequence']]

drug_target_data=pd.merge(target_info,drug_target_data_B,on='UniProt ID')
drug_target_data=drug_target_data[['DrugBank ID','Name','SMILES','UniProt ID','UniProt Name','Sequence']]
drug_target_data.dropna(axis=0,inplace=True)
drug_target_data.to_csv('drug_target_data_Part_1.csv', encoding='utf-8', index=False)
