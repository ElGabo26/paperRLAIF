import pandas  as pd
import re
from pathlib import Path

BASES="/workspace/papaerRLAIF/codPreferences/bases"
RUTAPROMPTS = "/workspace/papaerRLAIF/codPreferences/prompts.csv"


db=pd.read_csv(Path(RUTAPROMPTS))
print(db.shape)

patron = r"result(.*?)_"
root=Path(BASES)
bases=list(root.iterdir())
models=set([re.match(patron,x.name).group(1) for x in bases])
print(models)

#funcion para  crear  las  4   respuestas
def make4(db:pd.DataFrame):
    a= db.columns
    a=[x for x in a if 'response' in x]
    a.insert(0,'prompt')
    db1=db[a].copy()
    return db1


dataset={}
for model in models:
    dataset[model]=[pd.read_csv(root/x,index_col=0) for x in bases if model in x.name]




for k in dataset.keys():
    r=prompts
    dbs=dataset[k]
    for db1  in dbs:
        r=pd.merge(r,db1,on='prompt')
        print(r.shape)
    dataset[k]=r


tocalify={}
root="/workspace/papaerRLAIF/codPreferences/tocalify"
for k in dataset.keys():
    r=make4(dataset[k])
    print('MADE ==',k)
    r.to_csv(root+k+".csv")