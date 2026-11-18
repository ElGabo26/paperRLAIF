
import pandas  as pd
import re
import os
import matplotlib.pyplot as plt
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
dataset={}
for model in models:
    dataset[model]=[pd.read_csv(root+"/"+x,index_col=0) for x in bases if model in x]

prompts=dataset['Qwen2.5-1.5B-Instruct'][0]['prompt'].copy()
for k in dataset.keys():
    r=prompts.copy()
    dbs=dataset[k]
    for db  in dbs:
        r=pd.merge(r,db,on='prompt')
        print(r.shape)
    dataset[k]=r
#revisamos  si  existen  respuestas  iguales
from itertools import product
llama=dataset['Qwen2.5-1.5B-Instruct']
print(llama.shape)
a={1,2,3,4}
b={1,2,3,4}
c=list(product(a,b))
c=[x for x in c if x[0]!=x[1]]
for par in c:
    print(llama[llama[f'response_{par[0]}']==llama[f'response_{par[1]}']].shape)
dataset['DeepSeek-R1-Distill-Qwen-1.5B'].columns
#funcion para  crear  las  4   respuestas
def make4(db:pd.DataFrame):
    a= db.columns
    a=[x for x in a if 'response' in x]
    a.insert(0,'prompt')
    db1=db[a].copy()
    return db1
tocalify={}
root=r'C:\Users\Gabo\Desktop\PAPPER Rlaif\codPreferences\calify/'
for k in dataset.keys():
    r=make4(dataset[k])
    r.to_csv(root+k+".csv")