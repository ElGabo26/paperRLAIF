
from huggingface_hub import snapshot_download


LOCALROOT="/workspace/models/deberta-v3-large"
MODELROOT="microsoft/deberta-v3-large"

def  dowloadModel(modelroot:str, localRoot:str):
    nameModel=modelroot.split("/")[-1]
    localRoot=localRoot+"/"+nameModel
    ruta_modelo = snapshot_download(
    repo_id=modelroot,
    local_dir=localRoot)
    print("Modelo descargado en:", ruta_modelo)
    return ruta_modelo

if __name__=="__main__":
    m=MODELROOT
    ruta =dowloadModel(m,LOCALROOT)
    print(ruta)