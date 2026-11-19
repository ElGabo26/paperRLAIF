
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import TensorDataset

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

MODEL_PATH = (
    "/workspace/models/deberta-v3-large/deberta-v3-large"
)

DATASET_PATH = (
    "/workspace/paperRLAIF/codTraining/cabezales/"
    "trainingCabezales/skill.csv"
)

VECTORS_PATH = (
    "/workspace/paperRLAIF/codTraining/cabezales/"
    "vectorBases/skill_mean.pt"
)

OUTPUT_PATH = (
    "/workspace/paperRLAIF/codTraining/cabezales/"
    "verificacion_embeddings_skill.csv"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MAX_LENGTH = 256
BATCH_SIZE = 8
NUM_SAMPLES = 100
SEED = 42

# Tolerancias para diagnóstico numérico.
ATOL = 1e-5
RTOL = 1e-4


# ============================================================
# 2. CARGAR LOS VECTORES ORIGINALES
# ============================================================

torch.serialization.add_safe_globals([TensorDataset])

dataset = torch.load(
    VECTORS_PATH,
    map_location="cpu",
    weights_only=True
)

original_embeddings = dataset.tensors[0].float()
original_labels = dataset.tensors[1]

print("Vectores originales:", original_embeddings.shape)
print("Etiquetas:", original_labels.shape)


# ============================================================
# 3. CARGAR LOS TEXTOS ORIGINALES
# ============================================================

# Se utiliza index_col=0 porque así se cargó el CSV
# cuando se generó skill_mean.pt.

data = pd.read_csv(
    DATASET_PATH,
    index_col=0
)

assert list(data.columns) == ["text", "labels"], (
    f"Columnas inesperadas: {data.columns.tolist()}"
)

assert len(data) == len(original_embeddings), (
    "La cantidad de textos no coincide con los embeddings."
)

# Importante:
# Esta comparación presupone que el CSV no fue modificado
# después de generar skill_mean.pt y conserva su orden.

print("Textos originales:", len(data))

# ============================================================
# 3.1 VERIFICAR CORRESPONDENCIA DE ETIQUETAS
# ============================================================


import joblib
import numpy as np

LABEL_ENCODER_PATH = (
    "/workspace/paperRLAIF/codTraining/"
    "cabezales/encoders/skill.joblib"
)

label_encoder = joblib.load(LABEL_ENCODER_PATH)

etiquetas_csv = label_encoder.transform(
    data["labels"].astype(str).str.strip()
)

etiquetas_tensor = original_labels.numpy()

coincidencias = (
    etiquetas_csv == etiquetas_tensor
)

print(
    "Etiquetas coincidentes:",
    coincidencias.sum(),
    "/",
    len(coincidencias)
)

print(
    "Porcentaje:",
    coincidencias.mean() * 100
)

# ============================================================
# 4. SELECCIONAR ÍNDICES
# ============================================================

rng = np.random.default_rng(SEED)

n = min(NUM_SAMPLES, len(data))

indices = np.sort(
    rng.choice(
        len(data),
        size=n,
        replace=False
    )
)

textos = data.iloc[indices]["text"].tolist()
vectores_originales = original_embeddings[indices]



# ============================================================
# 5. CARGAR TOKENIZER Y DEBERTA
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    fix_mistral_regex=True
)

encoder = AutoModel.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

encoder = encoder.to(
    device=DEVICE,
    dtype=torch.float32
)

encoder.eval()

print("Tokenizer:", type(tokenizer))
print("Encoder:", type(encoder))


# ============================================================
# 6. MEAN POOLING
# ============================================================

def mean_pooling(last_hidden_state, attention_mask):

    mask = attention_mask.unsqueeze(-1).to(
        last_hidden_state.dtype
    )

    embeddings = (
        last_hidden_state * mask
    ).sum(dim=1)

    count = mask.sum(dim=1).clamp(min=1.0)

    return embeddings / count


# ============================================================
# 7. REGENERAR USANDO LOTES ORIGINALES
# ============================================================

@torch.inference_mode()
def generar_embeddings_comparables(indices):

    resultados = {}

    # Identificar los lotes originales que contienen
    # los registros seleccionados.
    lotes = sorted(set(
        int(i) // BATCH_SIZE
        for i in indices
    ))

    for numero_lote in lotes:

        inicio = numero_lote * BATCH_SIZE
        fin = min(inicio + BATCH_SIZE, len(data))

        # Recuperar el lote completo original.
        batch = data.iloc[inicio:fin]["text"].tolist()

        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        tokens = {
            k: v.to(DEVICE)
            for k, v in tokens.items()
        }

        outputs = encoder(**tokens)

        embeddings = mean_pooling(
            outputs.last_hidden_state,
            tokens["attention_mask"]
        ).float().cpu()

        # Conservar los embeddings de las filas solicitadas.
        for indice in indices:

            if inicio <= indice < fin:

                posicion = int(indice) - inicio

                resultados[int(indice)] = (
                    embeddings[posicion]
                )

    return torch.stack([
        resultados[int(i)]
        for i in indices
    ])


vectores_nuevos = generar_embeddings_comparables(indices)

# ============================================================
# 8. COMPARAR EMBEDDINGS
# ============================================================

assert vectores_originales.shape == vectores_nuevos.shape, (
    "Las dimensiones no coinciden."
)

diferencia = (
    vectores_originales - vectores_nuevos
).abs()

similitud_coseno = F.cosine_similarity(
    vectores_originales,
    vectores_nuevos,
    dim=1
)

error_maximo_por_vector = diferencia.max(dim=1).values
error_medio_por_vector = diferencia.mean(dim=1)

equivalentes = torch.isclose(
    vectores_originales,
    vectores_nuevos,
    atol=ATOL,
    rtol=RTOL
).all(dim=1)

print("\n========== RESULTADOS ==========")

print(
    "Similitud coseno promedio:",
    similitud_coseno.mean().item()
)

print(
    "Similitud coseno mínima:",
    similitud_coseno.min().item()
)

print(
    "Error absoluto máximo:",
    diferencia.max().item()
)

print(
    "Error absoluto medio:",
    diferencia.mean().item()
)

print(
    "Vectores equivalentes:",
    f"{equivalentes.sum().item()}/{len(equivalentes)}"
)

print(
    "Porcentaje equivalente:",
    f"{equivalentes.float().mean().item()*100:.2f}%"
)


# ============================================================
# 9. EXPORTAR RESULTADOS
# ============================================================

resultados = pd.DataFrame({
    "indice_original": indices,
    "texto": textos,
    "similitud_coseno": similitud_coseno.numpy(),
    "error_maximo": error_maximo_por_vector.numpy(),
    "error_medio": error_medio_por_vector.numpy(),
    "equivalente": equivalentes.numpy()
})

Path(OUTPUT_PATH).parent.mkdir(
    parents=True,
    exist_ok=True
)

resultados.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nResultados guardados en:", OUTPUT_PATH)

if equivalentes.all().item():
    print(
        "\nRESULTADO: Los embeddings son equivalentes "
        "bajo las tolerancias configuradas."
    )
else:
    print(
        "\nADVERTENCIA: Se encontraron diferencias que "
        "superan las tolerancias configuradas."
    )
