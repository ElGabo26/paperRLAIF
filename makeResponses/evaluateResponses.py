"""Evaluar CSV de respuestas con la vectorización de createVectorBases.py.

Se calcula un único conjunto de embeddings por CSV (lotes de 8), que se
comparte entre los cabezales. Se preservan filas duplicadas y nulas.
"""
from pathlib import Path

import pandas as pd
import torch

from evaluateResponsesTools import (
    load_encoder,
    create_embeddings,
    load_complete_classifier,
    load_label_encoder,
    classify_embeddings,
)

# ============================================================
# 1. Configuración: misma que createVectorBases.py
# ============================================================
ROUTE_BERT = Path("/workspace/models/deberta-v3-large/deberta-v3-large")
ROUTE_CABEZALES = Path("/workspace/adaptedModels/cabezales")
ROUTE_ENCODER = Path("/workspace/paperRLAIF/codTraining/cabezales/encoders")
ROUTE_RESPONSES = Path("/workspace/paperRLAIF/makeResponses/responses")
OUTPUT_ROUTE = Path("/workspace/paperRLAIF/makeResponses/evaluatedResponses")

MAX_LENGTH = 256
EMBEDDING_BATCH_SIZE = 8  # Igual que createVectorBases.py
CLASSIFIER_BATCH_SIZE = 12  # Solo afecta el paso del cabezal
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    OUTPUT_ROUTE.mkdir(parents=True, exist_ok=True)
    if not ROUTE_CABEZALES.is_dir():
        raise FileNotFoundError(ROUTE_CABEZALES)
    if not ROUTE_RESPONSES.is_dir():
        raise FileNotFoundError(ROUTE_RESPONSES)

    # Modelos completos originales guardados como {cabezal}.pt.
    # No incluir archivos de métricas, checkpoints ni state_dict.
    cabezales = sorted(
        p for p in ROUTE_CABEZALES.glob("*.pt") if p.is_file()
    )
    bases = sorted(p for p in ROUTE_RESPONSES.glob("*.csv") if p.is_file())
    if not cabezales:
        raise FileNotFoundError(f"No hay cabezales .pt en {ROUTE_CABEZALES}")

    # Cargar DeBERTa una sola vez, misma configuración que creación de vectores.
    tokenizer, bert = load_encoder(ROUTE_BERT, DEVICE)
    print(f"Dispositivo: {DEVICE}, cabezales: {len(cabezales)}, CSV: {len(bases)}")

    for base in bases:
        data = pd.read_csv(base)
        if "response" not in data.columns:
            raise KeyError(f"Falta columna 'response' en {base}")
        print(f"\nArchivo: {base.name} | filas originales: {len(data)}")
        validos = data["response"].map(lambda x: isinstance(x, str) and bool(x.strip()))
        indices_validos = data.index[validos]
        resultado = data.copy()  # Conserva todas las columnas y posiciones originales.

        if len(indices_validos) == 0:
            print("No hay respuestas válidas; se conserva el CSV sin clasificar.")
        else:
            # El orden de respuestas NO se altera, incluidos los textos repetidos.
            respuestas = data.loc[indices_validos, "response"].tolist()
            embeddings = create_embeddings(
                prompts=respuestas,
                tokenizer=tokenizer,
                encoder=bert,
                device=DEVICE,
                batch_size=EMBEDDING_BATCH_SIZE,
                max_length=MAX_LENGTH,
            )
            print("Embeddings:", tuple(embeddings.shape))

            for ruta_cabezal in cabezales:
                nombre = ruta_cabezal.stem
                ruta_labels = ROUTE_ENCODER / f"{nombre}.joblib"
                labels = load_label_encoder(ruta_labels)
                classifier = load_complete_classifier(ruta_cabezal, DEVICE)
                print(f"Cabezal: {nombre}; etiquetas: {list(labels.classes_)}")
                evaluado = classify_embeddings(
                    embeddings=embeddings,
                    classifier=classifier,
                    label_encoder=labels,
                    device=DEVICE,
                    batch_size=CLASSIFIER_BATCH_SIZE,
                )
                if len(evaluado) != len(indices_validos):
                    raise RuntimeError(f"Número de predicciones incorrecto para {nombre}")
                for columna in evaluado.columns:
                    nueva = f"{columna}_{nombre}"
                    if nueva in resultado.columns:
                        raise ValueError(f"Columna ya existente: {nueva}")
                    # Asignar por posición de fila, NUNCA usar merge por texto.
                    serie = pd.Series(evaluado[columna].to_numpy(), index=indices_validos)
                    resultado[nueva] = serie
                del classifier

        output = OUTPUT_ROUTE / f"evaluated_{base.name}"
        resultado.to_csv(output, index=False)
        print(f"Guardado: {output} | filas: {len(resultado)}")


if __name__ == "__main__":
    main()
