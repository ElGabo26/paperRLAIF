"""Inferencia de cabezales RLAIF compatible con createVectorBases.py.

La vectorización reproduce el procedimiento de preparar_dataset_vectores
(pooling='mean') empleado por createVectorBases.py:
AutoTokenizer(..., fix_mistral_regex=True), AutoModel, eval(), tokenización
con padding dinámico y longitud 256, lotes de embeddings de 8, promedio
con máscara y conversión final a float32 en CPU.

La equivalencia bit a bit no está garantizada entre versiones/hardware.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoTokenizer


TRAINING_EMBEDDING_BATCH_SIZE = 8
TRAINING_MAX_LENGTH = 256


def load_encoder(encoder_path: str | Path, device: torch.device):
    """Carga el tokenizador y el encoder como en createVectorBases.py."""
    ruta = Path(encoder_path)
    if not ruta.is_dir():
        raise FileNotFoundError(f"No existe el directorio: {ruta}")

    # Importante: NO forzar use_fast=False. createVectorBases.py no lo fuerza.
    tokenizer = AutoTokenizer.from_pretrained(
        str(ruta),
        local_files_only=True,
        fix_mistral_regex=True,
    )
    encoder = AutoModel.from_pretrained(str(ruta), local_files_only=True)
    encoder = encoder.to(device)
    for parametro in encoder.parameters():
        parametro.requires_grad = False
    encoder.eval()
    print(f"Tokenizer: {type(tokenizer).__name__}, is_fast={tokenizer.is_fast}")
    print(f"Encoder: {type(encoder).__name__}; hidden_size={encoder.config.hidden_size}")
    return tokenizer, encoder


def load_complete_classifier(classifier_path: str | Path, device: torch.device) -> nn.Module:
    """Abre modelos completos guardados con torch.save(model, ...).

    Solo cargar modelos propios/confiables: weights_only=False deserializa Python.
    """
    ruta = Path(classifier_path)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el cabezal: {ruta}")
    classifier = torch.load(ruta, map_location=device, weights_only=False)
    if not isinstance(classifier, nn.Module):
        raise TypeError(f"{ruta} no contiene un modelo torch.nn.Module completo")
    classifier = classifier.to(device=device, dtype=torch.float32)
    classifier.eval()
    return classifier


def load_label_encoder(labels_path: str | Path) -> LabelEncoder:
    """El joblib debe ser el LabelEncoder usado al construir los tensores."""
    ruta = Path(labels_path)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró LabelEncoder: {ruta}")
    objeto = joblib.load(ruta)  # Solo archivos propios/confiables.
    if not isinstance(objeto, LabelEncoder) or not hasattr(objeto, "classes_"):
        raise TypeError(f"{ruta} no contiene un LabelEncoder ajustado")
    return objeto


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Misma operación que preparar_dataset_vectores(..., pooling='mean')."""
    if last_hidden_state.ndim != 3 or attention_mask.shape != last_hidden_state.shape[:2]:
        raise ValueError("Dimensiones incompatibles en mean_pooling")
    expanded_mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    suma = (last_hidden_state * expanded_mask).sum(dim=1)
    validos = expanded_mask.sum(dim=1).clamp(min=1.0)
    return suma / validos


@torch.inference_mode()
def create_embeddings(
    prompts: Sequence[str],
    tokenizer,
    encoder: nn.Module,
    device: torch.device,
    batch_size: int = TRAINING_EMBEDDING_BATCH_SIZE,
    max_length: int = TRAINING_MAX_LENGTH,
) -> torch.Tensor:
    """Embeddings mean con la misma preparación y lote 8 de createVectorBases.py.

    Para comparar contra skill_mean.pt, los textos y la composición de
    cada lote deben conservar el orden original del CSV de entrenamiento.
    """
    if not prompts:
        raise ValueError("La colección de respuestas está vacía")
    if batch_size <= 0 or max_length <= 0:
        raise ValueError("batch_size y max_length deben ser positivos")
    if not all(isinstance(t, str) for t in prompts):
        raise TypeError("Todas las respuestas deben ser texto; filtre nulos antes")
    encoder.eval()
    all_embeddings = []
    for inicio in range(0, len(prompts), batch_size):
        lote = list(prompts[inicio:inicio + batch_size])
        tokens = tokenizer(
            lote,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {clave: valor.to(device) for clave, valor in tokens.items()}
        outputs = encoder(**tokens)
        embeddings = mean_pooling(outputs.last_hidden_state, tokens["attention_mask"])
        # Igual que preparar_dataset_vectores: float32 y CPU tras el pooling.
        all_embeddings.append(embeddings.float().cpu())
    return torch.cat(all_embeddings, dim=0)


@torch.inference_mode()
def classify_embeddings(
    embeddings: torch.Tensor,
    classifier: nn.Module,
    label_encoder: LabelEncoder,
    device: torch.device,
    batch_size: int = 12,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Evalúa embeddings precomputados sin alterar las etiquetas originales."""
    if batch_size <= 0:
        raise ValueError("batch_size debe ser positivo")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold debe estar entre 0 y 1")
    classifier = classifier.to(device=device, dtype=torch.float32)
    classifier.eval()
    classes = [str(x) for x in label_encoder.classes_]
    rows = []
    for inicio in range(0, len(embeddings), batch_size):
        lote = embeddings[inicio:inicio + batch_size].to(device=device, dtype=torch.float32)
        logits = classifier(lote)
        if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
            if len(classes) != 2:
                raise ValueError("El cabezal binario tiene una salida, pero no dos clases")
            logits = logits.reshape(-1)
            if not torch.isfinite(logits).all():
                raise ValueError("Logits no finitos en clasificación binaria")
            positives = torch.sigmoid(logits).cpu().numpy()
            for prob in positives:
                idx = int(prob >= threshold)
                neg = 1.0 - float(prob)
                pos = float(prob)
                rows.append({
                    "predicted_index": idx,
                    "predicted_label": classes[idx],
                    "confidence": max(neg, pos),
                    "probability_class_0": neg,
                    "probability_class_1": pos,
                    f"prob_{classes[0]}": neg,
                    f"prob_{classes[1]}": pos,
                })
        elif logits.ndim == 2:
            if logits.shape[1] != len(classes):
                raise ValueError(
                    f"Salidas del cabezal: {logits.shape[1]}, clases del encoder: {len(classes)}"
                )
            if not torch.isfinite(logits).all():
                raise ValueError("Logits no finitos en clasificación multiclase")
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for prob in probs:
                idx = int(np.argmax(prob))
                row = {
                    "predicted_index": idx,
                    "predicted_label": str(label_encoder.inverse_transform([idx])[0]),
                    "confidence": float(prob[idx]),
                }
                row.update({f"prob_{nombre}": float(prob[j]) for j, nombre in enumerate(classes)})
                rows.append(row)
        else:
            raise ValueError(f"Dimensión de logits inesperada: {tuple(logits.shape)}")
    return pd.DataFrame(rows)


@torch.inference_mode()
def classify_prompts(
    prompts, tokenizer, encoder, classifier, label_encoder, device,
    batch_size: int = TRAINING_EMBEDDING_BATCH_SIZE,
    max_length: int = TRAINING_MAX_LENGTH,
    threshold: float = 0.5,
):
    """API de compatibilidad: vectoriza y clasifica una lista de textos."""
    vectores = create_embeddings(prompts, tokenizer, encoder, device, batch_size, max_length)
    resultados = classify_embeddings(vectores, classifier, label_encoder, device, batch_size, threshold)
    resultados.insert(0, "response", list(prompts))
    return resultados


def evaluateResponses(
    BASE_RESPONSES: str,
    responseColumn: str,
    ENCODER_PATH: str,
    CLASSIFIER_PATH: str | Path,
    LABELS_PATH: str | Path,
    BATCH_SIZE: int,
    MAX_LENGTH: int,
    DEVICE: torch.device,
) -> pd.DataFrame:
    """Función pública compatible con el script anterior.

    Los nulos se omiten aquí; el script principal preserva su fila de origen.
    """
    data = pd.read_csv(BASE_RESPONSES)
    if responseColumn not in data.columns:
        raise KeyError(f"Falta la columna {responseColumn}")
    responses = [x for x in data[responseColumn].tolist() if isinstance(x, str) and x.strip()]
    if not responses:
        raise ValueError("No hay respuestas válidas para evaluar")
    tokenizer, encoder = load_encoder(ENCODER_PATH, DEVICE)
    classifier = load_complete_classifier(CLASSIFIER_PATH, DEVICE)
    labels = load_label_encoder(LABELS_PATH)
    return classify_prompts(
        responses, tokenizer, encoder, classifier, labels, DEVICE,
        batch_size=BATCH_SIZE, max_length=MAX_LENGTH,
    )
