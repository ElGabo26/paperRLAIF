import gc
import torch
import pandas as pd

from time import perf_counter
from pathlib import Path
from tqdm import tqdm

from tools import testModel
from metrics import medir_recursos


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

REPETITIONS = 4

BATCH_SIZE = 32

MAX_NEW_TOKENS = 600

TEMPERATURE = 0.3

TOP_P = 0.9

ADD_GENERATION_PROMPT = True

RUTAOUTPUT = "/workspace/papaerRLAIF/codPreferences/bases"

RUTAPROMPTS = "/workspace/papaerRLAIF/codPreferences/prompts.csv"

MODELOS = [
    "/workspace/models/Qwen2.5-1.5B-Instruct",
    "/workspace/models/DeepSeek-R1-Distill-Qwen-1.5B",
    "/workspace/models/meta-llama/Llama-3.2-1B-Instruct"
]


# ============================================================
# 2. GENERACIÓN DE RESPUESTAS POR LOTES
# ============================================================

def makeResponseBatch(
    tokenizer,
    model,
    prompts,
    temperature=TEMPERATURE
):
    """
    Genera respuestas para múltiples prompts simultáneamente.

    Aplica el chat template correspondiente al tokenizer
    del modelo evaluado.

    Parámetros
    ----------
    tokenizer:
        Tokenizer de Hugging Face.

    model:
        Modelo de Hugging Face cargado en GPU.

    prompts:
        Lista de prompts.

    temperature:
        Temperatura de generación.

    Retorna
    -------
    list[str]:
        Una respuesta generada por cada prompt,
        manteniendo el orden original.
    """

    # --------------------------------------------------------
    # 1. Construir mensajes para cada prompt
    # --------------------------------------------------------

    mensajes = [
        [
            {
                "role": "user",
                "content": str(prompt)
            }
        ]
        for prompt in prompts
    ]

    # --------------------------------------------------------
    # 2. Aplicar chat template por modelo
    # --------------------------------------------------------

    textos = [
        tokenizer.apply_chat_template(
            mensaje,
            tokenize=False,
            add_generation_prompt=ADD_GENERATION_PROMPT
        )
        for mensaje in mensajes
    ]

    # --------------------------------------------------------
    # 3. Tokenizar el lote
    # --------------------------------------------------------

    inputs = tokenizer(
        textos,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False
    ).to(model.device)

    # Longitud compartida de entrada incluyendo padding.
    input_length = inputs["input_ids"].shape[1]

    # --------------------------------------------------------
    # 4. Generación simultánea
    # --------------------------------------------------------

    with torch.inference_mode():

        salida = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temperature,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id
        )

    # --------------------------------------------------------
    # 5. Extraer solamente tokens generados
    # --------------------------------------------------------

    generados = salida[:, input_length:]

    # --------------------------------------------------------
    # 6. Decodificar todas las respuestas
    # --------------------------------------------------------

    respuestas = tokenizer.batch_decode(
        generados,
        skip_special_tokens=True
    )

    return respuestas


# ============================================================
# 3. EVALUACIÓN DE UN LOTE Y MÉTRICAS
# ============================================================

def getresults_batch(tokenizer, model, prompts):
    """
    Ejecuta un lote completo de prompts.

    Retorna la misma estructura del código original:

        prompt
        response
        tiempo
        ram_mb
        gpu_mb

    El tiempo reportado representa el tiempo total
    del lote dividido para la cantidad de prompts.

    Las métricas de memoria se toman al finalizar
    la generación del lote.
    """

    # Sincronización previa para medir correctamente CUDA.
    torch.cuda.synchronize()

    t0 = perf_counter()

    # Generación de respuestas
    respuestas = makeResponseBatch(
        tokenizer=tokenizer,
        model=model,
        prompts=prompts,
        temperature=TEMPERATURE
    )

    # Esperar a que finalice la generación en GPU.
    torch.cuda.synchronize()

    t1 = perf_counter()

    # Medir recursos después de la generación.
    despues = medir_recursos()

    tiempo_total = t1 - t0

    tiempo_promedio = tiempo_total / len(prompts)

    resultados = [
        (
            prompt,
            respuesta,
            tiempo_promedio,
            despues["ram_mb"],
            despues["gpu_mb"]
        )
        for prompt, respuesta in zip(prompts, respuestas)
    ]

    return resultados


# ============================================================
# 4. EJECUCIÓN EXPERIMENTAL
# ============================================================

def main():

    if not torch.cuda.is_available():
        raise RuntimeError("No se detectó una GPU CUDA.")

    # Crear directorio de salida.
    Path(RUTAOUTPUT).mkdir(
        parents=True,
        exist_ok=True
    )

    # Cargar prompts
    base = pd.read_csv(RUTAPROMPTS)

    prompts1 = base["prompt"].tolist()

    total = len(prompts1)

    print("=" * 60)
    print("CONFIGURACIÓN EXPERIMENTAL")
    print("=" * 60)

    print(f"Total de prompts: {total}")
    print(f"Repeticiones: {REPETITIONS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Temperatura: {TEMPERATURE}")
    print(f"Top-p: {TOP_P}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ========================================================
    # 5. ITERAR SOBRE LOS MODELOS
    # ========================================================

    for ruta in MODELOS:

        print("\n" + "=" * 60)
        print(f"MODELO: {ruta}")
        print("=" * 60)

        # Cargar una sola vez por modelo.
        token, model = testModel(ruta)

        model.eval()

        # ----------------------------------------------------
        # Configurar padding para modelos autoregresivos
        # ----------------------------------------------------

        token.padding_side = "left"

        if token.pad_token is None:
            token.pad_token = token.eos_token

        # ----------------------------------------------------
        # Repeticiones
        # ----------------------------------------------------

        for i in range(REPETITIONS):

            print(f"\nRepetición_{i+1}")

            columnas = [
                "prompt",
                f"response_{i+1}",
                f"tiempo_{i+1}",
                f"ram_mb_{i+1}",
                f"gpu_mb_{i+1}"
            ]

            resultados = []

            tiempo_repeticion = perf_counter()

            # ------------------------------------------------
            # Dividir prompts en lotes
            # ------------------------------------------------

            with tqdm(
                total=total,
                desc=f"Repetición {i+1}",
                unit="prompt"
            ) as barra:

                for inicio in range(0, total, BATCH_SIZE):

                    lote = prompts1[
                        inicio:inicio + BATCH_SIZE
                    ]

                    # Generación paralela dentro de la GPU.
                    resultado_lote = getresults_batch(
                        token,
                        model,
                        lote
                    )

                    resultados.extend(resultado_lote)

                    barra.update(len(lote))

            # ------------------------------------------------
            # Guardar resultados
            # ------------------------------------------------

            r = pd.DataFrame(
                columns=columnas,
                data=resultados
            )

            name = "result" + Path(ruta).name

            archivo = (
                Path(RUTAOUTPUT) /
                f"{name}_{i+1}.csv"
            )

            r.to_csv(
                archivo,
                index=False
            )

            tiempo_total = (
                perf_counter() - tiempo_repeticion
            )

            print(f"RESPUESTAS: {i+1} REALIZADAS")
            print(f"Tiempo: {tiempo_total:.2f} segundos")
            print(f"Archivo: {archivo}")

        # ----------------------------------------------------
        # Liberar modelo antes de cargar el siguiente
        # ----------------------------------------------------

        del model
        del token

        gc.collect()

        torch.cuda.empty_cache()

        print("Memoria liberada.")


# ============================================================
# 6. MAIN
# ============================================================

if __name__ == "__main__":
    main()