# import gradio as gr
import openai
import pandas as pd
import os
import re
import unicodedata
import json
import time

from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity

openai.api_key = os.environ["OPEN_AI_KEY"]


def embed_text(conocimiento_df):
    conocimiento_df["Embedding"] = conocimiento_df["equipo"].apply(
        lambda x: get_embedding(x, engine="text-embedding-ada-002")
    )
    return conocimiento_df


def buscar(busqueda, datos, n_resultados=5):
    busqueda_embed = get_embedding(busqueda, engine="text-embedding-ada-002")
    datos["Similitud"] = datos["Embedding"].apply(
        lambda x: cosine_similarity(x, busqueda_embed)
    )
    datos = datos.sort_values("Similitud", ascending=False)
    return datos.iloc[:n_resultados][["equipo", "Similitud", "Embedding"]]


def datos_almacenados():
    datos = pd.read_csv("embeddings/equipos_embeddings.csv")
    datos["Embedding"] = datos["Embedding"].apply(lambda x: json.loads(x))
    return datos


equipos = [
    "Tigres vs Puebla",
    "Necaxa vs Atlas",
    "Mazatlán vs C. Monterrey",
    "Tijuana vs León",
    "Pachuca vs A. d. S. Luis",
    "Chivas vs C. Azul",
    "C. América vs P. UNAM",
    "T. FC vs F. Juárez",
    "S. Laguna vs Querétaro",
    "Puebla vs Tijuana",
    "F. Juárez vs C. América",
    "A. d. S. Luis vs Atlas",
    "C. Monterrey vs P. UNAM",
    "Chivas vs Mazatlán",
    "C. Azul vs S. Laguna",
    "T. FC vs Necaxa",
    "Querétaro vs Pachuca",
    "León vs Tigres",
]


def carga_equipo(equipo):
    df = pd.DataFrame([equipo], columns=["equipo"])
    datos = embed_text(df)
    return datos


if __name__ == "__main__":
    start_time = time.time()

    cargar_equipos = False

    if cargar_equipos:
        df_concatenado = pd.DataFrame()
        for equipo in equipos:
            equipo_emmbedding = carga_equipo(equipo)
            df_concatenado = pd.concat(
                [df_concatenado, equipo_emmbedding], ignore_index=True
            )

        df_concatenado.to_csv("equipos_embeddings.csv", index=False)

    datos = datos_almacenados()
    str.upper()
    busqueda = "Atletico de san luis FC v atlas de guadalajara"
    result = buscar(busqueda, datos)
    print("pregunta:", busqueda)
    print("resultado: ", result)

    end_time = time.time()
    execution_time = end_time - start_time
    print("Tiempo de ejecución:", execution_time, "segundos")
