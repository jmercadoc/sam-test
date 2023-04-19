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

openai.api_key = os.environ['OPEN_AI_KEY']


def eliminar_signos_diacriticos(texto):
    return ''.join((c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn'))



def limpiar_texto(texto, permitir_numeros=True): 
    texto_limpio = eliminar_signos_diacriticos(texto)
 
    if permitir_numeros:
        patron = r"[^a-zA-Z0-9\s]+"
    else:
        patron = r"[^a-zA-Z\s]+"
 
    texto_limpio = re.sub(patron, '', texto_limpio)
    texto_limpio = texto_limpio.lower()
    texto_limpio = re.sub(r"\s+", " ", texto_limpio).strip()

    return texto_limpio


def limpiar_nota(nota):
    nota_limpia = []
    for line in nota:
        nota_limpia.append(limpiar_texto(line, False))
    return nota_limpia


def leer_nota(nota):
    lineas = []
    with open(nota) as f:
        lineas = f.readlines()
    texto = ''.join([l for l in lineas if not l == "\n"])
    return texto


def embed_text(conocimiento_df):
    conocimiento_df['Embedding'] = conocimiento_df['texto'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
    return conocimiento_df



def buscar(busqueda, datos, n_resultados=5):
    busqueda_embed = get_embedding(busqueda, engine="text-embedding-ada-002")
    datos["Similitud"] = datos['Embedding'].apply(lambda x: cosine_similarity(x, busqueda_embed))
    datos = datos.sort_values("Similitud", ascending=False)
    return datos.iloc[:n_resultados][["texto", "Similitud", "Embedding"]]


def carga_nota(path):
    nota = leer_nota(path)
    df = pd.DataFrame([nota], columns=['texto'])
    datos = embed_text(df)

    datos_almacenados = datos_almacenados()
    df_concatenado = pd.concat([datos_almacenados, datos], ignore_index=True)
    df_concatenado.to_csv('embeddings.csv', index=False)

def datos_almacenados():
    datos = pd.read_csv('embeddings/embeddings.csv')
    datos["Embedding"] = datos["Embedding"].apply(lambda x: json.loads(x))
    return datos

if __name__ == "__main__":
    start_time = time.time()
    notas = ["burbuja_local.txt","marchas.txt"]
    cargar_notas = False
    
    if cargar_notas:
        for nota in notas:
            carga_nota(f'./notas/{nota}')

    
    datos = datos_almacenados()
      
    busqueda = 'mujeres el mundo'
    result = buscar(busqueda, datos) 
    print('pregunta:', busqueda)
    print('resultado: ', result) 
    end_time = time.time()
    execution_time = end_time - start_time

    print("Tiempo de ejecución:", execution_time, "segundos")