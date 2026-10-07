"""
embeddings.py — Envuelve el modelo de embeddings en español.

Un EMBEDDING es un vector (una lista de ~768 números) que representa el
"significado" de un texto. Textos con significado parecido dan vectores
cercanos. Buscar por significado = comparar vectores, no palabras.

Usamos `hiiamsid/sentence_similarity_spanish_es`: un modelo entrenado
específicamente para similitud semántica en español. Importa que sea en
español — un modelo en inglés embebería mal el texto legal en castellano.

Este módulo aísla el modelo detrás de una interfaz mínima (`encode`) para que
el resto del código no dependa de la librería concreta. Si mañana cambiás de
modelo, tocás solo este archivo.
"""

from __future__ import annotations
from sentence_transformers import SentenceTransformer

# Nombre del modelo en HuggingFace. La primera vez que corras esto, se descarga
# (~400 MB) y queda cacheado localmente; las siguientes cargas son instantáneas.
MODELO = "hiiamsid/sentence_similarity_spanish_es"


class Embedder:
    def __init__(self, modelo: str = MODELO):
        # `SentenceTransformer` carga el modelo en memoria. Si tenés GPU la usa
        # sola; si no, corre en CPU (más lento pero perfectamente usable para
        # 1077 chunks — son segundos).
        self.model = SentenceTransformer(modelo)

    def encode(self, textos: list[str]) -> list[list[float]]:
        """
        Convierte una lista de textos en una lista de vectores.

        normalize_embeddings=True hace que todos los vectores tengan longitud 1.
        Esto es importante: con vectores normalizados, la similitud coseno (que
        es lo que usa Chroma para buscar) se vuelve un simple producto punto, y
        las distancias quedan en un rango consistente [0, 2]. Sin normalizar,
        los scores serían difíciles de interpretar y de umbralizar.
        """
        vectores = self.model.encode(
            textos,
            normalize_embeddings=True,
            show_progress_bar=len(textos) > 100,
        )
        # `encode` devuelve un numpy array; Chroma quiere listas de Python.
        return vectores.tolist()

    def encode_una(self, texto: str) -> list[float]:
        """Atajo para embeber una sola consulta."""
        return self.encode([texto])[0]
