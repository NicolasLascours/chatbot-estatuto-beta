"""
retrieval.py — Motor de recuperación del RAG del Estatuto Docente.

Pipeline por pregunta:

  0. Atajo por identificador: si la pregunta menciona un número de artículo
     explícito ("artículo 114"), vamos DIRECTO a indice_padres.json. Es
     determinista y más confiable que cualquier búsqueda — no tiene sentido
     dejarlo en manos de embeddings o BM25 cuando el docente ya nos dio la
     clave exacta.

  1. Clasificación de capítulo + HyDE en UNA sola llamada a Groq: le pedimos
     al LLM que devuelva {capitulo_probable, confianza, documento_hipotetico}.
     Fusionamos ambas tareas porque HyDE ya requiere una llamada a LLM, así
     que la clasificación la obtenemos "gratis" en el mismo call.

  2. Búsqueda densa en Chroma, embebiendo el documento_hipotetico (no la
     pregunta original — ver docstring de _hyde_y_clasificar).

  3. Búsqueda BM25 (léxica) por DOS bandas: sobre el documento_hipotetico
     (mismo texto que el denso) y, desde esta sesión, también sobre la
     PREGUNTA ORIGINAL del docente. La segunda banda existe porque la
     primera, sola, hereda la varianza del HyDE — si el documento
     hipotético se va de tema en una tirada puntual (ver
     experimento_estabilidad_hyde.py), BM25(hyde) se va de tema con él y
     deja de cumplir su función real, que es pescar vocabulario jurídico
     exacto que el HyDE no adivina bien (ver paper RAGSUDOCU). BM25(pregunta)
     es invariante a esa varianza por construcción.

  4. Fusión de las TRES bandas (denso(hyde) + bm25(hyde) + bm25(pregunta))
     con Reciprocal Rank Fusion (RRF, k=60): se fusiona por ORDEN, no por
     magnitud, porque similitud coseno y score BM25 viven en escalas no
     comparables. Ver _buscar_hibrido_tres_bandas para la validación
     empírica (eval set completo, 31 preguntas: hit-rate empatado, MRR
     +7.7%, sin regresiones de hit-rate) que motivó este cambio.

  5. Filtro por capítulo BLANDO: si la confianza del clasificador es alta,
     restringimos la búsqueda (densa y BM25) a ese capítulo. Si la evidencia
     resultante es débil, hacemos fallback a búsqueda global.

  6. Umbral de "no encontrado": si ni siquiera después del fallback hay
     evidencia sólida, el motor devuelve que no encontró la respuesta en el
     Estatuto, en vez de forzar un contexto pobre hacia el generador.

  7. Small2Big: los chunks ganadores (a nivel inciso) se mapean a su
     artículo padre y se devuelve el texto COMPLETO (ley + reglamentación +
     estado_reglamentacion) desde indice_padres.json.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import chromadb
from rank_bm25 import BM25Okapi

from embeddings import Embedder

try:
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None

try:
    from google import genai as google_genai
    from google.genai import types as google_types
except ImportError:  # pragma: no cover
    google_genai = None
    google_types = None


# ----------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------

DIR_CHROMA = "./chroma_db"
RUTA_PADRES = "./indice_padres.json"
COLECCION = "estatuto"
MODELO_GROQ = "openai/gpt-oss-120b"

# k de Reciprocal Rank Fusion. 60 es el valor que usó el paper RAGSUDOCU;
# lo mantenemos como punto de partida y lo tratamos como hiperparámetro a
# revisar en la etapa de evaluación, no como una constante sagrada.
RRF_K = 60

# Umbral de similitud coseno (post-fallback) por debajo del cual el motor
# admite que no encontró evidencia sólida. Es un valor inicial CONSERVADOR
# a propósito: preferimos decir "no encontré" de más antes que alucinar.
# Se recalibra empíricamente en la etapa de evaluación con preguntas que
# sabemos que el Estatuto no responde.
UMBRAL_NO_ENCONTRADO = 0.35

# Umbral por debajo del cual una búsqueda filtrada por capítulo se considera
# "evidencia débil" y dispara el fallback a búsqueda global. Es más laxo que
# UMBRAL_NO_ENCONTRADO a propósito: acá no estamos decidiendo si hay
# respuesta, solo si vale la pena confiar en el filtro de capítulo.
UMBRAL_FALLBACK_CAPITULO = 0.45

TOP_K_CANDIDATOS = 15   # candidatos por canal (denso y BM25) antes de fusionar
TOP_ARTICULOS = 5       # artículos padre devueltos tras Small2Big

# Detecta menciones explícitas a un número de artículo, con o sin acento,
# con o sin abreviatura: "artículo 114", "articulo 114", "art. 114", "art 114".
PATRON_ARTICULO = re.compile(r"art(?:[íi]culo)?\.?\s*n?[ºo°]?\s*(\d+)", re.IGNORECASE)
# El atajo directo solo debe dispararse cuando la pregunta realmente pide "qué
# dice el artículo N", señalado por marco de autoridad ("según/conforme/de
# acuerdo ... artículo N"). Sin esto, cualquier escenario que mencione un
# artículo como referencia secuestraba la recuperación. Ver
# _buscar_por_identificador.
_MARCA_LOOKUP_DIRECTO = re.compile(r"seg[úu]n|conforme|de acuerdo", re.IGNORECASE)

# Contexto adicional para capítulos cuyo NOMBRE PELADO no alcanza para
# que el clasificador entienda qué preguntas responde. No es una lista de
# excepciones por pregunta del eval set -- es contenido real que el
# título del capítulo no transmite, encontrado comparando las fallas
# estables (mismo capítulo equivocado en las tres corridas) contra el
# texto real del Estatuto. Si en el futuro aparece una falla estable
# nueva y se confirma que es del mismo tipo (nombre insuficiente, no
# ambigüedad genuina), se agrega acá -- no como ejemplo en el prompt.
_CONTEXTO_EXTRA_CAPITULOS: dict[int, str] = {
    1: (
        " (aunque no tiene nombre propio en la ley, acá se define la "
        "situación de revista del docente: Activa o Pasiva)"
    ),
    2: (
        " (catálogo GENERAL de derechos y obligaciones del docente "
        "titular, y del suplente/provisional por extensión -- consultalo "
        "cuando la pregunta sea sobre si un derecho general (estabilidad, "
        "antigüedad, categoría escalafonaria, derecho a ascender, a "
        "permuta, a traslado) se MANTIENE o se PIERDE al atravesar una "
        "situación especial como una licencia, un Cambio de Funciones o "
        "una suplencia -- el MECANISMO detallado de cada uno de esos "
        "temas vive en su propio capítulo, acá solo está la enumeración "
        "general del derecho en sí, y suele hacer falta consultar los "
        "dos capítulos juntos. Ejemplo: la pregunta 'si un suplente es "
        "declarado en Tarea Pasiva por la Junta Médica, ¿qué "
        "consecuencias tiene en su estabilidad y fecha de cese?' NO "
        "tiene ninguna palabra que apunte a este capítulo, pero el "
        "derecho del suplente que se pierde o se mantiene está "
        "enumerado ACÁ, no en el capítulo de Estabilidad ni en el de "
        "Licencias, aunque la pregunta use esas palabras)"
    ),
    5: (
        " (regula cuándo se PIERDE o se CONSOLIDA la estabilidad en el "
        "cargo, en general -- pero cuando la pregunta es sobre las "
        "consecuencias de una situación puntual del suplente o "
        "provisional, como un Cambio de Funciones o una Tarea Pasiva, "
        "sobre su situación, la respuesta suele estar en el capítulo 2 "
        "o el 20, NO acá, aunque la pregunta use la palabra "
        "\"estabilidad\")"
    ),
    10: (
        " (acá se calcula el puntaje docente: títulos, antecedentes, "
        "años de servicio y promedio de calificaciones)"
    ),
    14: (
        " (regula el MECANISMO de asignación de funciones jerárquicas y "
        "concursos de ascenso, incluida la licencia del cargo de base "
        "para ejercer una función jerárquica transitoria, y el Cambio de "
        "Funciones como impedimento para acceder a esa función -- NO "
        "incluye la enumeración general del derecho a ascender, que está "
        "en el capítulo 2)"
    ),
    20: (
        " (además de las licencias en sentido estricto, incluye el "
        "Cambio de Funciones por disminución o pérdida de aptitud "
        "psicofísica, aunque no implique ausentarse del cargo)"
    ),
}

def _tokenizar(texto: str) -> list[str]:
    """Tokenización simple para BM25: minúsculas, solo secuencias alfanuméricas.

    No usamos un tokenizador lingüístico sofisticado (sin stemming, sin
    stopwords) porque BM25 acá cumple un rol puntual: pescar coincidencias
    léxicas exactas (números de artículo, términos jurídicos específicos)
    que el canal denso puede pasar por alto. Un tokenizador más agresivo
    podría, paradójicamente, borrar justo la señal que buscamos (ej. quitar
    dígitos o normalizar "artículo" perdería el "114").
    """
    return re.findall(r"\w+", texto.lower())


def _a_int_o_none(valor) -> int | None:
    """Normaliza el `capitulo_probable` que devuelve el LLM a int o None.

    El clasificador tiene que devolver el número de capítulo como int o
    null, pero en la práctica a veces vuelve como string ("14"), a veces
    con texto pegado ("Capítulo 14", "14 - DE LOS ASCENSOS"). Un string
    no matchea las claves int de _bm25_por_capitulo (cae a BM25 global) ni
    el filtro de metadata de Chroma (capitulo_num es int, el where con
    string no matchea nada, la similitud densa cae a 0 y se dispara un
    fallback que no hacía falta). Esto quedaba blindado solo en la rama
    experimental de bias; lo subimos al camino de producción, que es donde
    se consume de verdad, para no arrastrar esa varianza espuria a ninguna
    medición ni al comportamiento real. None si no hay ningún entero
    rescatable -- es lo mismo que "no estoy seguro del capítulo", que el
    pipeline ya sabe manejar (búsqueda global)."""
    if valor is None:
        return None
    if isinstance(valor, bool):  # un bool es int en Python, no lo queremos acá
        return None
    if isinstance(valor, int):
        return valor
    m = re.search(r"-?\d+", str(valor))
    return int(m.group(0)) if m else None


@dataclass
class Diagnostico:
    """Metadata de CÓMO se llegó a la respuesta — útil para debug y para que
    el docente (o vos, mientras evaluás) entienda cuánto confiar en el resultado."""
    modo: str  # "identificador_directo" | "recuperado" | "no_encontrado"
    capitulo_elegido: int | None = None  # primer capítulo de la lista, compatibilidad
    capitulos_elegidos: list[int] | None = None  # lista completa
    confianza_capitulo: str | None = None
    fallback_a_global: bool = False
    mejor_similitud: float | None = None
    documento_hipotetico: str | None = None


@dataclass
class ResultadoBusqueda:
    diagnostico: Diagnostico
    articulos: list[dict] = field(default_factory=list)


class MotorRecuperacion:
    def __init__(
        self,
        dir_chroma: str = DIR_CHROMA,
        ruta_padres: str = RUTA_PADRES,
        coleccion: str = COLECCION,
        modelo_groq: str = MODELO_GROQ,
        proveedor_llm: str = "groq",
        modelo_gemini: str = "gemini-2.5-flash",
        temperatura_clasificacion: float = 0.0,
    ):
        """
        proveedor_llm: "groq" (default, sin cambios de comportamiento) o
        "gemini". Afecta SOLO a _hyde_y_clasificar -- generacion.py sigue
        usando Groq siempre, no pasa por acá. Se agregó para poder correr
        experimento_filtro_capitulo.py con Gemini cuando la cuota diaria
        de Groq (TPD) está agotada, sin esperar al reset.

        temperatura_clasificacion: la temperature de _hyde_y_clasificar.
        AHORA el default es 0.0 (antes 0.3). Con 0.3 el HyDE variaba de
        corrida a corrida (+-12 puntos de hit-rate en ramas que dependen de
        la búsqueda global, con el mismo código), y esa varianza hacía que
        el mismo docente pudiera recuperar artículos distintos preguntando
        lo mismo dos veces -- inaceptable para un sistema de veracidad
        público, además de ensuciar cualquier medición. El diagnóstico de
        recall confirmó que 3 de 16 "faltas" eran puro ruido de temperatura
        (arts. 108, 110 y 121 aparecían o desaparecían según la tirada). Con
        0.0 la recuperación es reproducible. Se puede subir explícitamente
        para experimentos que quieran ver la varianza.
        """
        self._proveedor_llm = proveedor_llm
        self._temperatura_clasificacion = temperatura_clasificacion

        if proveedor_llm == "groq":
            if Groq is None:
                raise ImportError(
                    "Falta instalar el cliente de Groq: pip install groq"
                )
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "No se encontró la variable de entorno GROQ_API_KEY. "
                    "Conseguila en https://console.groq.com/keys y expórtala "
                    "antes de correr este script."
                )
            self._groq = Groq(api_key=api_key)
            self._modelo_groq = modelo_groq
        elif proveedor_llm == "gemini":
            if google_genai is None:
                raise ImportError(
                    "Falta instalar el cliente de Gemini: pip install google-genai"
                )
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "No se encontró la variable de entorno GOOGLE_API_KEY. "
                    "Exportala antes de correr este script (la misma que "
                    "usás en RAG_avanzado.ipynb)."
                )
            self._gemini = google_genai.Client(api_key=api_key)
            self._modelo_gemini = modelo_gemini
        else:
            raise ValueError(f"proveedor_llm desconocido: {proveedor_llm!r} (usar 'groq' o 'gemini')")

        # Embedder: el mismo modelo usado en ingest.py. Si no coincidiera,
        # los vectores de la consulta y del índice no vivirían en el mismo
        # espacio semántico y la búsqueda densa sería basura silenciosa.
        self._embedder = Embedder()

        # --- Chroma ---
        cliente = chromadb.PersistentClient(path=dir_chroma)
        self._col = cliente.get_collection(coleccion)

        # --- indice_padres.json (Small2Big) ---
        with open(ruta_padres, encoding="utf-8") as f:
            self._padres: dict = json.load(f)

        # --- Traer todo el corpus una vez para armar BM25 y lookups ---
        # include=["documents","metadatas"] trae también los ids por default.
        todo = self._col.get(include=["documents", "metadatas"])
        ids = todo["ids"]
        docs = todo["documents"]
        metas = todo["metadatas"]

        self._metadata_por_id: dict[str, dict] = dict(zip(ids, metas))

        # Mapa capitulo_num -> nombre, para el prompt del clasificador.
        self._capitulos: dict[int, str] = {}
        for m in metas:
            self._capitulos.setdefault(m["capitulo_num"], m["capitulo_nombre"])

        # --- BM25 global ---
        # Prefijamos "Artículo {numero}." al texto indexado SOLO para BM25.
        # Motivo: texto_busqueda (lo que se embebe) no contiene el número de
        # artículo literal (ver chunking.py: el prefijo ahí es el capítulo,
        # no el número). Sin este prefijo, BM25 tampoco podría matchear
        # "artículo 114" por token — quedaría tan ciego como el denso, que es
        # justo la falla que el paper RAGSUDOCU identificó y que la fusión
        # densa+BM25 está pensada para resolver.
        self._ids_bm25: list[str] = ids
        corpus_tokenizado = [
            _tokenizar(f"Artículo {m['art_numero']}. {doc}")
            for doc, m in zip(docs, metas)
        ]
        self._bm25_global = BM25Okapi(corpus_tokenizado)

        # --- BM25 por capítulo (precalculado, son ~24 capítulos) ---
        # Un BM25 filtrado post-hoc sobre los resultados del global no es lo
        # mismo que un BM25 indexado solo con los documentos del capítulo: el
        # IDF (qué tan "raro" es un término) cambia según el universo de
        # documentos. Precalcular por capítulo da IDF correcto para ese
        # subconjunto, no heredado del corpus completo.
        self._bm25_por_capitulo: dict[int, tuple[BM25Okapi, list[str]]] = {}
        por_cap: dict[int, list[tuple[str, list[str]]]] = {}
        for id_, tokens, m in zip(ids, corpus_tokenizado, metas):
            por_cap.setdefault(m["capitulo_num"], []).append((id_, tokens))
        for cap, pares in por_cap.items():
            ids_cap = [p[0] for p in pares]
            corpus_cap = [p[1] for p in pares]
            self._bm25_por_capitulo[cap] = (BM25Okapi(corpus_cap), ids_cap)

    # ------------------------------------------------------------------
    # Paso 0: atajo por identificador explícito
    # ------------------------------------------------------------------
    def _buscar_por_identificador(self, pregunta: str) -> dict | None:
        # El atajo devuelve SOLO el artículo citado y saltea toda la
        # recuperación. Eso es correcto cuando la pregunta es realmente "qué
        # dice el artículo N", pero secuestraba las preguntas de escenario que
        # mencionan un artículo como referencia (ej. "un docente que supera el
        # límite del Art. 28, ¿qué sanción?"), donde la respuesta vive en OTRO
        # artículo -- el diagnóstico de recall lo confirmó en N3-6 (devolvía
        # solo el 28, la respuesta estaba en el 30) y N3-10 (devolvía solo el
        # 65, faltaba el 18). Por eso ahora solo dispara cuando se cita
        # EXACTAMENTE UN artículo Y está en marco de autoridad
        # ("según/conforme el artículo N"). Si no, se deja pasar al pipeline
        # completo, que además fuerza el capítulo del artículo citado (ver
        # _capitulos_de_articulos_citados y buscar()).
        nums: list[str] = []
        for n in PATRON_ARTICULO.findall(pregunta):
            if n not in nums:
                nums.append(n)
        if len(nums) != 1 or _MARCA_LOOKUP_DIRECTO.search(pregunta) is None:
            return None
        art_id = f"art_{nums[0]}"
        if art_id not in self._padres:
            # Número que no existe en el Estatuto (172 artículos): seguimos el
            # pipeline normal por si la mención era parte de otra cosa.
            return None
        return self._padres[art_id]

    def _capitulos_de_articulos_citados(self, pregunta: str) -> list[int]:
        """Capítulos de los artículos que la pregunta menciona por número.
        Señal determinista y de alta confianza: si el docente nombró un
        artículo, su capítulo casi seguro es relevante aunque el clasificador
        LLM no lo proponga. buscar() los suma al filtro de capítulo siempre,
        incluso con confianza baja. Corrige las preguntas de escenario que
        citan un artículo cuyo capítulo el clasificador se perdía (N3-6: art.
        28 citado -> cap. 7, donde vive el 30; N3-10: art. 18 citado -> cap.
        5)."""
        caps: list[int] = []
        for n in PATRON_ARTICULO.findall(pregunta):
            padre = self._padres.get(f"art_{n}")
            if padre is not None:
                c = padre["capitulo_num"]
                if c not in caps:
                    caps.append(c)
        return caps

    def _llamar_llm_json(self, system: str, user: str) -> str:
        """Devuelve el texto crudo (se espera JSON) de una llamada al LLM
        configurado, sea Groq o Gemini. _hyde_y_clasificar es hoy el único
        llamador -- generacion.py tiene su propio cliente Groq aparte, no
        pasa por acá."""
        if self._proveedor_llm == "groq":
            resp = self._groq.chat.completions.create(
                model=self._modelo_groq,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=self._temperatura_clasificacion,
            )
            return resp.choices[0].message.content
        else:  # "gemini"
            resp = self._gemini.models.generate_content(
                model=self._modelo_gemini,
                contents=user,
                config=google_types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=self._temperatura_clasificacion,
                    response_mime_type="application/json",
                ),
            )
            return resp.text or ""

    # ------------------------------------------------------------------
    # Paso 1: clasificación de capítulo + HyDE, en una sola llamada
    # ------------------------------------------------------------------
    def _hyde_y_clasificar(self, pregunta: str) -> dict:
        """
        Le pedimos al LLM un JSON con:
          - capitulos_probables: lista de 1 a 3 números de capítulo, o lista
            vacía si no está seguro. Puede devolver más de un capítulo cuando
            la pregunta necesita contenido normativo de dos capítulos DISTINTOS
            para responderse por completo (ver system prompt para la
            distinción exacta con "dos pasos dentro del mismo capítulo").
          - confianza: "alta" | "baja"
          - documento_hipotetico: un párrafo redactado en TONO NORMATIVO que
            respondería la pregunta, como si fuera un fragmento real del
            Estatuto. Esto es HyDE: no es la respuesta real (el LLM puede
            inventar detalles), es un ancla léxica-semántica para la
            búsqueda densa. Nunca se lo mostramos al docente.

        NOTA DE TRANSICIÓN: el resto del pipeline (buscar(),
        _buscar_hibrido_tres_bandas, sondear_chunks.py) todavía consume
        "capitulo_probable" (singular), no "capitulos_probables". Por eso acá
        abajo derivamos capitulo_probable como el primer elemento de la lista
        -- ningún llamador existente se entera de este cambio todavía. Cuando
        se integre la búsqueda multi-capítulo (paso 4 del plan), los llamadores
        pasan a leer la lista completa y esta clave singular puede quedar
        como alias de compatibilidad o eliminarse.
        """
        # El nombre pelado de algunos capítulos no alcanza para que el
        # clasificador entienda qué preguntas responde -- _CONTEXTO_EXTRA_CAPITULOS
        # agrega aclaraciones puntuales sin convertirse en una lista de
        # excepciones por pregunta (ver docstring de la constante).
        lista_capitulos = "\n".join(
            f"{num}: {nombre}{_CONTEXTO_EXTRA_CAPITULOS.get(num, '')}"
            for num, nombre in sorted(self._capitulos.items())
        )
        system = (
            "Sos un asistente que prepara una consulta de recuperación de "
            "información sobre el Estatuto Docente de la Provincia de Buenos "
            "Aires (Ley 10.579). Dada la pregunta de un docente, respondé "
            "ÚNICAMENTE con un objeto JSON (sin texto adicional, sin markdown) "
            "con estas claves:\n"
            '  "capitulos_probables": una lista de uno a tres números de '
            "capítulo de la siguiente lista, ordenados de más a menos "
            "relevante, o una lista vacía si no estás razonablemente seguro. "
            "Devolvé MÁS DE UNO solo cuando la pregunta necesita contenido "
            "normativo de dos capítulos DISTINTOS para responderse por "
            "completo, no alcanza con que la pregunta mencione dos "
            "situaciones o dos pasos si ambos se regulan dentro del mismo "
            "capítulo.\n"
            "Ejemplo de necesitar dos capítulos: una pregunta sobre qué pasa "
            "con el sueldo y la antigüedad del cargo de base de un docente "
            "al que se asigna transitoriamente una función jerárquica, ahí "
            "hace falta el capítulo de Ascensos para la asignación en sí Y "
            "el capítulo que regula el cómputo de antigüedad, son normas "
            "separadas. Otro ejemplo: una pregunta sobre si una licencia "
            "puede ser causal de cese de una función jerárquica transitoria, "
            "ahí hace falta el capítulo de Licencias Y el de Ascensos, "
            "porque la reglamentación de licencias remite explícitamente a "
            "un artículo del capítulo de Ascensos para resolver esa duda.\n"
            "Ejemplo de NO necesitar dos capítulos: una pregunta sobre "
            "cuántas horas cátedra debe relevar un docente que ganó un "
            "concurso de Vicedirector, ahí el concurso y el relevo están "
            "regulados los dos dentro del mismo capítulo de Ascensos, un "
            "solo capítulo alcanza aunque la pregunta mencione dos pasos.\n"
            "IMPORTANTE: si la pregunta menciona explícitamente un número de "
            "ARTÍCULO (\"según el Artículo 13\", \"conforme al Art. 75\"), ese "
            "número identifica un ARTÍCULO, no un capítulo -- nunca lo uses "
            "como número de capítulo salvo que coincida por casualidad con "
            "el capítulo real al que pertenece ese artículo, que vos no "
            "sabés de antemano.\n"
            '  "confianza": "alta" si estás bastante seguro de los capítulos, '
            '"baja" en caso contrario,\n'
            '  "documento_hipotetico": un párrafo breve (3-5 líneas), '
            "redactado en el mismo registro normativo/legal que el Estatuto, "
            "que respondería la pregunta. Es una respuesta HIPOTÉTICA para "
            "mejorar la búsqueda semántica, no necesita ser jurídicamente "
            "exacta.\n\n"
            f"Capítulos disponibles:\n{lista_capitulos}"
        )
        crudo = self._llamar_llm_json(system, pregunta)
        try:
            data = json.loads(crudo)
        except (json.JSONDecodeError, TypeError):
            data = {}

        # Normalizamos cada elemento de la lista con el mismo helper que ya
        # blindaba capitulo_probable contra strings sueltos o texto pegado
        # ("14", "Capítulo 14", "14 - DE LOS ASCENSOS"). Filtramos None,
        # deduplicamos preservando orden (por si el LLM repite un capítulo),
        # y cortamos a 3 aunque el LLM mande más.
        crudos = data.get("capitulos_probables")
        if not isinstance(crudos, list):
            crudos = []
        capitulos_probables: list[int] = []
        for v in crudos:
            cap = _a_int_o_none(v)
            if cap is not None and cap not in capitulos_probables:
                capitulos_probables.append(cap)
        capitulos_probables = capitulos_probables[:3]

        return {
            "capitulos_probables": capitulos_probables,
            # Compatibilidad con todo lo que hoy lee "capitulo_probable"
            # (singular) -- primer elemento de la lista, o None si vino vacía.
            "capitulo_probable": capitulos_probables[0] if capitulos_probables else None,
            "confianza": data.get("confianza", "baja"),
            "documento_hipotetico": data.get("documento_hipotetico") or pregunta,
        }

    # ------------------------------------------------------------------
    # Pasos 2-4: búsqueda densa + BM25 + fusión RRF
    # ------------------------------------------------------------------
    def _rankings_crudos(
        self, texto_consulta: str, capitulo: int | None
    ) -> tuple[list[str], list[str], float]:
        """
        Devuelve (ids_densos, ids_bm25_top, mejor_similitud), SIN fusionar.
        Extraído de _buscar_hibrido para poder reusar el mismo cómputo desde
        _buscar_hibrido_global_con_bias (ver más abajo) sin duplicar lógica
        ni tocar el comportamiento de _buscar_hibrido.
        """
        vector = self._embedder.encode_una(texto_consulta)

        where = {"capitulo_num": capitulo} if capitulo is not None else None
        denso = self._col.query(
            query_embeddings=[vector],
            n_results=TOP_K_CANDIDATOS,
            where=where,
            include=["distances"],
        )
        ids_densos = denso["ids"][0]
        distancias = denso["distances"][0]
        similitudes = [1 - d for d in distancias]
        mejor_similitud = max(similitudes) if similitudes else 0.0

        tokens_consulta = _tokenizar(texto_consulta)
        if capitulo is not None and capitulo in self._bm25_por_capitulo:
            bm25, ids_bm25 = self._bm25_por_capitulo[capitulo]
        else:
            bm25, ids_bm25 = self._bm25_global, self._ids_bm25
        scores_bm25 = bm25.get_scores(tokens_consulta)
        ranking_bm25 = sorted(
            range(len(scores_bm25)), key=lambda i: scores_bm25[i], reverse=True
        )[:TOP_K_CANDIDATOS]
        ids_bm25_top = [ids_bm25[i] for i in ranking_bm25]

        return ids_densos, ids_bm25_top, mejor_similitud

    def _buscar_hibrido(
        self, texto_consulta: str, capitulo: int | None
    ) -> tuple[list[str], float]:
        """
        Devuelve (chunk_ids fusionados por RRF, mejor similitud densa cruda).

        La mejor similitud densa se devuelve aparte porque el score RRF no
        tiene escala interpretable (es una suma de 1/rank) — no sirve para
        decidir umbrales de confianza, solo para ORDENAR. Para eso usamos la
        similitud coseno cruda del mejor resultado denso.
        """
        ids_densos, ids_bm25_top, mejor_similitud = self._rankings_crudos(
            texto_consulta, capitulo
        )
        fusionados = self._rrf(ids_densos, ids_bm25_top)
        return fusionados, mejor_similitud

    def _buscar_hibrido_tres_bandas(
        self, pregunta: str, hyde_doc: str, capitulos: list[int] | None
    ) -> tuple[list[str], float]:
        """
        Fusión de PRODUCCIÓN — denso(hyde_doc) + bm25(hyde_doc) + bm25(pregunta),
        corrida UNA VEZ POR CADA capítulo de la lista y fusionada con una sola
        llamada a _rrf.

        (mismo docstring que ya tenías hasta acá, sin cambios en la parte de
        la justificación del bm25(pregunta) -- lo que sigue es lo nuevo.)

        CAMBIO DE ESTA SESIÓN: `capitulos` ahora es una lista, no un capítulo
        singular. Cuando trae más de un elemento, se corre _rankings_crudos y
        _bm25_top_ids UNA VEZ POR CADA capítulo, cada uno sobre su propio
        universo con el IDF correcto de _bm25_por_capitulo (no un IDF global
        con bias, que perdería la precisión por la que se separó
        _bm25_por_capitulo capítulo por capítulo en primer lugar), y se
        fusionan TODAS las listas resultantes con una sola llamada a _rrf.
        RRF fusiona por ORDEN, no por magnitud, así que combinar rankings que
        vinieron de universos de distinto tamaño no necesita ningún ajuste
        extra -- es la misma propiedad que ya usa _buscar_hibrido_doble_hyde
        para fusionar rankings de dos documentos hipotéticos distintos.

        capitulos=None o lista vacía -> comportamiento idéntico a antes de
        este cambio, una sola búsqueda global (cap=None).

        Esto es lo que resuelve la familia del art. 75 (ver docstring del
        módulo, _hyde_y_clasificar y las notas de sesión): antes, un artículo
        cuyo capítulo nunca ganaba el `capitulo_probable` singular quedaba
        afuera del universo de búsqueda sin ninguna chance de aparecer, así
        fuera el segundo o tercer capítulo más relevante para el clasificador.
        Ahora alcanza con que ese capítulo entre en ALGUNA posición de la
        lista que devuelve _hyde_y_clasificar para que su universo entero se
        busque y compita en la fusión final. Confirmado en clasificación
        sola contra N2-2 y N3-3 (ver clasificacion_eval_resultados.json de
        esta sesión) -- falta confirmar el efecto en la búsqueda real, que es
        justo lo que habilita este cambio.
        """
        # CAMBIO DE FONDO (impulso aditivo estilo RAGSUDOCU, Ec. 2): SIEMPRE se
        # corre una pasada global (cap=None) ADEMÁS de una por cada capítulo
        # probable. Antes esto filtraba: si el capítulo correcto no estaba en la
        # lista, su universo NO se buscaba y el artículo quedaba excluido, sin
        # ninguna chance de aparecer -- la causa raíz de las fallas de
        # clasificación (N1-3 que perdió Licencias, las cruzadas de dos
        # institutos que perdían el segundo). Ahora la pasada global garantiza
        # que NINGÚN artículo se excluya: todos compiten. Los capítulos
        # probables no excluyen a los demás, solo reciben más representación en
        # la fusión, porque sus chunks aparecen en la pasada global Y en su
        # pasada por capítulo -- o sea en más rankings, así que el RRF los
        # empuja hacia arriba. Es el impulso aditivo del paper (un valor
        # inferido empuja el orden pero no saca nada de la lista), logrado por
        # multiplicidad de rankings en vez de un bonus con alpha a calibrar. La
        # pasada por capítulo conserva el IDF correcto de _bm25_por_capitulo.
        capitulos_a_correr = [None] + [c for c in (capitulos or []) if c is not None]
        rankings: list[list[str]] = []
        mejor_similitud = 0.0
        for cap in capitulos_a_correr:
            ids_densos, ids_bm25_hyde, sim = self._rankings_crudos(hyde_doc, cap)
            ids_bm25_pregunta = self._bm25_top_ids(pregunta, cap)
            rankings.append(ids_densos)
            rankings.append(ids_bm25_hyde)
            rankings.append(ids_bm25_pregunta)
            mejor_similitud = max(mejor_similitud, sim)
        fusionados = self._rrf(*rankings)
        return fusionados, mejor_similitud

    def _buscar_hibrido_global_con_bias(
        self, texto_consulta: str, capitulo_bias: int | None, alpha: float
    ) -> tuple[list[str], float]:
        """
        EXPERIMENTAL, no usado por buscar() todavía -- solo para
        experimento_filtro_capitulo.py, rama de bias aditivo.

        A diferencia de _buscar_hibrido, NUNCA restringe la búsqueda por
        capítulo (denso y BM25 corren siempre sobre el corpus completo).
        En cambio, a los chunks cuyo capitulo_num coincide con
        capitulo_bias se les suma un bonus aditivo de tamaño
        alpha * unidad, unidad = 1/(RRF_K+1) -- la misma idea que el
        estado de diálogo con procedencia del paper RAGSUDOCU (Ecuación 2):
        un valor inferido no excluye documentos, solo empuja el orden.
        alpha=0 es idéntico a no tener ningún bias (fusión RRF pura,
        global). El bonus es FLAT, no depende de la posición del chunk en
        ninguno de los dos rankings -- valen las mismas RRF_K unidades
        para cualquier chunk del capítulo elegido, sea que haya aparecido
        primero o último en denso/BM25.
        """
        ids_densos, ids_bm25_top, mejor_similitud = self._rankings_crudos(
            texto_consulta, None  # siempre global, nunca restringido
        )
        scores: dict[str, float] = {}
        for ranking in (ids_densos, ids_bm25_top):
            for rank, id_ in enumerate(ranking):
                scores[id_] = scores.get(id_, 0.0) + 1.0 / (RRF_K + rank + 1)

        if capitulo_bias is not None and alpha:
            try:
                capitulo_bias = int(capitulo_bias)
            except (TypeError, ValueError):
                capitulo_bias = None
        if capitulo_bias is not None and alpha:
            unidad = 1.0 / (RRF_K + 1)
            bonus = alpha * unidad
            for id_ in scores:
                meta = self._metadata_por_id.get(id_, {})
                if meta.get("capitulo_num") == capitulo_bias:
                    scores[id_] += bonus

        fusionados = sorted(scores, key=scores.get, reverse=True)
        return fusionados, mejor_similitud

    @staticmethod
    def _scores_rrf(*rankings: list[str]) -> dict[str, float]:
        """Los scores RRF crudos, sin ordenar -- _rrf ordena esto mismo.
        Separado para que el disparador de margen pueda mirar la
        DIFERENCIA entre scores, no solo el orden final."""
        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, id_ in enumerate(ranking):
                scores[id_] = scores.get(id_, 0.0) + 1.0 / (RRF_K + rank + 1)
        return scores

    @staticmethod
    def _rrf(*rankings: list[str]) -> list[str]:
        """Reciprocal Rank Fusion: score(id) = Σ 1/(RRF_K + rank), rank 0-based."""
        scores = MotorRecuperacion._scores_rrf(*rankings)
        return sorted(scores, key=scores.get, reverse=True)

    def _top_articulos_de_ranking(self, chunk_ids: list[str], n: int) -> list[str]:
        """Los primeros n art_id distintos, en el orden en que aparece su
        primer chunk en chunk_ids -- como el chunk_ids ya viene ordenado
        por score, el primer chunk de un artículo es su chunk de mejor
        score, así que esto es "los n artículos con mejor chunk", sin
        tener que reimplementar el agrupamiento que ya hace
        _expandir_a_articulos."""
        vistos: list[str] = []
        for cid in chunk_ids:
            meta = self._metadata_por_id.get(cid)
            if not meta:
                continue
            art_id = meta["art_id"]
            if art_id not in vistos:
                vistos.append(art_id)
            if len(vistos) >= n:
                break
        return vistos

    def _margen_top2(
        self, chunk_ids: list[str], scores: dict[str, float], arts_top2: list[str]
    ) -> float:
        """score del mejor chunk del artículo #1 menos el del artículo #2.
        Un margen grande = ranking decisivo (el #1 le saca mucha ventaja
        al #2); un margen chico = ranking dudoso, el #2 le pisa los
        talones al #1. Si hay menos de 2 artículos candidatos, devuelve
        infinito -- no hay con qué dudar."""
        if len(arts_top2) < 2:
            return float("inf")

        def _mejor_score(art_id: str) -> float:
            mejor = 0.0
            for cid in chunk_ids:
                meta = self._metadata_por_id.get(cid)
                if meta and meta["art_id"] == art_id:
                    return scores[cid]  # el primero que aparece ES el mejor
            return mejor

        return _mejor_score(arts_top2[0]) - _mejor_score(arts_top2[1])

    def _buscar_con_disparador_margen(
        self,
        texto_consulta: str,
        capitulo: int | None,
        confianza: str | None,
        margen_minimo_extra: float = 0.0,
    ) -> tuple[list[str], float, dict]:
        """
        EXPERIMENTAL, no usado por buscar() todavía -- pensado para
        reemplazar el Paso 1 + Paso 5 actuales (filtro si confianza=="alta",
        fallback solo si la similitud cruda es baja).

        El problema que ataca: `confianza` no discrimina -- la sesión de
        hoy encontró que las fallas de recuperación reportan "alta" casi
        siempre. Este disparador no confía en lo que el modelo DICE sobre
        sí mismo, mira si el propio ranking fusionado, filtrado contra
        global, coincide o no en su artículo top-1:

          - Si coinciden en el top-1, se usa el FILTRADO (su ventaja de
            precisión se mantiene). Nota de sesión: se probó cambiar
            esto a "usar el global" (la idea era que un segundo artículo
            necesario en otro capítulo, caso real N2-2, art_75, quedaba
            excluido por el filtro aunque el top-1 coincidiera) y el
            resultado empírico fue NEGATIVO sobre el eval set completo,
            64% a 56% de hit-rate a nivel de artículo, filtrado contra
            global, en preguntas de un solo tema el filtrado tiene menos
            competencia dentro de su capítulo y ubica mejor a artículos
            secundarios reales; el global reintroduce ruido de otros
            capítulos que los desplaza del top 5. La ganancia sobre casos
            multi-artículo tampoco se sostuvo de forma confiable en las
            pruebas puntuales. Revertido -- el problema de preguntas que
            necesitan dos artículos en capítulos distintos sigue abierto,
            pero no se resuelve tocando esta rama, necesita algo que
            detecte esa necesidad y dispare búsquedas separadas por tema,
            no un cambio en cómo se fusiona un único ranking.
          - Si discrepan, se compara qué tan DECISIVO es cada ranking (el
            margen entre su artículo #1 y #2 -- ver _margen_top2). Gana
            el más decisivo.
          - Si el filtrado pierde esa comparación, no se descarta el
            capítulo elegido de un saque (eso es lo que le costaba caro al
            filtro dubro cuando el clasificador se equivocaba) -- se cae
            al bias aditivo (alpha=1.0) en vez de al filtro duro NI a la
            búsqueda global pura, así el capítulo elegido sigue pesando
            sin poder excluir al artículo correcto si está en otro lado.

        margen_minimo_extra (default 0.0, sin cambio de comportamiento):
        no alcanza con que margen_global sea apenas mayor que margen_filtro
        para activar el bias, tiene que ganarle por más de este valor.
        Se agregó después de ver, en la corrida con temperature=0, dos
        casos (no hipotéticos, reales) donde el filtro ya acertaba y el
        bias por discrepancia lo arruinaba por completo -- la sospecha es
        que esos casos tenían una diferencia de margen chica, un umbral
        más exigente podría filtrarlos sin perder los rescates genuinos.
        Sin verificar todavía, para eso es el barrido.

        Cuesta el DOBLE de búsquedas locales que _buscar_hibrido (filtrada
        + global, siempre, no solo cuando hace falta) -- cero llamadas a
        Groq extra, la única llamada sigue siendo _hyde_y_clasificar.
        """
        ids_densos_g, ids_bm25_g, sim_global = self._rankings_crudos(texto_consulta, None)
        scores_global = self._scores_rrf(ids_densos_g, ids_bm25_g)
        fusionados_global = sorted(scores_global, key=scores_global.get, reverse=True)

        capitulo_filtro = capitulo if confianza == "alta" else None
        if capitulo_filtro is None:
            return fusionados_global, sim_global, {"decision": "global (confianza baja o sin capítulo)"}

        ids_densos_f, ids_bm25_f, sim_filtro = self._rankings_crudos(texto_consulta, capitulo_filtro)
        scores_filtro = self._scores_rrf(ids_densos_f, ids_bm25_f)
        fusionados_filtro = sorted(scores_filtro, key=scores_filtro.get, reverse=True)

        top2_filtro = self._top_articulos_de_ranking(fusionados_filtro, 2)
        top2_global = self._top_articulos_de_ranking(fusionados_global, 2)

        margen_filtro = self._margen_top2(fusionados_filtro, scores_filtro, top2_filtro)
        margen_global = self._margen_top2(fusionados_global, scores_global, top2_global)

        if top2_filtro and top2_global and top2_filtro[0] == top2_global[0]:
            return fusionados_filtro, sim_filtro, {
                "decision": "coincide", "articulo_top": top2_filtro[0],
                "margen_filtro": margen_filtro, "margen_global": margen_global,
            }

        margen_filtro = self._margen_top2(fusionados_filtro, scores_filtro, top2_filtro)
        margen_global = self._margen_top2(fusionados_global, scores_global, top2_global)

        if margen_filtro >= margen_global - margen_minimo_extra:
            return fusionados_filtro, sim_filtro, {
                "decision": "filtro (margen mayor o igual)",
                "margen_filtro": margen_filtro, "margen_global": margen_global,
            }

        fusionados_bias, sim_bias = self._buscar_hibrido_global_con_bias(
            texto_consulta, capitulo, 1.0
        )
        return fusionados_bias, sim_bias, {
            "decision": "bias (discrepancia, global más decisivo)",
            "margen_filtro": margen_filtro, "margen_global": margen_global,
        }

    # ------------------------------------------------------------------
    # EXPERIMENTAL — varianza del HyDE (temperatura 0.3): ver
    # experimento_estabilidad_hyde.py. Ninguno de estos tres métodos lo
    # usa buscar(), son variantes candidatas para atacar el problema real
    # encontrado en sesión: con la misma pregunta, distintas tiradas de
    # _hyde_y_clasificar pueden anclar la búsqueda densa en un inciso
    # vecino semánticamente parecido (ej. "atención de familiar enfermo"
    # en vez de "enfermedad propia del docente", los dos primeros incisos
    # del art. 114) y el chunk correcto directamente no entra al pool
    # fusionado — no es un problema de presupuesto ni de ranking bajo,
    # es ausencia total en una tirada puntual.
    # ------------------------------------------------------------------

    def _buscar_hibrido_peso_bm25(
        self, texto_consulta: str, capitulo: int | None, peso_bm25: float = 2.0
    ) -> tuple[list[str], float]:
        """Variante C — la misma fusión RRF de _buscar_hibrido, pero con
        el ranking de BM25 pesado peso_bm25 veces más que el denso al
        sumar los términos 1/(RRF_K + rank + 1).

        No hace ningún llamado extra a Groq ni a Chroma más allá de los
        que ya hacía _buscar_hibrido -- reutiliza _rankings_crudos, solo
        cambia cómo se combinan los dos rankings ya calculados. La idea:
        BM25 no depende de que el LLM haya redactado un buen documento
        hipotético, depende de coincidencia léxica directa contra la
        pregunta original tokenizada, así que una mala tirada de HyDE
        (que arruina al canal denso, anclado en el documento hipotético)
        no debería arruinar tan fácil al canal BM25 si tiene más peso en
        la fusión final.

        peso_bm25=1.0 es idéntico a _buscar_hibrido (mismo resultado,
        más lento porque no comparte código, pero matemáticamente
        equivalente) -- se deja explícito así en vez de tener una rama
        aparte para peso_bm25=1.0, para no bifurcar comportamiento según
        el valor del parámetro."""
        ids_densos, ids_bm25_top, mejor_similitud = self._rankings_crudos(
            texto_consulta, capitulo
        )
        scores: dict[str, float] = {}
        for rank, id_ in enumerate(ids_densos):
            scores[id_] = scores.get(id_, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, id_ in enumerate(ids_bm25_top):
            scores[id_] = scores.get(id_, 0.0) + peso_bm25 / (RRF_K + rank + 1)
        fusionados = sorted(scores, key=scores.get, reverse=True)
        return fusionados, mejor_similitud

    def _hyde_y_clasificar_doble(self, pregunta: str) -> tuple[dict, dict]:
        """Variante B, parte 1 — dos llamados independientes a
        _hyde_y_clasificar con la misma pregunta (misma temperatura de
        producción, 0.3 salvo que se instancie el motor con otra). Dos
        tiradas del dado en vez de una, para no depender de que la única
        tirada haya sido buena.

        No colapsa nada acá -- devuelve las dos clasificaciones
        completas (puede que capitulo_probable difiera entre ellas, poco
        común pero posible) para que el llamador decida cómo fusionar."""
        return self._hyde_y_clasificar(pregunta), self._hyde_y_clasificar(pregunta)

    def _buscar_hibrido_doble_hyde(
        self, doc1: str, doc2: str, capitulo: int | None
    ) -> tuple[list[str], float]:
        """Variante B, parte 2 — corre _rankings_crudos una vez por cada
        documento hipotético (2 búsquedas densas + 2 BM25, mismo capítulo
        para las cuatro) y fusiona las CUATRO listas con RRF estándar,
        sin peso extra a ninguna. BM25 sobre el mismo capitulo con dos
        documentos hipotéticos distintos va a dar rankings casi
        idénticos entre sí (BM25 no usa el documento hipotético como
        ancla semántica, tokeniza la pregunta... en realidad sí usa
        texto_consulta como entrada a _tokenizar, así que dos HyDE
        distintos SÍ pueden dar tokens BM25 distintos -- no es
        redundante, aunque menos sensible que el canal denso al cambio
        de documento hipotético).

        mejor_similitud devuelve el máximo entre las dos corridas densas,
        no el promedio -- para mantener el mismo criterio que
        UMBRAL_NO_ENCONTRADO usa en producción (la mejor evidencia
        encontrada, no la evidencia típica)."""
        ids_densos_1, ids_bm25_1, sim_1 = self._rankings_crudos(doc1, capitulo)
        ids_densos_2, ids_bm25_2, sim_2 = self._rankings_crudos(doc2, capitulo)
        scores = self._scores_rrf(ids_densos_1, ids_bm25_1, ids_densos_2, ids_bm25_2)
        fusionados = sorted(scores, key=scores.get, reverse=True)
        return fusionados, max(sim_1, sim_2)

    # ------------------------------------------------------------------
    # EXPERIMENTAL — desacople de la entrada de BM25 respecto del denso.
    #
    # Hallazgo que motiva esto: en producción, _rankings_crudos tokeniza
    # `texto_consulta` para BM25, y buscar() le pasa el documento
    # hipotético del HyDE, no la pregunta original. O sea, el canal léxico
    # también depende del HyDE, cuando su razón de ser (ver docstring del
    # módulo y del paper RAGSUDOCU) es pescar vocabulario que el HyDE no
    # adivina bien. Consecuencias: (1) una tirada mala de HyDE degrada los
    # DOS canales a la vez, no solo el denso, lo que explica por qué
    # _buscar_hibrido_peso_bm25 no dio ventaja contra la varianza del HyDE
    # (pesar más un canal que igual depende del HyDE no lo vuelve inmune);
    # (2) preguntas cuyo vocabulario coincide con el artículo correcto pero
    # cuyo HyDE se fue de tema pierden esa coincidencia léxica -- caso
    # sospechado del art. 75 (N2-2 comparte docente/cargo/horas/cátedra/
    # jerárquic con art_75::reglamentacion::7::4, tokens que BM25 sobre la
    # pregunta recuperaría y que se pierden si BM25 mira un HyDE anclado en
    # licencias o incompatibilidad).
    #
    # Ninguno de estos métodos lo usa buscar(); son para
    # experimento_bm25_desacoplado.py. Costo: cero llamadas extra a Groq
    # (el HyDE ya viene computado), a lo sumo un get_scores de BM25 más por
    # búsqueda, barato.
    # ------------------------------------------------------------------

    def _bm25_top_ids(self, texto: str, capitulo: int | None) -> list[str]:
        """Top-K ids del canal BM25 para un texto arbitrario y un capítulo
        (o global si capitulo es None). Es la mitad léxica de
        _rankings_crudos aislada, para poder alimentarla con un texto
        distinto al que se embebe en el denso. Se deja duplicada a
        propósito, sin refactorizar _rankings_crudos, para no tocar en nada
        el camino de producción mientras esto sea experimental -- mismo
        criterio que las otras ramas EXPERIMENTAL de este archivo."""
        tokens = _tokenizar(texto)
        if capitulo is not None and capitulo in self._bm25_por_capitulo:
            bm25, ids_bm25 = self._bm25_por_capitulo[capitulo]
        else:
            bm25, ids_bm25 = self._bm25_global, self._ids_bm25
        scores_bm25 = bm25.get_scores(tokens)
        ranking = sorted(
            range(len(scores_bm25)), key=lambda i: scores_bm25[i], reverse=True
        )[:TOP_K_CANDIDATOS]
        return [ids_bm25[i] for i in ranking]

    def _buscar_experimental_bm25(
        self,
        pregunta: str,
        hyde_doc: str,
        capitulo: int | None,
        confianza: str | None,
        modo: str,
    ) -> tuple[list[str], float, bool]:
        """Reproduce los pasos 2-5 de buscar() (fusión + filtro por capítulo
        + fallback a global si la evidencia es débil), pero controlando de
        qué texto sale el canal BM25 según `modo`. El denso siempre embebe
        el hyde_doc, como en producción -- lo único que se mueve es la
        entrada léxica, así la comparación aísla ese cambio y nada más.

        No llama a _hyde_y_clasificar: recibe hyde_doc/capitulo/confianza ya
        calculados, para que el harness pueda pasarle la MISMA tirada de
        HyDE a las cuatro variantes dentro de una repetición y que la
        varianza del HyDE no se cuele en la comparación entre modos (se
        muestrea aparte, repitiendo la corrida entera).

        modo:
          "produccion"  -> denso(hyde) + bm25(hyde). OJO: a partir de esta
                           sesión esto ya NO es lo que corre en buscar() --
                           buscar() adoptó "tres_bandas" como default (ver
                           _buscar_hibrido_tres_bandas) tras la validación
                           sobre las 31 preguntas del eval set. Se mantiene
                           el nombre "produccion" en este harness sin
                           renombrar, por compatibilidad con los resultados
                           ya guardados de experimento_bm25_desacoplado.py y
                           experimento_tres_bandas_eval_completo.py -- leer
                           como "línea base de dos bandas (la fusión vieja)",
                           no como "lo que corre hoy".
          "pregunta"    -> denso(hyde) + bm25(pregunta). BM25 invariante al
                           HyDE.
          "concat"      -> denso(hyde) + bm25(pregunta + " " + hyde). Conserva
                           la señal léxica del HyDE y le suma la de la
                           pregunta.
          "tres_bandas" -> denso(hyde) + bm25(hyde) + bm25(pregunta). Fusión
                           RRF a tres canales, la variante más conservadora
                           (no saca nada, solo agrega la señal invariante).
                           Es la que adoptó buscar() en producción.

        Devuelve (chunk_ids fusionados, mejor_similitud densa, hubo_fallback).
        """
        capitulo_filtro = capitulo if confianza == "alta" else None

        def _fusion(cap: int | None) -> tuple[list[str], float]:
            ids_densos, ids_bm25_hyde, sim = self._rankings_crudos(hyde_doc, cap)
            if modo == "produccion":
                rankings = [ids_densos, ids_bm25_hyde]
            elif modo == "pregunta":
                rankings = [ids_densos, self._bm25_top_ids(pregunta, cap)]
            elif modo == "concat":
                rankings = [
                    ids_densos,
                    self._bm25_top_ids(f"{pregunta} {hyde_doc}", cap),
                ]
            elif modo == "tres_bandas":
                rankings = [
                    ids_densos,
                    ids_bm25_hyde,
                    self._bm25_top_ids(pregunta, cap),
                ]
            else:
                raise ValueError(f"modo desconocido: {modo!r}")
            return self._rrf(*rankings), sim

        fusionados, mejor_similitud = _fusion(capitulo_filtro)
        fallback = False
        if capitulo_filtro is not None and mejor_similitud < UMBRAL_FALLBACK_CAPITULO:
            fallback = True
            fusionados, mejor_similitud = _fusion(None)
        return fusionados, mejor_similitud, fallback

    # ------------------------------------------------------------------
    # Paso 7: Small2Big (jerarquía de 3 niveles: inciso -> grupo -> artículo)
    # ------------------------------------------------------------------
    @staticmethod
    def _parsear_pos(chunk_id: str) -> int | None:
        """Extrae `pos` de un chunk_id 'art_id::tipo::pos::sufijo'
        (ver articulo_a_chunks en chunking.py). None si el formato no
        coincide -- no debería pasar con ids que salen de nuestra propia
        ingestión, pero preferimos degradar sin excepción antes que
        romper la expansión entera por un id inesperado."""
        partes = chunk_id.split("::")
        if len(partes) != 4:
            return None
        try:
            return int(partes[2])
        except ValueError:
            return None

    @staticmethod
    def _grupo_para_pos(padre: dict, tipo: str, pos: int) -> dict | None:
        """Busca en padre[f'{tipo}_grupos'] el grupo cuyo rango
        [pos_inicio, pos_fin] contiene `pos`. Este es el paso que resuelve
        la colisión de markers repetidos: no buscamos por valor de marker,
        buscamos por posición, que es única dentro del tipo."""
        for grupo in padre.get(f"{tipo}_grupos", []):
            if grupo["pos_inicio"] <= pos <= grupo["pos_fin"]:
                return grupo
        return None

    def _expandir_a_articulos(self, chunk_ids: list[str]) -> list[dict]:
        # --- Qué artículos entran: idéntico a antes, corta en TOP_ARTICULOS.
        # Esto no cambia -- es la selección ya validada, no la tocamos.
        vistos: list[str] = []
        for cid in chunk_ids:
            meta = self._metadata_por_id.get(cid)
            if not meta:
                continue
            art_id = meta["art_id"]
            if art_id not in vistos:
                vistos.append(art_id)
            if len(vistos) >= TOP_ARTICULOS:
                break

        # --- Qué grupos ganaron dentro de esos artículos: recorremos TODA
        # la lista fusionada (sin el corte de arriba), pero solo sumamos
        # señal para artículos que ya están en `vistos`. Un chunk que cae
        # más abajo en el ranking y pertenece a un artículo no elegido se
        # sigue ignorando igual que antes; si pertenece a uno elegido,
        # ahora sí aporta qué inciso puntual lo hizo ganar.
        vistos_set = set(vistos)
        grupos_ganadores: dict[str, dict[str, list[dict]]] = {
            art_id: {"ley": [], "reglamentacion": []} for art_id in vistos
        }
        rangos_vistos: dict[str, set[tuple[str, int, int]]] = {
            art_id: set() for art_id in vistos
        }
        for cid in chunk_ids:
            meta = self._metadata_por_id.get(cid)
            if not meta:
                continue
            art_id = meta["art_id"]
            if art_id not in vistos_set:
                continue
            tipo = meta["tipo"]
            pos = self._parsear_pos(cid)
            if pos is None:
                continue
            padre = self._padres.get(art_id)
            if not padre:
                continue
            grupo = self._grupo_para_pos(padre, tipo, pos)
            if grupo is None:
                continue
            clave = (tipo, grupo["pos_inicio"], grupo["pos_fin"])
            if clave in rangos_vistos[art_id]:
                continue
            rangos_vistos[art_id].add(clave)
            grupos_ganadores[art_id][tipo].append(grupo)

        articulos = []
        for art_id in vistos:
            padre = self._padres.get(art_id)
            if not padre:
                continue
            articulos.append({
                "art_id": art_id,
                **padre,
                "ley_grupos_ganadores": grupos_ganadores[art_id]["ley"],
                "reglamentacion_grupos_ganadores": grupos_ganadores[art_id]["reglamentacion"],
            })
        return articulos

    # ------------------------------------------------------------------
    # Punto de entrada
    # ------------------------------------------------------------------
    def buscar(self, pregunta: str) -> ResultadoBusqueda:
        # Paso 0
        directo = self._buscar_por_identificador(pregunta)
        if directo is not None:
            diag = Diagnostico(modo="identificador_directo")
            return ResultadoBusqueda(diagnostico=diag, articulos=[directo])

        # Paso 1
        clasificacion = self._hyde_y_clasificar(pregunta)
        capitulos = clasificacion["capitulos_probables"]
        confianza = clasificacion["confianza"]
        hyde_doc = clasificacion["documento_hipotetico"]

        # Los capítulos del LLM (si vino confiado) más los de los artículos
        # citados en la pregunta forman el conjunto de IMPULSO: capítulos que
        # reciben representación extra en la fusión. Ya NO es un filtro que
        # excluye -- _buscar_hibrido_tres_bandas siempre corre además una pasada
        # global, así que estos capítulos empujan pero nunca esconden a los
        # demás (impulso aditivo estilo RAGSUDOCU).
        base = capitulos if (confianza == "alta" and capitulos) else []
        capitulos_citados = self._capitulos_de_articulos_citados(pregunta)
        capitulos_impulso = list(dict.fromkeys(base + capitulos_citados)) or None

        # Pasos 2-4 (tres bandas, siempre global + impulso por capítulo).
        fusionados, mejor_similitud = self._buscar_hibrido_tres_bandas(
            pregunta, hyde_doc, capitulos_impulso
        )

        diag = Diagnostico(
            modo="recuperado",
            capitulo_elegido=capitulos[0] if capitulos else None,
            capitulos_elegidos=capitulos,
            confianza_capitulo=confianza,
            # Ya no hay fallback: la búsqueda siempre incluye la pasada global,
            # nunca se restringe, así que no hay nada de lo que caer de vuelta.
            fallback_a_global=False,
            mejor_similitud=round(mejor_similitud, 4),
            documento_hipotetico=hyde_doc,
        )

        # Paso 6: umbral de "no encontrado"
        if mejor_similitud < UMBRAL_NO_ENCONTRADO:
            diag.modo = "no_encontrado"
            return ResultadoBusqueda(diagnostico=diag, articulos=[])

        # Paso 7
        articulos = self._expandir_a_articulos(fusionados)
        return ResultadoBusqueda(diagnostico=diag, articulos=articulos)