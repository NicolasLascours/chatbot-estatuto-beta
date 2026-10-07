"""
buscador.py — Segunda estrategia del piloto. NO concluye sobre el caso.

Mientras el asistente (generacion.responder) aplica la norma al caso del
docente, el buscador hace lo contrario: muestra qué dicen los artículos que se
recuperaron, organizado por artículo, y deja la aplicación al lector. Es la
estrategia más segura para el objetivo de no afirmar algo que no se puede
respaldar, y la más barata (una sola llamada a Groq, sin ciclo de corrección).

Reusa la MISMA recuperación y el MISMO contexto que el asistente, así las dos
estrategias se comparan sobre idéntica evidencia y no se gasta una búsqueda de
más. La única llamada a Groq usa un prompt que prohíbe explícitamente concluir
sobre el caso.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from generacion import (
    construir_contexto,
    formatear_para_docente,
    _llamar_groq,
    MODELO_GROQ,
    PRESUPUESTO_CONTEXTO_CHARS,
    MAX_TOKENS_RESPUESTA,
)

PROMPT_BUSCADOR = """\
Sos un buscador del Estatuto Docente de la Provincia de Buenos Aires (Ley \
10.579). Tu tarea NO es resolver el caso del docente ni decirle qué le \
corresponde. Tu tarea es mostrarle QUÉ DICEN los artículos que se \
recuperaron sobre el tema de su consulta, para que los lea y los aplique a \
su situación.

Reglas:
1. Respondé SOLO con el contexto que se te da. Nunca uses conocimiento propio.
2. Organizá la respuesta por artículo. Para cada artículo relevante, resumí \
en pocas frases qué regula y qué dice, copiando los plazos, montos y \
condiciones con las palabras del texto. Cerrá cada afirmación normativa con \
su cita en el formato EXACTO [ARTÍCULO N, CAP. M] o \
[ARTÍCULO N, CAP. M — REGLAMENTACIÓN]. La cita va sola entre corchetes, \
nunca la envuelvas en otra cosa: escribí [ARTÍCULO 120, CAP. 20], NUNCA \
[CITA: [ARTÍCULO 120, CAP. 20]] ni variantes.
2.b. No pongas títulos que repitan lo que ya dice la cita. No escribas \
encabezados del tipo "Artículo 120, Cap. 20 — Reglamentación" arriba del \
párrafo: nombrá el artículo dentro de la frase y cerrá con la cita. El dato \
de si es ley o reglamentación va solo adentro de la cita, una vez, no en un \
título aparte.
3. NO concluyas sobre el caso. No escribas "en tu caso", "te corresponde", \
"sí podés" ni "no podés". No apliques la regla a la situación del docente. Si \
el docente dio datos de su caso, ignoralos para decidir y limitate a mostrar \
la norma.
4. Si distintos sujetos o supuestos tienen reglas distintas (titular, \
provisional, suplente; corta o larga duración), mostralos todos por separado, \
sin elegir cuál aplica.
5. Si algo que se pregunta no está en el contexto, decí "Esto no está en el \
fragmento del Estatuto que tengo disponible". Nunca digas que el Estatuto "no \
prevé" o "no establece" algo.
6. Terminá con una línea que invite a leer el artículo completo para la \
situación concreta, o a consultar con el gremio.

El resultado es un mapa de qué dice la norma, no una respuesta sobre el caso."""


import re

# Red de seguridad por si el modelo igual envuelve la cita: desenvuelve
# [CITA: [ARTÍCULO N, CAP. M]] y [CITA: ARTÍCULO N, CAP. M] a [ARTÍCULO N, CAP. M].
# Después formatear_para_docente le saca el sufijo "— REGLAMENTACIÓN".
_ENVOLTORIO_CITA = re.compile(
    r"\[\s*CITA:\s*\[?\s*(ART[IÍ]CULO[^\]]*?)\s*\]?\s*\]", re.IGNORECASE)


def _limpiar_citas(texto: str) -> str:
    return _ENVOLTORIO_CITA.sub(r"[\1]", texto)


@dataclass
class RespuestaBuscador:
    texto: str
    texto_docente: str = ""
    articulos_excluidos_por_presupuesto: list[int] = field(default_factory=list)


def responder_buscador(
    pregunta: str,
    resultado_busqueda,
    cliente_groq,
    modelo: str = MODELO_GROQ,
    presupuesto_chars: int = PRESUPUESTO_CONTEXTO_CHARS,
    max_tokens: int = MAX_TOKENS_RESPUESTA,
) -> RespuestaBuscador:
    """Misma firma que generacion.responder, para que la app las trate igual.
    No tiene ciclo de verificación/corrección porque no concluye sobre el caso:
    el riesgo que ese ciclo controla (afirmar algo falso sobre el caso) no
    aplica acá. El verificador de citas igual puede correrse después, offline,
    sobre el texto guardado."""
    diag = resultado_busqueda.diagnostico
    if getattr(diag, "modo", None) == "no_encontrado":
        msg = ("No encontré en el Estatuto un artículo que trate claramente "
               "tu consulta. Probá reformularla o nombrando el artículo si lo "
               "conocés.")
        return RespuestaBuscador(texto=msg, texto_docente=msg)

    articulos = resultado_busqueda.articulos
    contexto = construir_contexto(articulos, presupuesto_chars)
    user = f"Contexto:\n{contexto.texto}\n\nConsulta del docente: {pregunta}"
    texto = _llamar_groq(cliente_groq, modelo, PROMPT_BUSCADOR, user, max_tokens)
    texto = _limpiar_citas(texto)

    return RespuestaBuscador(
        texto=texto,
        texto_docente=formatear_para_docente(texto, articulos),
        articulos_excluidos_por_presupuesto=contexto.articulos_excluidos,
    )