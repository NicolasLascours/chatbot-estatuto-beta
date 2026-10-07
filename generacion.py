"""
generacion.py — Genera la respuesta final para el docente, con citas
verificadas contra los artículos recuperados.

Deliberadamente NO importa retrieval.py (ver TYPE_CHECKING más abajo) —
`responder()` solo necesita que el objeto que le pasás tenga
`.diagnostico.modo` y `.articulos` (duck typing).

CAMBIO DE FORMATO ESTA SESIÓN: las citas ya no usan la palabra "LEY" — un
artículo del Estatuto no es "ley" vs. "otra cosa", es el artículo mismo vs.
su reglamentación (que puede no existir todavía). Formato actual:

    [ARTÍCULO 114, CAP. 20]                    — texto del artículo
    [ARTÍCULO 114, CAP. 20 — REGLAMENTACIÓN]   — su reglamentación
    [ARTÍCULO 114 — ESTADO: SIN REGLAMENTAR]   — obligatorio si el estado
                                                  no es "reglamentado"

GARANTÍA DEL TAG DE ESTADO: no confiamos solo en que el LLM lo escriba.
`verificar_respuesta()` ya detecta si falta (fallo_estado); si después de
la corrección sigue faltando, `_asegurar_notas_estado()` lo agrega por
código, de forma determinista, usando el dato que YA tenemos
(`estado_reglamentacion`) — no hace falta que el LLM nos diga algo que
nosotros ya sabemos con certeza. Esto también ahorra una llamada a Groq:
si el ÚNICO problema de la primera respuesta es un tag de estado faltante
(sin fallos de cita ni de cobertura), no regeneramos — lo completamos
nosotros directo.

Pipeline de responder():

    no_encontrado → mensaje fijo, sin Groq
    top excluido por presupuesto → mensaje honesto, sin Groq
    si no:
        generar_respuesta()
            → verificar_respuesta()
                → sin fallos de cita/cobertura (a lo sumo falta el tag de
                  estado) → se garantiza por código, listo
                → fallos de cita/cobertura → generar_correccion() → verificar
                  otra vez → mismo criterio
                → si persisten fallos de cita/cobertura → filtrar_no_verificado()
                  + garantía de estado igual, sobre lo que sí quedó
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from verificacion_citas import (
    verificar_respuesta,
    filtrar_no_verificado,
    reparar_citas_huerfanas,
    PATRON_ESTADO,
)

if TYPE_CHECKING:
    from retrieval import ResultadoBusqueda
    from verificacion_citas import FalloEstadoNoDeclarado


MODELO_GROQ = "openai/gpt-oss-120b"

TEMPERATURA_GENERACION = 0.2

# gpt-oss-120b es un modelo de RAZONAMIENTO: antes de escribir la respuesta
# visible, gasta tokens en una cadena de pensamiento interna, y Groq
# factura esos tokens contra el mismo max_tokens de la respuesta.
# CONFIRMADO en sesión (antes era una sospecha sin confirmar, ver el
# comentario en _llamar_groq): con un contexto multi-capítulo real (tres
# capítulos, cinco artículos), el razonamiento interno se comió los 900
# tokens completos antes de escribir una sola palabra visible --
# finish_reason='length', contenido vacío.
#
# Subir max_tokens NO alcanza para arreglar esto de forma confiable: al
# esfuerzo de razonamiento por default de Groq ("medium"), el razonamiento
# tiende a expandirse para llenar el presupuesto que le des, así que un
# max_tokens más alto puede seguir devolviendo vacío igual. La solución
# confirmada (reportada por otros equipos que pisaron este mismo bug con
# este mismo modelo en Groq) es bajar el esfuerzo de razonamiento en sí.
# reasoning_effort es un parámetro no estándar que Groq acepta SOLO para
# la familia gpt-oss -- da 400 en cualquier otro modelo, incluidos otros
# modelos de Groq que no razonan, así que _llamar_groq lo manda
# condicionado al nombre del modelo, no incondicional.
REASONING_EFFORT_GENERACION = "low"

PRESUPUESTO_CONTEXTO_CHARS = 14_000  # EXPERIMENTAL, no definitivo — configurable
MAX_TOKENS_RESPUESTA = 900

MENSAJE_NO_ENCONTRADO = (
    "No encontré en los artículos recuperados información suficiente para "
    "responder esta consulta con respaldo normativo. Podés reformular la "
    "pregunta o consultar directamente la normativa correspondiente."
)

_DESCRIPCION_ESTADO = {
    "sin_reglamentar": (
        "sin reglamentar. El derecho existe en la ley pero no tiene "
        "reglamentación operativa vigente."
    ),
    "parcial": (
        "parcialmente reglamentado. Algunos incisos de este artículo no "
        "tienen reglamentación."
    ),
    "sin_dato": "estado de reglamentación no determinado.",
}


def _mensaje_articulo_mas_relevante_excluido(numero: int) -> str:
    return (
        f"El artículo que mejor responde tu consulta (Artículo {numero} del "
        "Estatuto Docente) es demasiado extenso para procesarlo con el "
        "modelo disponible en este momento. Te recomendamos consultarlo "
        "directamente, o reformular la pregunta para acotar la consulta a "
        "una parte más específica de ese artículo."
    )


# ----------------------------------------------------------------------
# Prompt de sistema
# ----------------------------------------------------------------------

PROMPT_SISTEMA = """\
Sos un asistente que ayuda a docentes de la Provincia de Buenos Aires a \
encontrar qué dice el Estatuto Docente (Ley 10.579) sobre su duda. Tu \
trabajo es ubicar en el contexto el artículo y el inciso que corresponden \
y contarlos con sus datos exactos. No interpretás ni razonás más allá del \
texto. Respondés ÚNICAMENTE con el contexto que se te da, nunca con \
conocimiento propio sobre el Estatuto ni sobre legislación laboral.

CÓMO ARMAR LA RESPUESTA

Empezá por lo que el docente preguntó (si puede, cuántos días, en qué \
condiciones) y decí siempre en qué artículo e inciso está. Los plazos, \
montos y condiciones van COPIADOS del texto, con sus mismas palabras y \
números. Ante la duda entre copiar y parafrasear, copiá.

Si el texto responde distinto según la situación (titular o suplente, \
tipo de estudio, tramo de antigüedad), dá cada caso por separado en vez \
de elegir uno. Si la respuesta depende de un dato que el docente no \
dio, decilo y presentá las alternativas que el texto trae.

El inciso se nombra siempre, en tu propia oración y afuera de los \
corchetes, con la etiqueta tal como aparece en el texto (por ejemplo \
"inciso c" o "inciso ll), punto 1.1"). Nunca lo pongas adentro de la \
cita.

REGLAS DE CITA (obligatorias)

Cada afirmación normativa (plazo, monto, condición, derecho, obligación) \
termina con una cita en uno de estos dos formatos EXACTOS.

  [ARTÍCULO N, CAP. M]                   para el texto del artículo
  [ARTÍCULO N, CAP. M — REGLAMENTACIÓN]  para su reglamentación

Usá el número de artículo y de capítulo tal cual aparecen en el contexto. \
El Estatuto es una sola ley, así que nunca escribas "LEY" en la cita. \
Nada más va adentro de los corchetes, ni incisos ni decretos, y la \
oración con una cita mal formada se descarta entera.

  CORRECTO
  "Tenés diez (10) días corridos de licencia por matrimonio, según el \
inciso b del artículo 114 [ARTÍCULO 114, CAP. 20]."

  INCORRECTO
  "...[ARTÍCULO 114 — LEY]"
  "...según el Art. 114."
  "...[114-L]"
  "...[ARTÍCULO 75, CAP. 14 — REGLAMENTACIÓN, INCISO 5]"

No cites un artículo que no esté literalmente en el contexto.

ESTADO DE REGLAMENTACIÓN (obligatorio)

Si citás un artículo y el contexto trae para él un bloque \
"[ESTADO DE REGLAMENTACIÓN DEL ARTÍCULO N]" que indica que NO está \
reglamentado o que lo está solo en parte, incluí en tu respuesta el tag \
[ARTÍCULO N — ESTADO: <descripción breve>] explicando esa situación, \
aunque el docente no lo pregunte. Tiene que saber si el derecho ya es \
operativo.

CUANDO EL CONTEXTO NO ALCANZA

Si algo de lo que preguntan no está en el contexto, decí "Esto no está \
en el fragmento del Estatuto que tengo disponible", adaptado a lo que \
falte. NUNCA digas que el Estatuto "no fija", "no establece", "no \
prevé", "no contempla" o "no otorga" algo, porque vos no sabés si la \
ley lo prevé, solo sabés que no lo tenés delante. Si la pregunta tiene \
varias partes, respondé las respaldadas y nombrá cuáles no pudiste \
responder. Si el contexto describe un mecanismo pero no un valor \
concreto, no inventes el valor.

FIDELIDAD AL TEXTO

a) No cambies el verbo ni el derecho de la cláusula. Si el texto dice \
"usufructuar", no escribas "reincorporarse".
b) Una lista cerrada de opciones o combinaciones se chequea entera. Si \
el caso no encaja exacto en ninguna, esa lista no lo habilita.
c) Antes de aplicar una regla, verificá que el caso cumple su condición \
de entrada (a quién y en qué momento se aplica). Si queda afuera, decí \
que la regla no lo alcanza.
d) No afirmes que un artículo se conecta con otro salvo que el texto \
lo diga con esas palabras.
"""


# ----------------------------------------------------------------------
# Construcción del contexto (con presupuesto)
# ----------------------------------------------------------------------

@dataclass
class ContextoConstruido:
    texto: str
    # Artículos donde ley Y reglamentación (las que tengan texto de origen)
    # entraron completas — nivel 3.
    articulos_incluidos: list[int] = field(default_factory=list)
    # Artículos donde al menos ley o reglamentación tuvo que recurrir a la
    # versión por grupos de incisos porque el texto completo no entraba —
    # nivel 2. Un artículo puede tener, por ejemplo, la ley completa y la
    # reglamentación parcial al mismo tiempo; en ese caso cae acá, no en
    # articulos_incluidos.
    articulos_expandidos_parcialmente: list[int] = field(default_factory=list)
    # Artículos que no lograron meter ni un carácter — ni completo ni
    # parcial. Es el único caso real de "no entra en el presupuesto".
    articulos_excluidos: list[int] = field(default_factory=list)
    # Nivel real con que entró CADA sección al contexto enviado a Groq:
    # {numero: {"ley": nivel, "reglamentacion": nivel}} con nivel en
    # {"completo", "parcial", "nada"}. Es lo que verificar_respuesta()
    # necesita para chequear las citas contra lo que EFECTIVAMENTE se envió,
    # no contra el padre recuperado (que es un superconjunto: incluye
    # secciones que quedaron afuera por presupuesto y reglamentaciones "Sin
    # reglamentar." que se descartan). Se registra para todos los artículos
    # de la lista, incluidos los excluidos del todo (ahí ambas secciones son
    # "nada"). Ver la fuga que cierra en el docstring de verificar_respuesta.
    niveles_por_seccion: dict[int, dict[str, str]] = field(default_factory=dict)


def _etiqueta(tipo: str, numero: int, capitulo) -> str:
    if tipo == "reglamentacion":
        return f"[ARTÍCULO {numero}, CAP. {capitulo} — REGLAMENTACIÓN]"
    return f"[ARTÍCULO {numero}, CAP. {capitulo}]"


_AVISO_PARCIAL = (
    "(Extracto parcial del artículo, no el texto completo — puede haber "
    "incisos adicionales no incluidos por espacio. Si la consulta pide "
    "algo que no aparece acá, decí que no está en el extracto disponible, "
    "no que el Estatuto no lo prevé.)"
)


def _es_sin_reglamentar(texto: str) -> bool:
    """Igual que la función homónima de parser_estatuto.py — un grupo de
    reglamentación cuyo texto es solo la anotación administrativa "Sin
    reglamentar.", sin contenido normativo real. Se reimplementa acá,
    chica y autocontenida, para no importar parser_estatuto.py desde la
    etapa de generación — son etapas del pipeline deliberadamente
    separadas (ver ways-of-working del proyecto)."""
    limpio = texto.strip().rstrip(".").strip().lower()
    return limpio == "sin reglamentar"


def _markers_relevantes(art: dict) -> set[str]:
    """Qué incisos de este artículo son relevantes para ESTA pregunta.

    Dos correcciones sobre un primer intento fallido de esta misma
    sesión (ver notas): no alcanza con mirar `ley_grupos_ganadores` solo.
    En la práctica, el chunk de ley de un inciso suele ser una frase
    corta ("c) Por matrimonio.") mientras que el chunk de reglamentación
    del mismo inciso trae todo el contenido rico (plazos, condiciones) —
    en la búsqueda híbrida gana casi siempre el de reglamentación, el de
    ley muchas veces ni entra en el ranking fusionado. Por eso acá se
    combinan los markers ganadores de los dos lados, que un inciso haya
    ganado por cualquiera de las dos rutas ya es señal suficiente de que
    es relevante para la pregunta.

    Además, `_buscar_por_identificador` (retrieval.py) devuelve el
    artículo directo desde `indice_padres.json` cuando la pregunta
    menciona el número, sin pasar por `_expandir_a_articulos` — nunca
    tiene `ley_grupos_ganadores` ni `reglamentacion_grupos_ganadores`.
    Ahí no hay "inciso puntual" que la búsqueda haya identificado, todo
    el artículo está en foco porque el docente lo pidió por número, así
    que tratamos esa ausencia total de las dos claves como señal de
    "todo el artículo es relevante", no como ausencia de señal — se
    distingue de una búsqueda que sí corrió pero no encontró incisos con
    letra relevantes (eso devuelve las claves presentes pero vacías, y
    ahí sí no hay nada que decir)."""
    tiene_ley = "ley_grupos_ganadores" in art
    tiene_reg = "reglamentacion_grupos_ganadores" in art
    if not tiene_ley and not tiene_reg:
        return _markers_letra_simple(art.get("ley_grupos", []))
    return _markers_letra_simple(art.get("ley_grupos_ganadores", [])) | \
        _markers_letra_simple(art.get("reglamentacion_grupos_ganadores", []))


def _calcular_estado_efectivo(art: dict) -> str | None:
    """Estado de reglamentación de lo que es RELEVANTE para esta pregunta
    puntual, no del artículo completo.

    Por qué hace falta: `estado_reglamentacion` (calculado por
    parser_estatuto.py) es un dato a nivel de artículo entero. Un
    artículo "parcialmente reglamentado" puede tener, para los incisos
    relevantes para esta pregunta puntual (ver _markers_relevantes),
    cobertura completa (o al revés).

    Comparamos esos markers, letra por letra y por conjunto (no por
    posición, que es donde parser_estatuto.py tiene un bug conocido y
    acotado — ver notas de sesión), contra los grupos de reglamentación
    disponibles para el artículo completo (no solo los que ganaron por
    búsqueda: si la reglamentación de un inciso existe pero no fue un
    grupo ganador, igual cuenta como reglamentada, lo que falta ahí es
    texto en el contexto, no vigencia normativa — ese aviso ya lo cubre
    _AVISO_PARCIAL por separado).

    Devuelve None (use el campo original) en tres casos: no hay ningún
    inciso con letra relevante para esta pregunta, el artículo ya viene
    en 'sin_dato' (incertidumbre deliberada de parser_estatuto.py, no la
    pisamos con una respuesta más confiada pero no confiable, típicamente
    estructura anidada con romanos), o la reglamentación no está
    desglosada por letra (bloque único, donde el campo original ya es
    confiable)."""
    markers = _markers_relevantes(art)
    if not markers:
        return None

    if art.get("estado_reglamentacion") == "sin_dato":
        return None

    reg_por_marker = {
        g["marker"].lower(): g["texto"]
        for g in art.get("reglamentacion_grupos", [])
        if g.get("marker") and len(g["marker"]) == 1 and g["marker"].isalpha()
    }
    if not reg_por_marker:
        return None

    estados = []
    for marker in markers:
        texto_reg = reg_por_marker.get(marker)
        if texto_reg is None or _es_sin_reglamentar(texto_reg):
            estados.append("sin_reglamentar")
        else:
            estados.append("reglamentado")

    if all(e == "reglamentado" for e in estados):
        return "reglamentado"
    if all(e == "sin_reglamentar" for e in estados):
        return "sin_reglamentar"
    return "parcial"


def _bloque_estado(art: dict) -> str:
    numero = art["numero"]
    estado = art.get("estado_reglamentacion_efectivo", art.get("estado_reglamentacion"))
    if not estado or estado == "reglamentado":
        return ""
    descripcion = _DESCRIPCION_ESTADO.get(estado, f"estado: {estado}.")
    return (
        f"[ESTADO DE REGLAMENTACIÓN DEL ARTÍCULO {numero}]\n"
        f"Este artículo está {descripcion} Si citás este artículo, "
        f"incluí también el tag [ARTÍCULO {numero} — ESTADO: ...] en tu "
        "respuesta (ver regla 2). Este bloque es un dato estructural "
        "calculado por el sistema, no texto del Estatuto."
    )


def _bloque_parcial(
    art: dict, tipo: str, presupuesto_restante: int, numero: int, capitulo
) -> tuple[str, int]:
    """Nivel 2: arma el bloque de `tipo` ('ley' o 'reglamentacion') a partir
    de sus grupos de incisos, cuando el texto completo no entró.

    El grupo de encabezado (marker == "") va primero siempre que exista,
    aunque no haya ganado por búsqueda — suele traer las condiciones
    generales que le dan sentido a los incisos sueltos. Después se suman
    los grupos_ganadores en el orden en que ya vienen, que es el orden de
    primera aparición en el ranking RRF fusionado (heredado de
    retrieval.py, no se reordena acá). Si ni el encabezado entra en el
    presupuesto, no hay bloque parcial posible y se devuelve vacío."""
    grupos_todos = art.get(f"{tipo}_grupos", [])
    ganadores = art.get(f"{tipo}_grupos_ganadores", [])

    encabezado = next((g for g in grupos_todos if g.get("marker", "") == ""), None)
    rango_encabezado = (
        (encabezado["pos_inicio"], encabezado["pos_fin"]) if encabezado else None
    )

    candidatos = [encabezado] if encabezado is not None else []
    for g in ganadores:
        if (g["pos_inicio"], g["pos_fin"]) == rango_encabezado:
            continue  # ya está como encabezado, no lo dupliquemos
        candidatos.append(g)

    if not candidatos:
        return "", 0

    etiqueta = _etiqueta(tipo, numero, capitulo)
    costo_fijo = len(etiqueta) + 1 + len(_AVISO_PARCIAL) + 1  # + saltos de línea

    partes_incluidas: list[str] = []
    chars_grupos = 0
    for g in candidatos:
        marker = g.get("marker", "")
        cuerpo = g["texto"].strip()
        bloque_grupo = f"[INCISO {marker}] {cuerpo}" if marker else cuerpo
        costo = len(bloque_grupo) + (1 if partes_incluidas else 0)
        if costo_fijo + chars_grupos + costo > presupuesto_restante:
            if not partes_incluidas:
                # Ni el encabezado entró — no hay versión parcial posible.
                return "", 0
            break
        partes_incluidas.append(bloque_grupo)
        chars_grupos += costo

    cuerpo_final = "\n".join(partes_incluidas)
    texto = f"{etiqueta}\n{_AVISO_PARCIAL}\n{cuerpo_final}"
    return texto, costo_fijo + chars_grupos


_TIPOS_LETRA = {
    "minuscula_par", "mayuscula_par", "mayuscula_punto", "mayuscula_dos_puntos",
}


def _markers_letra_simple(grupos: list[dict]) -> set[str]:
    """Markers que son genuinamente una LETRA de inciso (a, b, c... ñ),
    no un romano ni un número — excluye el encabezado y cualquier
    sub-bloque romano o numérico, incluso cuando el marker es un solo
    carácter y por texto solo sería indistinguible de una letra (el caso
    de "I" romano vs. letra "i", ver art. 90 y 133 en notas de sesión).

    Usa el campo `tipo` que chunking.py empezó a propagar desde
    parser_estatuto.py en _agrupar_por_racha() — es el dato confiable,
    calculado en el parseo original, no una inferencia por forma del
    texto. Si un grupo todavía no tiene `tipo` (viene de un
    indice_padres.json generado antes de ese cambio, sin regenerar
    todavía), cae al heurístico anterior (un solo carácter alfabético)
    para esos grupos puntuales, en vez de fallar — más laxo, con el
    mismo riesgo de confusión romano/letra que había antes de este
    arreglo, pero no rompe nada mientras se termina de regenerar."""
    resultado = set()
    for g in grupos:
        marker = g.get("marker")
        if not marker:
            continue
        tipo = g.get("tipo")
        if tipo is not None:
            if tipo in _TIPOS_LETRA and len(marker) == 1:
                resultado.add(marker.lower())
        elif len(marker) == 1 and marker.isalpha():
            resultado.add(marker.lower())
    return resultado


def _intentar_tipo(
    art: dict, tipo: str, presupuesto_restante: int
) -> tuple[str, str, int]:
    """Intenta incluir `tipo` ('ley' o 'reglamentacion') de un artículo:
    primero completo (nivel 3) y, si no entra, en modo parcial por grupos
    (nivel 2). Devuelve (texto_bloque, nivel, chars_usados), con nivel en
    {'completo', 'parcial', 'nada'}. 'nada' cubre tanto la ausencia de
    texto de origen como el caso en que ni el encabezado del grupo entró.

    CASO ESPECIAL (agregado en sesión, ver notas): cuando `tipo` es
    'reglamentacion' y el texto fuente es literalmente "Sin
    reglamentar." sin nada más, tratamos eso como 'nada' a propósito, no
    como contenido a incluir. Ese texto no aporta ningún dato que el
    bloque de ESTADO (_bloque_estado, más abajo en construir_contexto)
    no diga ya en prosa -- incluirlo es presupuesto de caracteres
    gastado en información duplicada. Encontrado en sesión: con contexto
    multi-capítulo ajustado, dos artículos cuyo único aporte posible era
    ese texto vacío ("Sin reglamentar.") se comieron presupuesto real
    mientras un tercer artículo con contenido sustancial (art. 38)
    quedaba afuera por falta de espacio. Reusa _es_sin_reglamentar(), la
    misma función que ya usa _calcular_estado_efectivo para este chequeo,
    en vez de reimplementarlo."""
    numero = art["numero"]
    capitulo = art.get("capitulo_num", "?")
    texto_fuente = (art.get(tipo) or "").strip()
    if not texto_fuente:
        return "", "nada", 0

    if tipo == "reglamentacion" and _es_sin_reglamentar(texto_fuente):
        return "", "nada", 0

    etiqueta = _etiqueta(tipo, numero, capitulo)
    bloque_completo = f"{etiqueta}\n{texto_fuente}"
    if len(bloque_completo) <= presupuesto_restante:
        return bloque_completo, "completo", len(bloque_completo)

    texto_parcial, chars = _bloque_parcial(
        art, tipo, presupuesto_restante, numero, capitulo
    )
    if texto_parcial:
        return texto_parcial, "parcial", chars
    return "", "nada", 0


def construir_contexto(
    articulos: list[dict],
    presupuesto_chars: int = PRESUPUESTO_CONTEXTO_CHARS,
) -> ContextoConstruido:
    separador = "\n\n---\n\n"
    bloques: list[str] = []
    incluidos: list[int] = []
    parciales: list[int] = []
    excluidos: list[int] = []
    # Nivel real con que entró cada sección, por artículo (ver el campo
    # homónimo en ContextoConstruido). Se registra para TODOS los artículos,
    # también los que quedan afuera del todo, porque el verificador lo usa
    # para distinguir "esta sección no se envió" de "el padre la tenía vacía".
    niveles_por_seccion: dict[int, dict[str, str]] = {}
    chars_usados = 0

    for art in articulos:
        numero = art["numero"]
        # Arranca todo en "nada" y queda así en cualquier camino de exclusión
        # temprana. Es el mismo objeto referenciado desde el mapa, así que las
        # actualizaciones de abajo se reflejan sin re-asignar.
        niveles_art = {"ley": "nada", "reglamentacion": "nada"}
        niveles_por_seccion[numero] = niveles_art

        costo_separador = len(separador) if bloques else 0
        restante = presupuesto_chars - chars_usados - costo_separador
        if restante <= 0:
            excluidos.append(numero)
            continue

        partes_articulo: list[str] = []
        niveles_usados: set[str] = set()
        gastado_articulo = 0

        for tipo in ("ley", "reglamentacion"):
            texto, nivel, costo = _intentar_tipo(
                art, tipo, restante - gastado_articulo
            )
            niveles_art[tipo] = nivel
            if nivel == "nada":
                continue
            gastado_articulo += costo + (2 if partes_articulo else 0)
            partes_articulo.append(texto)
            niveles_usados.add(nivel)

        # Estado efectivo para ESTA pregunta, no para el artículo completo
        # — ver _calcular_estado_efectivo. Se guarda en el propio dict del
        # artículo (mismo objeto que después recibe verificar_respuesta())
        # para que la garantía de estado en verificacion_citas.py use el
        # mismo dato sin tener que recalcularlo ni que se lo pasemos aparte.
        estado_efectivo = _calcular_estado_efectivo(art)
        if estado_efectivo is not None:
            art["estado_reglamentacion_efectivo"] = estado_efectivo

        bloque_estado = _bloque_estado(art)
        if bloque_estado and partes_articulo:
            costo_estado = len(bloque_estado) + 2
            if gastado_articulo + costo_estado <= restante:
                partes_articulo.append(bloque_estado)
                gastado_articulo += costo_estado

        if not partes_articulo:
            excluidos.append(numero)
            continue

        bloques.append("\n\n".join(partes_articulo))
        chars_usados += costo_separador + gastado_articulo

        if "parcial" in niveles_usados:
            parciales.append(numero)
        else:
            incluidos.append(numero)

    return ContextoConstruido(
        texto=separador.join(bloques),
        articulos_incluidos=incluidos,
        articulos_expandidos_parcialmente=parciales,
        articulos_excluidos=excluidos,
        niveles_por_seccion=niveles_por_seccion,
    )


# ----------------------------------------------------------------------
# Garantía determinista del tag de estado
# ----------------------------------------------------------------------

def _asegurar_notas_estado(texto: str, fallos_estado: list["FalloEstadoNoDeclarado"]) -> str:
    """Si quedó algún artículo citado cuyo estado de reglamentación no es
    "reglamentado" y el LLM no lo declaró, lo agregamos NOSOTROS, por
    código — no dependemos de que el LLM lo haya hecho bien. `fallos_estado`
    ya viene calculado por verificar_respuesta(), así que no recalculamos
    nada acá."""
    if not fallos_estado:
        return texto
    notas = []
    for f in fallos_estado:
        descripcion = _DESCRIPCION_ESTADO.get(f.estado, f"estado: {f.estado}.")
        notas.append(f"⚠️ Artículo {f.numero}: está {descripcion}")
    return texto.rstrip() + "\n\n" + "\n".join(notas)


# ----------------------------------------------------------------------
# Llamadas a Groq
# ----------------------------------------------------------------------

def _llamar_groq(
    cliente_groq, modelo: str, system: str, user: str, max_tokens: int
) -> str:
    kwargs = dict(
        model=modelo,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=TEMPERATURA_GENERACION,
        max_tokens=max_tokens,
    )
    # Ver REASONING_EFFORT_GENERACION más arriba para el porqué. Condicionado
    # al nombre del modelo porque este parámetro no existe fuera de la
    # familia gpt-oss (400 en cualquier otro modelo de Groq).
    if "gpt-oss" in modelo:
        kwargs["reasoning_effort"] = REASONING_EFFORT_GENERACION

    resp = cliente_groq.chat.completions.create(**kwargs)
    contenido = (resp.choices[0].message.content or "").strip()
    finish_reason = getattr(resp.choices[0], "finish_reason", None)
    # Diagnóstico para el bug de respuesta vacía (casos reales N2-4, N2-8, y
    # confirmado en sesión con una pregunta multi-capítulo real) -- CAUSA
    # CONFIRMADA: el modelo de razonamiento (openai/gpt-oss-120b) gasta el
    # presupuesto de max_tokens en su cadena de pensamiento interna antes
    # de escribir el contenido final. reasoning_effort="low" (ver arriba)
    # es el arreglo; este log queda igual, por si reaparece con otro
    # esfuerzo o con un contexto todavía más grande de lo esperado.
    if not contenido:
        print(f"    [ATENCIÓN: Groq devolvió contenido vacío. "
              f"finish_reason={finish_reason!r}, max_tokens={max_tokens}, "
              f"reasoning_effort={kwargs.get('reasoning_effort')!r}]")
    return contenido


def generar_respuesta(
    pregunta: str,
    contexto: ContextoConstruido,
    cliente_groq,
    modelo: str = MODELO_GROQ,
    max_tokens: int = MAX_TOKENS_RESPUESTA,
) -> str:
    user = f"Contexto:\n{contexto.texto}\n\nPregunta del docente: {pregunta}"
    return _llamar_groq(cliente_groq, modelo, PROMPT_SISTEMA, user, max_tokens)


def generar_correccion(
    pregunta: str,
    contexto: ContextoConstruido,
    respuesta_previa: str,
    mensajes_fallos: list[str],
    cliente_groq,
    modelo: str = MODELO_GROQ,
    max_tokens: int = MAX_TOKENS_RESPUESTA,
) -> str:
    fallos_fmt = "\n".join(f"- {m}" for m in mensajes_fallos)
    user = (
        f"Contexto:\n{contexto.texto}\n\n"
        f"Pregunta del docente: {pregunta}\n\n"
        f'Tu respuesta anterior fue:\n"""\n{respuesta_previa}\n"""\n\n'
        f"Esa respuesta tuvo estos problemas:\n{fallos_fmt}\n\n"
        "Reescribí la respuesta completa corrigiendo estos problemas "
        "específicos, sin repetirlos. Recordá el formato EXACTO de cita: "
        "[ARTÍCULO N, CAP. M] o [ARTÍCULO N, CAP. M — REGLAMENTACIÓN] — "
        "nunca la palabra \"LEY\", nunca \"Art.\" suelto. Si no podés "
        "respaldar una afirmación con el contexto, NUNCA digas que el "
        "Estatuto \"no fija\", \"no establece\", \"no prevé\" o \"no "
        "contempla\" ese derecho, plazo o monto — vos no sabés si la ley "
        "lo prevé o no, solo sabés que no está en el fragmento que "
        "tenés. Usá siempre esta forma exacta, adaptada a lo que "
        "falte: \"Esto no está en el fragmento del Estatuto que tengo "
        "disponible.\" Nunca lo omitas en silencio ni inventes una cita."
    )
    return _llamar_groq(cliente_groq, modelo, PROMPT_SISTEMA, user, max_tokens)


# ----------------------------------------------------------------------
# Orquestación
# ----------------------------------------------------------------------

@dataclass
class RespuestaFinal:
    texto: str
    verificada: bool
    fue_corregida: bool
    fallos_persistentes: list[str] = field(default_factory=list)
    articulos_excluidos_por_presupuesto: list[int] = field(default_factory=list)
    # Artículos donde ley y/o reglamentación tuvieron que recurrir a la
    # versión por grupos de incisos (nivel 2) porque el texto completo no
    # entraba en el presupuesto. Dato de auditoría, igual que
    # articulos_excluidos_por_presupuesto — responder_cli.py puede
    # mostrarlo para que quede claro cuándo la respuesta se armó sobre un
    # recorte del artículo y no sobre el texto completo.
    articulos_expandidos_parcialmente: list[int] = field(default_factory=list)
    # Números de artículo para los que el tag de ESTADO faltaba y lo agregó
    # el código (no el LLM). Dato de auditoría: si esto no está vacío
    # seguido, es señal de que el prompt necesita más ajuste; mientras
    # tanto, la garantía de "siempre se aclara" se cumple igual.
    estados_agregados_automaticamente: list[int] = field(default_factory=list)
    # Oraciones (texto original, antes de la reparación) donde faltaba una
    # cita y se le copió la de la oración siguiente por código — ver
    # reparar_citas_huerfanas() en verificacion_citas.py. Dato de
    # auditoría, igual que estados_agregados_automaticamente: si esto no
    # está vacío seguido, es señal de que conviene ajustar el prompt para
    # que el modelo cite cada viñeta de una lista, no solo la última.
    citas_reparadas_automaticamente: list[str] = field(default_factory=list)
    # Versión del texto pensada para mostrarle al docente: sin el sufijo
    # técnico "— REGLAMENTACIÓN" en las citas (ley y reglamentación se ven
    # igual, la distinción sigue existiendo por dentro para verificar, no
    # hace falta que el docente la vea) y con el tag de ESTADO siempre
    # convertido a una frase en criollo, aunque el LLM haya escrito el
    # tag con su propia redacción. Ver formatear_para_docente(). `texto`
    # queda intacto arriba, es el que se viene mostrando hasta ahora.
    texto_docente: str = ""


def articulo_mas_relevante_excluido(
    resultado_busqueda: "ResultadoBusqueda", respuesta: RespuestaFinal
) -> bool:
    articulos = resultado_busqueda.articulos
    return bool(articulos) and (
        articulos[0]["numero"] in respuesta.articulos_excluidos_por_presupuesto
    )


_SUFIJO_REGLAMENTACION = re.compile(
    r"(\[ARTÍCULO\s+\d+\s*,\s*CAP\.?\s*\d+)\s*—\s*REGLAMENTACI[ÓO]N(\])",
    re.IGNORECASE,
)


def formatear_para_docente(texto: str, articulos: list[dict]) -> str:
    """Versión del texto de respuesta pensada para que la lea un docente,
    no para depurar el sistema. Corre DESPUÉS de que verificar_respuesta()
    ya aprobó el texto (o filtrar_no_verificado() ya lo recortó) — es
    puramente cosmética, no cambia qué se verificó ni cómo, y no toca
    `texto`, que sigue siendo la versión técnica de siempre.

    Dos transformaciones:

    1. Le saca el sufijo "— REGLAMENTACIÓN" a las citas. Por dentro
       seguimos distinguiendo texto de ley y texto de reglamentación
       (verificacion_citas.py lo necesita para chequear cada capa por
       separado, esa distinción sigue existiendo), pero no le aporta nada
       a un docente leyendo la respuesta — las dos se muestran igual,
       [ARTÍCULO N, CAP. M].

    2. Reemplaza cualquier tag [ARTÍCULO N — ESTADO: ...] — lo haya
       escrito el LLM con su propia redacción o lo haya agregado
       _asegurar_notas_estado() por código — por una frase en criollo,
       armada con _DESCRIPCION_ESTADO a partir del estado REAL que ya
       tenemos calculado para ese artículo, no del texto libre que el LLM
       haya puesto después de "ESTADO:" (el regex de verificación lo
       acepta deliberadamente laxo, sin forzar vocabulario exacto, así
       que no es confiable para mostrárselo tal cual al docente)."""
    texto = _SUFIJO_REGLAMENTACION.sub(r"\1\2", texto)

    por_numero = {a["numero"]: a for a in articulos}

    def _reemplazar_estado(m: re.Match) -> str:
        numero = int(m.group(1))
        art = por_numero.get(numero)
        estado = (
            art.get("estado_reglamentacion_efectivo", art.get("estado_reglamentacion"))
            if art else None
        )
        descripcion = _DESCRIPCION_ESTADO.get(
            estado, "estado de reglamentación no determinado."
        )
        return f"⚠️ Artículo {numero}: está {descripcion}"

    return PATRON_ESTADO.sub(_reemplazar_estado, texto)


def responder(
    pregunta: str,
    resultado_busqueda: "ResultadoBusqueda",
    cliente_groq,
    modelo: str = MODELO_GROQ,
    presupuesto_chars: int = PRESUPUESTO_CONTEXTO_CHARS,
    max_tokens: int = MAX_TOKENS_RESPUESTA,
) -> RespuestaFinal:
    if resultado_busqueda.diagnostico.modo == "no_encontrado":
        return RespuestaFinal(
            texto=MENSAJE_NO_ENCONTRADO,
            verificada=True,
            fue_corregida=False,
            texto_docente=MENSAJE_NO_ENCONTRADO,
        )

    articulos = resultado_busqueda.articulos
    contexto = construir_contexto(articulos, presupuesto_chars)

    print("=== CONTEXTO ENVIADO A GROQ ===")
    print(contexto.texto)
    print("=== FIN CONTEXTO ===")

    if articulos and articulos[0]["numero"] in contexto.articulos_excluidos:
        mensaje = _mensaje_articulo_mas_relevante_excluido(articulos[0]["numero"])
        return RespuestaFinal(
            texto=mensaje,
            verificada=True,
            fue_corregida=False,
            articulos_excluidos_por_presupuesto=contexto.articulos_excluidos,
            texto_docente=mensaje,
        )

    respuesta = generar_respuesta(pregunta, contexto, cliente_groq, modelo, max_tokens)
    respuesta, reparaciones_1 = reparar_citas_huerfanas(respuesta)
    verificacion = verificar_respuesta(respuesta, articulos, contexto.niveles_por_seccion)

    if not verificacion.fallo_vacio and not verificacion.fallos_cita and not verificacion.fallos_cobertura:
        # A lo sumo falta el tag de estado — lo garantizamos por código,
        # sin gastar una regeneración completa solo por esto.
        texto_final = _asegurar_notas_estado(respuesta, verificacion.fallos_estado)
        return RespuestaFinal(
            texto=texto_final,
            verificada=True,
            fue_corregida=False,
            articulos_excluidos_por_presupuesto=contexto.articulos_excluidos,
            articulos_expandidos_parcialmente=contexto.articulos_expandidos_parcialmente,
            estados_agregados_automaticamente=[f.numero for f in verificacion.fallos_estado],
            citas_reparadas_automaticamente=[o for o, _ in reparaciones_1],
            texto_docente=formatear_para_docente(texto_final, articulos),
        )

    respuesta_corregida = generar_correccion(
        pregunta, contexto, respuesta, verificacion.mensajes(), cliente_groq, modelo, max_tokens
    )
    respuesta_corregida, reparaciones_2 = reparar_citas_huerfanas(respuesta_corregida)
    reparaciones_totales = [o for o, _ in reparaciones_1] + [o for o, _ in reparaciones_2]
    verificacion_2 = verificar_respuesta(respuesta_corregida, articulos, contexto.niveles_por_seccion)

    if not verificacion_2.fallo_vacio and not verificacion_2.fallos_cita and not verificacion_2.fallos_cobertura:
        texto_final = _asegurar_notas_estado(respuesta_corregida, verificacion_2.fallos_estado)
        return RespuestaFinal(
            texto=texto_final,
            verificada=True,
            fue_corregida=True,
            articulos_excluidos_por_presupuesto=contexto.articulos_excluidos,
            articulos_expandidos_parcialmente=contexto.articulos_expandidos_parcialmente,
            estados_agregados_automaticamente=[f.numero for f in verificacion_2.fallos_estado],
            citas_reparadas_automaticamente=reparaciones_totales,
            texto_docente=formatear_para_docente(texto_final, articulos),
        )

    texto_filtrado = filtrar_no_verificado(respuesta_corregida, verificacion_2)
    texto_final = _asegurar_notas_estado(texto_filtrado, verificacion_2.fallos_estado)
    return RespuestaFinal(
        texto=texto_final,
        verificada=False,
        fue_corregida=True,
        fallos_persistentes=verificacion_2.mensajes(),
        articulos_excluidos_por_presupuesto=contexto.articulos_excluidos,
        articulos_expandidos_parcialmente=contexto.articulos_expandidos_parcialmente,
        estados_agregados_automaticamente=[f.numero for f in verificacion_2.fallos_estado],
        citas_reparadas_automaticamente=reparaciones_totales,
        texto_docente=formatear_para_docente(texto_final, articulos),
    )