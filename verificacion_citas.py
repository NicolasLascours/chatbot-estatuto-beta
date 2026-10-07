"""
verificacion_citas.py — Verificación determinista de citas del generador.

Este módulo NO llama a ningún LLM — la garantía de veracidad no puede
depender de que otro modelo "revise" al primero, tiene que apoyarse en algo
verificable por código contra los datos que el motor de recuperación
efectivamente devolvió.

Formato de cita (revisado esta sesión — antes decía "LEY", lo cual sugería
dos fuentes separadas cuando en realidad es un solo artículo del Estatuto
con una capa de texto y, opcionalmente, una capa de reglamentación):

    [ARTÍCULO 114, CAP. 20]                    — texto del artículo
    [ARTÍCULO 114, CAP. 20 — REGLAMENTACIÓN]   — reglamentación de ese artículo
    [ARTÍCULO 114 — ESTADO: SIN REGLAMENTAR]   — declaración obligatoria del
                                                  estado, cuando no es "reglamentado"

Trabaja sobre TRES tipos de fallo:

  - FALLO DE CITA: se citó un artículo/sección que no está entre los
    recuperados (`articulo_no_recuperado`), que está vacía en el padre
    recuperado (`seccion_vacia`), o que SÍ existía en el padre pero no se
    envió al generador para esta pregunta (`seccion_no_enviada`) porque
    quedó afuera por presupuesto de caracteres o porque su reglamentación
    era la anotación vacía "Sin reglamentar." que construir_contexto()
    descarta. Esta última razón cierra una fuga: el verificador tomaba
    como verdad de referencia el PADRE recuperado (indice_padres.json), que
    es un superconjunto de lo que efectivamente viajó a Groq, así que una
    cita a una sección recuperada-pero-no-enviada pasaba como válida. Ahora,
    cuando responder() le pasa el mapa de niveles por sección que produce
    construir_contexto(), la verdad de referencia es lo que se envió.

  - FALLO DE COBERTURA: una oración con contenido normativo sin ninguna
    cita adjunta (heurístico, no NLI — ver docstring de verificar_cobertura).

  - FALLO DE ESTADO NO DECLARADO: se citó un artículo cuyo
    `estado_reglamentacion` no es "reglamentado", pero la respuesta no
    incluye el tag [ARTÍCULO N — ESTADO: ...] que lo declara. Regla del
    proyecto: si un artículo está "sin reglamentar", eso se aclara SIEMPRE,
    sin excepción — este fallo es justamente el chequeo por código de esa
    regla (pedirlo solo en el prompt no garantiza que se cumpla, ya lo
    vimos con el formato de cita al principio de esta etapa).

Además de estos tres fallos, este módulo tiene reparar_citas_huerfanas(),
que NO detecta un fallo nuevo — corre ANTES de verificar_respuesta(), como
preprocesamiento, y arregla por código un patrón puntual de fallo de
cobertura (oración huérfana en una lista con viñetas cuya oración
siguiente sí tiene una única cita sin ambigüedad) para no perder ese dato
en vez de gastar una regeneración completa por algo resoluble sin LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ----------------------------------------------------------------------
# Patrones
# ----------------------------------------------------------------------

# Cita de contenido: número + capítulo, con "— REGLAMENTACIÓN" opcional.
# Ya no se usa la palabra "LEY" — un artículo del Estatuto no es "ley" vs.
# "otra cosa", es el artículo mismo vs. su reglamentación.
PATRON_CITA = re.compile(
    r"\[ARTÍCULO\s+(\d+)\s*,\s*CAP\.?\s*\d+\s*(?:—\s*(REGLAMENTACI[ÓO]N))?\]",
    re.IGNORECASE,
)

# Tag de estado, obligatorio cuando el artículo citado no está reglamentado.
# Deliberadamente laxo en el texto después de "ESTADO:" — no exigimos
# vocabulario exacto, solo que el tag exista para ese número de artículo.
PATRON_ESTADO = re.compile(
    r"\[ARTÍCULO\s+(\d+)\s*—\s*ESTADO:[^\]]+\]",
    re.IGNORECASE,
)

_DIVISOR_ORACIONES = re.compile(r"(?<=[.!?])\s+")

_ABREVIATURAS = [
    "Art", "Arts", "Nº", "Ud", "Uds", "Sr", "Sra", "Dr", "Dra",
    "etc", "pág", "cap", "inc", "núm",
]
_PATRON_ABREVIATURA = re.compile(
    r"\b(?:" + "|".join(re.escape(a) for a in _ABREVIATURAS) + r")\.(?=\s|$)",
    re.IGNORECASE,
)
_MARCA_PUNTO_PROTEGIDO = "\x00"

_MARCADORES_NORMATIVOS = re.compile(
    r"\b(corresponde[n]?|tiene[n]? derecho|deber[áa]n?|puede[n]? solicitar|"
    r"se otorga[n]?|se reconoce[n]?|goza[n]? de|con goce|sin goce|"
    r"licencia (?:de|por)|d[íi]as? (?:h[áa]biles?|corridos?)|"
    r"por ciento|años? de antig[üu]edad)\b",
    re.IGNORECASE,
)
_TIENE_CIFRA = re.compile(r"\d")

# AUSENCIA (genérica) vs. NEGACIÓN NORMATIVA (puntual) — la distinción que
# faltaba y que produjo el caso real N1-6 (calificación mínima del art. 18)
# y contribuyó a N1-2 (días de licencia por enfermedad).
#
# Antes había una sola lista, _MARCADORES_AUSENCIA, pensada para dejar
# pasar sin cita una admisión honesta del propio sistema ("no encontré
# esto en lo que tengo disponible" — no se puede citar una ausencia). El
# problema es que esa lista usaba verbos genéricos ("no establece", "no
# dispone", "no es posible determinar") que son EXACTAMENTE lo mismo que
# usaría una negación categórica y falsa sobre el Estatuto mismo ("el
# Estatuto no establece un promedio mínimo"). Las dos frases matcheaban
# igual y las dos quedaban exentas de necesitar cita, aunque una sea una
# admisión de límite del sistema y la otra sea una afirmación normativa
# (falsa) sobre la ley.
#
# La distinción real no está en el verbo, está en si la negación se
# adjunta a un CONCEPTO normativo concreto (un mínimo, un plazo, un
# monto, un derecho puntual) o no. "No es posible determinar" a secas es
# un límite honesto. "No es posible determinar cuántos DÍAS de licencia"
# ya está haciendo una afirmación sobre el contenido de la norma, no solo
# sobre lo que el sistema pudo ver, y esa sí necesita respaldo como
# cualquier otra afirmación normativa.
#
# Por eso _MARCADORES_NEGACION_NORMATIVA exige el verbo de negación Y,
# cerca, una palabra de concepto normativo — y en verificar_cobertura()
# esta lista se chequea ANTES que _MARCADORES_AUSENCIA y, si matchea,
# gana ella. Una negación genérica sin concepto normativo cerca sigue
# exenta como siempre.
#
# Dos formas, porque el concepto puede ir antes o después del verbo de
# negación ("no se describe la sanción" vs. "la sanción no está
# especificada"), y en oraciones largas con "ni" (ver caso real N3-6,
# "no se describen los pasos... ni la sanción que podría imponerse") el
# concepto puede quedar bastante lejos del verbo — la ventana es más
# ancha que la de _MARCADORES_NORMATIVOS por eso mismo, a costa de
# alguna falsa alarma ocasional, preferible a dejar pasar una negación
# categórica sin marcar.
_CONCEPTOS_NORMATIVOS = (
    r"m[íi]nimo[s]?|m[áa]ximo[s]?|plazo[s]?|d[íi]as?|por ciento|"
    r"derecho[s]?|promedio[s]?|antig[üu]edad|licencia[s]?|monto[s]?|"
    r"porcentaje[s]?|sanci[óo]n(?:es)?|procedimiento[s]?|cesant[íi]a|"
    r"beneficio[s]?|excepci[óo]n(?:es)?|requisito[s]?|remuneraci[óo]n|"
    r"retribuci[óo]n|categor[íi]a|ascenso[s]?|estabilidad|paso[s]?"
)
_MARCADORES_NEGACION_NORMATIVA = re.compile(
    r"\bno (?:fija|establece|prev[ée]|contempla|reconoce|otorga|tiene|"
    r"existe|hay|dispone|aparece|describe[n]?|contien(?:e|en)|incluye[n]?|"
    r"regula[n]?|es posible (?:determinar|indicar|precisar|"
    r"establecer)|se (?:establece|indica|especifica|"
    r"describe[n]?|mencionan?))\b"
    r".{0,70}\b(?:" + _CONCEPTOS_NORMATIVOS + r")\b"
    r"|"
    r"\b(?:" + _CONCEPTOS_NORMATIVOS + r")\b.{0,70}\bno (?:est[áa]n?|"
    r"son|fue(?:ron)?) (?:especificad[oa]s?|previst[oa]s?|"
    r"contemplad[oa]s?|regulad[oa]s?|determinad[oa]s?)\b",
    re.IGNORECASE,
)

# Excepción a la negación normativa: si la oración ya deja explícito que
# la ausencia es sobre LO QUE EL SISTEMA TIENE A MANO (el fragmento, el
# contexto, el texto que se le pasó), no sobre el Estatuto en sí, es
# justo la redacción correcta que pedimos en PROMPT_SISTEMA (regla 5) y
# en generar_correccion() — no tiene sentido forzarla a corrección ni
# arriesgarse a que filtrar_no_verificado() la borre y la reemplace por
# el mensaje genérico de "no pude construir una respuesta verificable",
# perdiendo una explicación honesta y ya bien encuadrada. Esto se chequea
# ANTES que _MARCADORES_NEGACION_NORMATIVA y gana si matchea.
_MARCADORES_AUSENCIA_ESCOPADA = re.compile(
    r"\b((?:en )?(?:el|los) (?:texto|fragmento|contexto)s?\b.{0,40}\b"
    r"(?:que (?:nos |te )?(?:entreg\w*|proporcion\w*|pas\w*)|"
    r"disponible[s]?|recuperado[s]?)|"
    r"no est[áa]n? en (?:el|los) (?:texto|fragmento|contexto)s?|"
    r"lo (?:recuperado|disponible))\b",
    re.IGNORECASE,
)

# Admisión honesta y genérica de un límite del propio sistema, sin
# adjuntarse a un concepto normativo concreto (ver arriba). Esto sigue
# exento de cita, como siempre — no se puede citar una ausencia.
_MARCADORES_AUSENCIA = re.compile(
    r"\b(no contiene|no es posible (?:determinar|indicar|precisar|"
    r"establecer)|no aparece|no est[áa] (?:especificad[oa]|inclu[íi]d[oa]|"
    r"determinad[oa])|no fue posible|no se (?:indica|especifica|menciona)|"
    r"no dispone|no establece|no hay (?:disposici[óo]n|informaci[óo]n)|"
    r"sin informaci[óo]n disponible)\b",
    re.IGNORECASE,
)

# VEREDICTO SIN CIFRA — una afirmación de permiso/prohibición ("sí puede",
# "le está permitido", "no le corresponde") es tan normativa como un plazo o
# un monto, pero no trae ninguna cifra ni ninguno de los verbos de
# _MARCADORES_NORMATIVOS, así que hasta ahora se escapaba del chequeo de
# cobertura y quedaba sin exigencia de cita. Es exactamente la fuga que
# produjo el caso real N2-8: el generador abrió con "Sí, el docente puede
# asumir un segundo cargo de base…" (conclusión falsa, art. 28 mal leído),
# el verificador filtró la oración que sí traía la cita pero dejó viva esta,
# y el docente terminó leyendo un "sí" rotundo y equivocado.
#
# El patrón está deliberadamente acotado al veredicto AFIRMATIVO, que es la
# forma que desinforma (un falso "sí puede"). Un "No," inicial se deja
# afuera a propósito: se solapa con las admisiones honestas de ausencia
# ("No está en el fragmento…", "No hay disposición…") que ya quedan exentas
# más arriba, y forzarlo arriesga borrar justo la redacción prudente que el
# prompt pide. Este chequeo corre DESPUÉS de las exenciones de ausencia y
# ausencia-escopada, así que un "no le corresponde" que en realidad es un
# límite del sistema ("…no le corresponde según el fragmento disponible")
# ya salió exento antes de llegar acá. La tilde de "Sí" y la coma del
# veredicto son las que lo separan del "Si" condicional de "Si tu consulta
# no quedó respondida…" (nota de pie del propio sistema), que NO debe caer.
_MARCADORES_VEREDICTO = re.compile(
    r"^\s*s[íi]\s*,"                                   # "Sí, ..." (veredicto)
    r"|\bs[íi]\s+puede[n]?\b"                          # "sí puede / sí pueden"
    r"|\b(?:le\s+)?est[áa]\s+"
    r"(?:permitid|autorizad|habilitad|prohibid|impedid|vedad)"
    r"|\bno\s+le\s+corresponde",
    re.IGNORECASE,
)


def _dividir_oraciones(texto: str) -> list[str]:
    protegido = _PATRON_ABREVIATURA.sub(
        lambda m: m.group(0)[:-1] + _MARCA_PUNTO_PROTEGIDO, texto
    )
    partes = _DIVISOR_ORACIONES.split(protegido.strip())
    return [
        p.replace(_MARCA_PUNTO_PROTEGIDO, ".").strip()
        for p in partes
        if p.strip()
    ]


# ----------------------------------------------------------------------
# Estructuras de resultado
# ----------------------------------------------------------------------

@dataclass
class FalloCita:
    numero: int
    tipo: str  # "LEY" (texto del artículo) | "REGLAMENTACIÓN"
    razon: str  # "articulo_no_recuperado" | "seccion_vacia" | "seccion_no_enviada"

    def mensaje(self) -> str:
        parte = "la reglamentación del" if self.tipo == "REGLAMENTACIÓN" else "el texto del"
        if self.razon == "articulo_no_recuperado":
            return (
                f"Citaste el ARTÍCULO {self.numero}, pero ese artículo no "
                "está entre los recuperados para esta pregunta."
            )
        if self.razon == "seccion_no_enviada":
            return (
                f"Citaste {parte} ARTÍCULO {self.numero}, pero esa sección no "
                "se incluyó en el contexto que tenés delante para esta "
                "pregunta (quedó afuera por espacio, o no tiene "
                "reglamentación operativa vigente). No la cites: no tenés su "
                "texto disponible para respaldar la afirmación."
            )
        return (
            f"Citaste {parte} ARTÍCULO {self.numero}, pero esa sección está "
            "vacía en el contexto — no hay texto ahí que respalde la "
            "afirmación."
        )


@dataclass
class FalloEstadoNoDeclarado:
    numero: int
    estado: str  # "sin_reglamentar" | "parcial" | "sin_dato"

    def mensaje(self) -> str:
        return (
            f"Citaste el ARTÍCULO {self.numero}, cuyo estado de "
            f"reglamentación es '{self.estado}' (no está reglamentado "
            "completamente), pero tu respuesta no incluye el tag "
            f"[ARTÍCULO {self.numero} — ESTADO: ...] que es obligatorio "
            "en ese caso."
        )


@dataclass
class ResultadoVerificacion:
    valida: bool
    fallos_cita: list[FalloCita] = field(default_factory=list)
    fallos_cobertura: list[str] = field(default_factory=list)
    fallos_estado: list[FalloEstadoNoDeclarado] = field(default_factory=list)
    # True si el texto llegó vacío o solo con espacios. Caso aparte de los
    # otros tres fallos porque no hay ninguna oración ni cita que evaluar
    # — las otras tres listas dan vacías por ausencia de contenido, no
    # porque el contenido esté bien, y sin este chequeo un texto vacío
    # pasaba como "válido" (ver casos reales N2-4 y N2-8, donde Groq
    # devolvió respuesta vacía y verificar_respuesta() la aprobó igual).
    fallo_vacio: bool = False

    def mensajes(self) -> list[str]:
        msgs = [f.mensaje() for f in self.fallos_cita]
        msgs += [f.mensaje() for f in self.fallos_estado]
        msgs += [
            f'La afirmación "{o}" tiene contenido normativo (plazos, montos, '
            "derechos) pero no tiene ninguna cita [ARTÍCULO N, CAP. M] adjunta."
            for o in self.fallos_cobertura
        ]
        if self.fallo_vacio:
            msgs.append(
                "La respuesta generada quedó vacía (sin texto), no hay "
                "contenido que mostrarle al docente."
            )
        return msgs


# ----------------------------------------------------------------------
# Funciones principales
# ----------------------------------------------------------------------

def extraer_citas(texto: str) -> list[tuple[int, str]]:
    """Citas de CONTENIDO (texto del artículo o su reglamentación). No
    incluye los tags de ESTADO — para eso está extraer_estados_declarados."""
    resultado = []
    for numero, marcador_reg in PATRON_CITA.findall(texto):
        tipo = "REGLAMENTACIÓN" if marcador_reg else "LEY"
        resultado.append((int(numero), tipo))
    return resultado


def extraer_estados_declarados(texto: str) -> set[int]:
    """Números de artículo para los que el texto incluye un tag
    [ARTÍCULO N — ESTADO: ...]."""
    return {int(n) for n in PATRON_ESTADO.findall(texto)}


def _dividir_oraciones_con_posiciones(texto: str) -> list[tuple[int, int]]:
    """Igual que _dividir_oraciones(), pero devuelve (inicio, fin) en vez
    del texto ya recortado — necesario para reparar_citas_huerfanas(),
    que inserta texto en posiciones exactas del ORIGINAL en vez de
    reconstruir todo por join (ver por qué en esa función).

    La sustitución de abreviaturas reemplaza un '.' por un solo carácter
    (`_MARCA_PUNTO_PROTEGIDO`), así que `protegido` tiene el mismo largo
    que `texto` y las posiciones son válidas en los dos sin traducción."""
    protegido = _PATRON_ABREVIATURA.sub(
        lambda m: m.group(0)[:-1] + _MARCA_PUNTO_PROTEGIDO, texto
    )
    spans: list[tuple[int, int]] = []
    inicio = 0
    for m in _DIVISOR_ORACIONES.finditer(protegido):
        fin = m.start()
        if protegido[inicio:fin].strip():
            spans.append((inicio, fin))
        inicio = m.end()
    if protegido[inicio:].strip():
        spans.append((inicio, len(protegido)))
    return spans


def reparar_citas_huerfanas(texto: str) -> tuple[str, list[tuple[str, str]]]:
    """Repara determinísticamente un patrón de fallo de cobertura muy
    específico: una oración con contenido normativo, sin cita propia,
    seguida inmediatamente por otra oración que SÍ tiene una única cita
    sin ambigüedad. Típico de listas con viñetas donde el modelo le pone
    la cita solo al último ítem de una racha que comparte artículo (caso
    real: titular/provisional vs. suplente del art. 114, ver notas de
    sesión de generación parcial).

    Deliberadamente conservador: solo actúa cuando la oración siguiente
    tiene EXACTAMENTE una cita. Si hay ambigüedad (0 o 2+ citas en la
    oración siguiente), no toca nada — la oración huérfana sigue su curso
    normal y puede terminar filtrada por filtrar_no_verificado(), como
    hasta ahora. No verifica acá si esa cita es válida contra el contexto
    recuperado (artículo existente, sección no vacía) — eso lo sigue
    haciendo verificar_respuesta() después, sobre el texto ya reparado, es
    el mismo chequeo de siempre, no uno nuevo.

    Trabaja sobre POSICIONES del texto original, no reconstruye todo por
    join — un primer intento de esta función unía las oraciones con
    `" ".join()`, y aunque no hubiera nada que reparar, ese round-trip de
    separar-y-volver-a-unir igual aplastaba saltos de línea entre viñetas
    y cualquier otro formato ajeno a las citas (confirmado con Groq real:
    una lista con viñetas en líneas separadas volvía pegada en una sola
    línea). Acá se inserta la cita en el punto exacto donde hace falta y
    el resto del texto queda byte a byte igual al original.

    Corre como PREPROCESAMIENTO, antes de verificar_respuesta(), no
    reemplaza a filtrar_no_verificado() — una reparación exitosa evita
    gastar una regeneración completa por algo que se podía arreglar por
    código; lo que esta función no logra reparar sigue el camino de
    siempre (corrección con el LLM, y si eso tampoco alcanza, se filtra)."""
    spans = _dividir_oraciones_con_posiciones(texto)
    reparadas: list[tuple[str, str]] = []
    inserciones: list[tuple[int, str]] = []  # (posición, texto a insertar)

    for i, (inicio, fin) in enumerate(spans):
        oracion = texto[inicio:fin]
        if PATRON_CITA.search(oracion) or PATRON_ESTADO.search(oracion):
            continue
        if _MARCADORES_AUSENCIA_ESCOPADA.search(oracion):
            continue
        # Mismo criterio que verificar_cobertura() — ver ahí el porqué.
        es_negacion_normativa = bool(_MARCADORES_NEGACION_NORMATIVA.search(oracion))
        if not es_negacion_normativa and _MARCADORES_AUSENCIA.search(oracion):
            continue
        es_normativa = bool(
            es_negacion_normativa
            or _MARCADORES_NORMATIVOS.search(oracion)
            or _TIENE_CIFRA.search(oracion)
        )
        if not es_normativa:
            continue

        if i + 1 >= len(spans):
            continue  # es la última oración del texto, no hay siguiente
        sig_inicio, sig_fin = spans[i + 1]
        siguiente = texto[sig_inicio:sig_fin]
        candidatas = list(PATRON_CITA.finditer(siguiente))
        if len(candidatas) != 1:
            continue  # ambigüedad (0 o 2+), sigue huérfana, la agarra el filtro de siempre

        cita_texto = candidatas[0].group(0)
        # Buscamos el último '.', '!' o '?' DENTRO de esta oración para
        # insertar la cita justo antes — si insertáramos después del
        # punto final, el separador de oraciones (que corta después de
        # '.', '!' o '?' seguido de espacio) volvería a partirla en dos
        # al re-analizar el texto reparado más adelante, y la cita
        # quedaría pegada a la oración siguiente en vez de a esta.
        oracion_sin_espacios = oracion.rstrip()
        offset_relativo = len(oracion_sin_espacios)
        if oracion_sin_espacios and oracion_sin_espacios[-1] in ".!?":
            offset_relativo -= 1
        posicion_insercion = inicio + offset_relativo
        inserciones.append((posicion_insercion, f" {cita_texto}"))
        reparadas.append((oracion.strip(), cita_texto))

    if not inserciones:
        return texto, []

    # De atrás para adelante, así insertar no corre las posiciones de las
    # inserciones que todavía faltan aplicar.
    resultado = texto
    for posicion, insercion in sorted(inserciones, key=lambda x: x[0], reverse=True):
        resultado = resultado[:posicion] + insercion + resultado[posicion:]

    return resultado, reparadas


def verificar_cobertura(texto: str) -> list[str]:
    sospechosas = []
    for oracion in _dividir_oraciones(texto):
        if PATRON_CITA.search(oracion) or PATRON_ESTADO.search(oracion):
            continue
        if _MARCADORES_AUSENCIA_ESCOPADA.search(oracion):
            continue
        # La negación normativa gana por sobre la exención de ausencia —
        # ver el docstring de _MARCADORES_NEGACION_NORMATIVA. Una negación
        # sobre un concepto normativo puntual necesita respaldo, no queda
        # exenta solo porque use un verbo de ausencia.
        es_negacion_normativa = bool(_MARCADORES_NEGACION_NORMATIVA.search(oracion))
        if not es_negacion_normativa and _MARCADORES_AUSENCIA.search(oracion):
            continue
        es_normativa = bool(
            es_negacion_normativa
            or _MARCADORES_NORMATIVOS.search(oracion)
            or _TIENE_CIFRA.search(oracion)
            or _MARCADORES_VEREDICTO.search(oracion)
        )
        if es_normativa:
            sospechosas.append(oracion)
    return sospechosas


def verificar_respuesta(
    texto: str,
    articulos: list[dict],
    niveles_por_seccion: dict[int, dict[str, str]] | None = None,
) -> ResultadoVerificacion:
    """`niveles_por_seccion` es el mapa {numero: {"ley": nivel,
    "reglamentacion": nivel}} que produce construir_contexto() en
    generacion.py, con nivel en {"completo", "parcial", "nada"}. Es la
    verdad de referencia de qué se ENVIÓ realmente al generador para esta
    pregunta, que NO coincide con los padres recuperados: un padre puede
    traer una sección que quedó afuera del contexto por presupuesto, o una
    reglamentación "Sin reglamentar." que construir_contexto() descarta. En
    esos casos la sección no viajó y una cita a ella no puede sostenerse
    (razón `seccion_no_enviada`), aunque el padre la tenga.

    responder() (el único camino de producción) SIEMPRE lo pasa. El default
    None conserva el comportamiento anterior, que chequea contra el padre
    recuperado — se mantiene solo para no romper llamadas y tests sintéticos
    viejos que no se ocupan de esta distinción. En None, la fuga que este
    parámetro cierra sigue abierta, por eso en producción no es opcional."""
    if not texto or not texto.strip():
        return ResultadoVerificacion(valida=False, fallo_vacio=True)

    por_numero = {a["numero"]: a for a in articulos}

    fallos_cita: list[FalloCita] = []
    vistos: set[tuple[int, str]] = set()
    numeros_citados: set[int] = set()

    for numero, tipo in extraer_citas(texto):
        numeros_citados.add(numero)
        if (numero, tipo) in vistos:
            continue
        vistos.add((numero, tipo))

        art = por_numero.get(numero)
        if art is None:
            fallos_cita.append(FalloCita(numero, tipo, "articulo_no_recuperado"))
            continue

        campo = "ley" if tipo == "LEY" else "reglamentacion"
        padre_no_vacio = bool((art.get(campo) or "").strip())

        if niveles_por_seccion is not None:
            # Verdad de referencia = lo que efectivamente se envió. Nivel
            # "completo" o "parcial" -> la sección viajó (la parcial manda el
            # texto recortado con su aviso, la cita es igual de legítima).
            # "nada" o ausente -> no viajó: si el padre la tenía es una
            # sección recuperada-pero-no-enviada, si el padre estaba vacío es
            # el seccion_vacia de siempre. Un nivel no-"nada" implica padre no
            # vacío por construcción de _intentar_tipo, así que no hace falta
            # cruzar los dos chequeos.
            nivel = niveles_por_seccion.get(numero, {}).get(campo, "nada")
            if nivel == "nada":
                razon = "seccion_no_enviada" if padre_no_vacio else "seccion_vacia"
                fallos_cita.append(FalloCita(numero, tipo, razon))
        else:
            # Comportamiento anterior (sin mapa): se chequea contra el padre.
            if not padre_no_vacio:
                fallos_cita.append(FalloCita(numero, tipo, "seccion_vacia"))

    fallos_cobertura = verificar_cobertura(texto)

    declarados = extraer_estados_declarados(texto)
    fallos_estado: list[FalloEstadoNoDeclarado] = []
    for numero in numeros_citados:
        art = por_numero.get(numero)
        if art is None:
            continue  # ya cubierto arriba por articulo_no_recuperado
        # Preferimos el estado calculado por construir_contexto() para ESTE
        # contexto puntual (ver _calcular_estado_efectivo en generacion.py)
        # sobre el campo de artículo completo que calcula parser_estatuto.py
        # — evita declarar "parcial" cuando lo que efectivamente se citó
        # está reglamentado. Si construir_contexto() no lo calculó (artículo
        # sin incisos con letra), usamos el campo original.
        estado = art.get("estado_reglamentacion_efectivo", art.get("estado_reglamentacion"))
        if estado and estado != "reglamentado" and numero not in declarados:
            fallos_estado.append(FalloEstadoNoDeclarado(numero, estado))

    return ResultadoVerificacion(
        valida=not fallos_cita and not fallos_cobertura and not fallos_estado,
        fallos_cita=fallos_cita,
        fallos_cobertura=fallos_cobertura,
        fallos_estado=fallos_estado,
    )


def filtrar_no_verificado(texto: str, resultado: ResultadoVerificacion) -> str:
    """Descarta oraciones con fallo de cita o de cobertura. Los fallos de
    ESTADO no se manejan acá — no hay una "oración a borrar" para un tag
    que falta, se resuelven agregando la nota por código en generacion.py
    (ver _asegurar_notas_estado), no borrando contenido."""
    numeros_tipos_fallidos = {(f.numero, f.tipo) for f in resultado.fallos_cita}
    oraciones_cobertura_fallida = set(resultado.fallos_cobertura)

    oraciones = _dividir_oraciones(texto)
    conservadas = []
    se_omitio_algo = resultado.fallo_vacio

    for oracion in oraciones:
        citas_de_la_oracion = set(extraer_citas(oracion))
        if citas_de_la_oracion & numeros_tipos_fallidos:
            se_omitio_algo = True
            continue
        if oracion in oraciones_cobertura_fallida:
            se_omitio_algo = True
            continue
        conservadas.append(oracion)

    texto_filtrado = " ".join(conservadas).strip()

    if se_omitio_algo:
        nota = (
            "\n\n(Nota: se omitió parte de la respuesta generada porque no "
            "pudo verificarse contra los artículos recuperados. Si tu "
            "consulta no quedó completamente respondida, te recomendamos "
            "revisar el artículo directamente o reformular la pregunta.)"
        )
        texto_filtrado = (texto_filtrado or "No pude construir una respuesta "
                           "verificable para esta consulta con las fuentes "
                           "recuperadas.") + nota

    return texto_filtrado