"""
streamlit_app.py — FASE 2 del piloto. App de evaluacion con dos estrategias.

Para cada consulta se corre UNA vez la recuperacion y, sobre esa misma
evidencia, dos estrategias: el ASISTENTE (aplica la norma al caso) y el
BUSCADOR (muestra que dicen los articulos, sin concluir). Se muestran lado a
lado en orden ALEATORIO, como "Respuesta A" y "Respuesta B", sin decir cual es
cual. El revisor juzga cada una y al final, una sola vez, dice cual usaria. El
juicio de correccion va primero y la preferencia al final, a proposito, para
que la preferencia no contamine si la respuesta es verdadera.

GUARDADO: por ahora LOCAL, una fila JSON por consulta en registro_piloto.jsonl.
En Streamlit Cloud el disco NO persiste entre reinicios, asi que esto sirve
para la prueba en tu maquina. Cuando el flujo este andando se cambia el destino
a una planilla de Google (cambio acotado en _guardar_fila()).

El verificador de implicancia NO se corre en vivo (gastaria otra llamada a Groq
por respuesta y la cuota es ajustada). Se corre despues, offline, sobre las
respuestas guardadas, con calibrar_implicancia.py.

Esta version todavia NO tiene control de acceso por codigo: eso se agrega para
el deploy. Para la prueba local no hace falta.

Local:
    $env:GROQ_API_KEY="tu_clave"
    streamlit run streamlit_app.py
"""

import json
import os
import random
import uuid
from datetime import datetime, timezone

import streamlit as st

st.set_page_config(page_title="Estatuto Docente — evaluacion", page_icon="📘",
                   layout="wide")

try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except FileNotFoundError:
    pass

# ID de la hoja de Google (el pedazo entre /d/ y /edit de la URL).
SHEET_ID = "1omx3fUEKQc1nLz-k4MnMWX0Dva4ek2t3GX7kF4ZQgwc"
HOJA = "registro"  # nombre de la pestaña dentro de la hoja; se crea si no existe

# Version del motor que respondio. Sirve para separar evaluaciones de antes y
# despues del aviso de incisos omitidos. Subila cuando cambie el motor.
VERSION_MOTOR = "2026-10-15-aviso-incisos"

# Tres preguntas de tipo de error, una vez por respuesta. Arrancan en "—"
# (sin responder) a proposito: si arrancaran en "No", quien no las mira dejaria
# un "No" que nadie decidio y la tasa de error saldria falsamente baja.
_PREGUNTAS_TIPO_ERROR = [
    ("afirmo_falso", "Afirmo algo falso?",
     "Alguna afirmacion contradice la norma o no esta respaldada por el texto "
     "citado. Incluye aplicar una regla a quien no le corresponde y usar un "
     "plazo o tope de otro inciso."),
    ("omitio_condicion", "Omitio una condicion o excepcion pertinente?",
     "Falta una condicion, excepcion o salvedad que cambia la respuesta para "
     "este caso (por ejemplo, que la regla vale solo para titulares, o una "
     "excepcion para suplentes)."),
    ("parafrasis_infiel", "Parafraseo de forma infiel?",
     "Reformula el texto cambiando su sentido o alcance (verbo, plazo, sujeto, "
     "'puede' por 'debe'), aunque la conclusion no termine siendo falsa."),
]

# Columnas de la hoja, en orden. Una fila por consulta. Las dos estrategias y
# el detalle tecnico quedan como texto JSON en su celda, para no perder nada.
COLUMNAS = [
    "codigo", "consulta_id", "momento_utc", "pregunta", "articulos_recuperados",
    "orden_A", "orden_B",
    "texto_A", "texto_B", "tecnico_A", "tecnico_B",
    "A_incorrecto", "A_falta", "A_detalle", "A_articulo",
    "B_incorrecto", "B_falta", "B_detalle", "B_articulo",
    "cual_usaria", "comentario", "contexto_enviado",
    "A_afirmo_falso", "A_omitio_condicion", "A_parafrasis_infiel",
    "B_afirmo_falso", "B_omitio_condicion", "B_parafrasis_infiel",
    "version_motor",
]

# Respaldo local SIEMPRE, ademas de la hoja: si Google falla por lo que sea,
# el dato no se pierde (en tu maquina persiste; en la nube es un colchon hasta
# el proximo reinicio). La hoja es la fuente principal.
RUTA_RESPALDO = "registro_piloto_respaldo.jsonl"

# Codigos de acceso de los revisores. Cada uno recibe un enlace con su codigo:
#   https://TU-APP.streamlit.app/?codigo=SUTEBA01
# El codigo queda en cada fila de la hoja, para saber quien hizo cada consulta.
# Lleva aparte una lista de que codigo le diste a cada persona y su rol.
CODIGOS_VALIDOS = {
    "SUTEBA01", "SUTEBA02", "SUTEBA03", "SUTEBA04", "SUTEBA05",
    "SUTEBA06", "SUTEBA07", "SUTEBA08", "SUTEBA09", "SUTEBA10",
}


def _codigo_revisor():
    """Lee el ?codigo= del enlace. Devuelve el codigo si es valido, o None."""
    try:
        valor = st.query_params.get("codigo", "")
    except Exception:
        valor = ""
    valor = (valor or "").strip().upper()
    return valor if valor in CODIGOS_VALIDOS else None



@st.cache_resource(show_spinner=False)
def _abrir_hoja():
    """Devuelve la pestana de la hoja de Google lista para escribir, o None si
    no hay credenciales o falla la conexion (en ese caso se usa solo el
    respaldo local). La credencial sale de st.secrets["gcp_service_account"]
    en la nube, o del archivo credenciales.json en local."""
    import json as _json
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except Exception:
        return None, "Falta instalar gspread (agregalo al requirements)."

    alcance = ["https://www.googleapis.com/auth/spreadsheets"]
    info = None
    try:
        if "gcp_service_account" in st.secrets:
            info = dict(st.secrets["gcp_service_account"])
    except FileNotFoundError:
        info = None
    if info is None and os.path.exists("credenciales.json"):
        with open("credenciales.json", encoding="utf-8") as f:
            info = _json.load(f)
    if info is None:
        return None, ("No encontre credenciales: ni gcp_service_account en "
                      "Secrets ni credenciales.json en la carpeta.")
    try:
        cred = Credentials.from_service_account_info(info, scopes=alcance)
        cliente = gspread.authorize(cred)
        libro = cliente.open_by_key(SHEET_ID)
        try:
            hoja = libro.worksheet(HOJA)
        except Exception:
            hoja = libro.add_worksheet(title=HOJA, rows=1000, cols=len(COLUMNAS))
        # Encabezados: los escribe una sola vez, si la primera fila esta vacia.
        if hoja.row_values(1) != COLUMNAS:
            hoja.update(range_name="A1", values=[COLUMNAS])
        return hoja, None
    except Exception as e:  # noqa: BLE001
        return None, f"No pude conectar con la hoja: {e}"


@st.cache_resource(show_spinner="Cargando el buscador (una sola vez)...")
def cargar_motor():
    from retrieval import MotorRecuperacion
    from groq import Groq
    motor = MotorRecuperacion()
    groq = Groq(api_key=os.environ["GROQ_API_KEY"])
    return motor, groq


def _generar_dos_respuestas(pregunta, motor, groq):
    """Una recuperacion, dos estrategias sobre la misma evidencia."""
    from generacion import responder
    from buscador import responder_buscador
    resultado = motor.buscar(pregunta)
    r_asistente = responder(pregunta, resultado, groq)
    r_buscador = responder_buscador(pregunta, resultado, groq)
    arts = [a["numero"] for a in resultado.articulos]
    # Texto de cada articulo recuperado, para mostrarselo al revisor como
    # fuente. Se guarda solo en la sesion, no en la hoja (alli van los numeros).
    fuentes = [{
        "numero": a["numero"],
        "capitulo": a.get("capitulo_num", "?"),
        "ley": (a.get("ley") or "").strip(),
        "reglamentacion": (a.get("reglamentacion") or "").strip(),
    } for a in resultado.articulos]
    # El texto que efectivamente se le da al sistema (con el presupuesto de
    # caracteres aplicado). Se guarda en la hoja, porque para diagnosticar una
    # respuesta mala hace falta saber si el dato llego o no al generador.
    contexto_texto = ""
    try:
        from generacion import construir_contexto, PRESUPUESTO_CONTEXTO_CHARS
        if resultado.articulos:
            contexto_texto = construir_contexto(
                resultado.articulos, PRESUPUESTO_CONTEXTO_CHARS).texto
    except Exception:
        contexto_texto = ""
    return resultado, r_asistente, r_buscador, arts, fuentes, contexto_texto


def _fila_a_columnas(fila):
    """Aplana el dict de la consulta al orden de COLUMNAS para la hoja."""
    A, B = fila["respuesta_A"], fila["respuesta_B"]
    jA, jB = fila["juicio_A"], fila["juicio_B"]
    v = {
        "codigo": fila.get("codigo", ""),
        "consulta_id": fila["consulta_id"],
        "momento_utc": fila["momento_utc"],
        "pregunta": fila["pregunta"],
        "articulos_recuperados": json.dumps(fila["articulos_recuperados"], ensure_ascii=False),
        "orden_A": fila["orden"]["A"],
        "orden_B": fila["orden"]["B"],
        "texto_A": A["texto"], "texto_B": B["texto"],
        "tecnico_A": json.dumps(A["tecnico"], ensure_ascii=False),
        "tecnico_B": json.dumps(B["tecnico"], ensure_ascii=False),
        "A_incorrecto": jA["incorrecto"], "A_falta": jA["falta"],
        "A_detalle": jA["detalle"], "A_articulo": jA["articulo"],
        "B_incorrecto": jB["incorrecto"], "B_falta": jB["falta"],
        "B_detalle": jB["detalle"], "B_articulo": jB["articulo"],
        "cual_usaria": fila["cual_usaria"], "comentario": fila["comentario"],
        # Limite de una celda de Google Sheets: 50.000 caracteres.
        "contexto_enviado": (fila.get("contexto") or "")[:45000],
        "A_afirmo_falso": jA.get("afirmo_falso", ""),
        "A_omitio_condicion": jA.get("omitio_condicion", ""),
        "A_parafrasis_infiel": jA.get("parafrasis_infiel", ""),
        "B_afirmo_falso": jB.get("afirmo_falso", ""),
        "B_omitio_condicion": jB.get("omitio_condicion", ""),
        "B_parafrasis_infiel": jB.get("parafrasis_infiel", ""),
        "version_motor": fila.get("version_motor", ""),
    }
    return [v.get(col, "") for col in COLUMNAS]


def _guardar_fila(fila):
    """Guarda en la hoja de Google y SIEMPRE deja un respaldo local.
    Devuelve (ok, error). ok=True si al menos la hoja acepto la fila."""
    # Respaldo local primero: barato y nunca estorba.
    try:
        with open(RUTA_RESPALDO, "a", encoding="utf-8") as f:
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
    except Exception:
        pass
    hoja, err = _abrir_hoja()
    if hoja is None:
        return False, (err or "Hoja no disponible") + " (quedo en el respaldo local)"
    try:
        hoja.append_row(_fila_a_columnas(fila), value_input_option="RAW")
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, f"{e} (quedo en el respaldo local)"


def _bloque_respuesta(letra, texto):
    st.markdown(f"#### Respuesta {letra}")
    st.write(texto)


def _bloque_fuentes(c):
    """Fuentes compartidas: las dos respuestas se armaron sobre la MISMA
    evidencia, asi que se muestra una sola vez, debajo de las dos respuestas."""
    fuentes = c.get("fuentes", [])
    st.markdown("### Artículos consultados (fuente)")
    st.caption("Estos son los artículos en los que se basaron las dos respuestas. "
               "Las citas [ARTÍCULO N, CAP. M] remiten a ellos. Abrí el que quieras "
               "verificar.")
    excluidos = sorted(set(c["A"]["excluidos"]) | set(c["B"]["excluidos"]))
    if excluidos:
        st.caption("Estos artículos fueron encontrados pero no entraron completos "
                   "en lo que leyó el sistema por límite de espacio: "
                   + ", ".join(f"Art. {n}" for n in excluidos) + ".")
    with st.expander("Ver el texto de los artículos"):
        for f in fuentes:
            st.markdown(f"**Artículo {f['numero']}** (capítulo {f['capitulo']})")
            if f["ley"]:
                st.text_area("Texto del artículo", f["ley"], height=180,
                             disabled=True, key=f"src_{c['id']}_{f['numero']}_ley")
            reg = f["reglamentacion"]
            if reg and reg.strip().lower().rstrip(".") != "sin reglamentar":
                st.text_area("Reglamentación", reg, height=180, disabled=True,
                             key=f"src_{c['id']}_{f['numero']}_reg")
            else:
                st.caption("Este artículo no tiene reglamentación.")


def _form_juicio(letra, key):
    """Formulario de correccion para una respuesta. Devuelve un dict."""
    st.markdown(f"**Tu evaluacion de la Respuesta {letra}**")
    incorrecto = st.radio(
        "Hay algo incorrecto?", ["No", "Si", "No estoy seguro"],
        key=f"inc_{key}", horizontal=True, index=0)
    falta = st.radio(
        "Falta algo importante?", ["No", "Si", "No estoy seguro"],
        key=f"fal_{key}", horizontal=True, index=0)
    detalle = ""
    articulo = ""
    if incorrecto == "Si" or falta == "Si":
        detalle = st.text_area(
            "Que esta mal o que falta? (opcional)", key=f"det_{key}", height=80)
        articulo = st.text_input(
            "Sabes que articulo corresponde? (opcional)", key=f"art_{key}")
    st.caption("Tipo de error en esta respuesta (marca las tres):")
    tipos = {}
    for clave, pregunta, ayuda in _PREGUNTAS_TIPO_ERROR:
        tipos[clave] = st.radio(
            pregunta, ["—", "Si", "No"], index=0,
            key=f"{clave}_{key}", horizontal=True, help=ayuda)
    return {"incorrecto": incorrecto, "falta": falta,
            "detalle": detalle.strip(), "articulo": articulo.strip(),
            **tipos}


# ---------------------------------------------------------------- UI

st.title("📘 Estatuto Docente — evaluacion del prototipo")
st.markdown(
    "Bienvenido al asistente del Estatuto Docente de la Provincia de Buenos "
    "Aires. Esta herramienta te ayuda a consultar que dice el Estatuto y te "
    "muestra los articulos en los que se basa. No reemplaza la lectura de la "
    "norma ni el asesoramiento del gremio, y la aplicacion a tu situacion "
    "particular siempre conviene confirmarla.\n\n"
    "Podes preguntar por licencias, acrecentamiento, ingreso, estabilidad, "
    "calificaciones, carrera docente y otras situaciones que regula la "
    "normativa. Funciona mejor con consultas concretas. Cuanto mas especifica "
    "sea la pregunta, mas precisa y verificable va a ser la respuesta.\n\n"
    "Tene en cuenta que es un prototipo y puede equivocarse. Ante cualquier "
    "duda, revisa el articulo que te cita o consulta con tu gremio."
)
st.caption("No ingreses nombres, DNI ni datos personales de terceros.")

if not os.environ.get("GROQ_API_KEY"):
    st.error("Falta configurar GROQ_API_KEY.")
    st.stop()

try:
    motor, groq = cargar_motor()
except Exception as e:  # noqa: BLE001
    st.error("No se pudo cargar el buscador.")
    st.exception(e)
    st.stop()

# Puerta de acceso: sin un codigo valido en el enlace, no se puede consultar.
CODIGO = _codigo_revisor()
if CODIGO is None:
    st.warning("Para usar el asistente, entra con el enlace personal que te "
               "enviaron. Si lo tenes y aun asi ves este mensaje, revisa que "
               "el enlace este completo.")
    st.stop()

if "consulta" not in st.session_state:
    st.session_state.consulta = None

pregunta = st.text_input(
    "Escribi una consulta sobre el Estatuto Docente",
    placeholder="Por ejemplo: un suplente en tarea pasiva, que pasa con su cese?")

col_a, col_b = st.columns([1, 1])
with col_a:
    consultar = st.button("Consultar", type="primary")
with col_b:
    nueva = st.button("Nueva consulta (descartar la actual)")

if nueva:
    st.session_state.consulta = None
    st.rerun()

if consultar and pregunta.strip():
    with st.spinner("Buscando en el Estatuto y generando las dos respuestas..."):
        try:
            (resultado, r_asis, r_busc, arts, fuentes,
             contexto_texto) = _generar_dos_respuestas(pregunta, motor, groq)
        except Exception as e:  # noqa: BLE001
            st.error("Error al generar las respuestas.")
            st.exception(e)
            st.stop()

    if random.random() < 0.5:
        a_estrategia, a_resp = "asistente", r_asis
        b_estrategia, b_resp = "buscador", r_busc
    else:
        a_estrategia, a_resp = "buscador", r_busc
        b_estrategia, b_resp = "asistente", r_asis

    st.session_state.consulta = {
        "id": str(uuid.uuid4())[:8],
        "pregunta": pregunta.strip(),
        "articulos_recuperados": arts,
        "fuentes": fuentes,
        "contexto": contexto_texto,
        "orden": {"A": a_estrategia, "B": b_estrategia},
        "A": {
            "estrategia": a_estrategia,
            "texto": a_resp.texto_docente or a_resp.texto,
            "excluidos": list(getattr(a_resp, "articulos_excluidos_por_presupuesto", [])),
            "tecnico": {
                "verificada": getattr(a_resp, "verificada", None),
                "fue_corregida": getattr(a_resp, "fue_corregida", None),
                "fallos_persistentes": getattr(a_resp, "fallos_persistentes", []),
            },
        },
        "B": {
            "estrategia": b_estrategia,
            "texto": b_resp.texto_docente or b_resp.texto,
            "excluidos": list(getattr(b_resp, "articulos_excluidos_por_presupuesto", [])),
            "tecnico": {
                "verificada": getattr(b_resp, "verificada", None),
                "fue_corregida": getattr(b_resp, "fue_corregida", None),
                "fallos_persistentes": getattr(b_resp, "fallos_persistentes", []),
            },
        },
    }

c = st.session_state.consulta
if c:
    st.divider()
    st.markdown(f"**Consulta:** {c['pregunta']}")

    izq, der = st.columns(2)
    with izq:
        _bloque_respuesta("A", c["A"]["texto"])
    with der:
        _bloque_respuesta("B", c["B"]["texto"])

    st.divider()
    _bloque_fuentes(c)

    st.divider()
    st.markdown("### Tu evaluacion")
    st.caption("Mira las fuentes antes de juzgar. Primero si cada respuesta es "
               "correcta y completa; la preferencia va al final.")

    jz_izq, jz_der = st.columns(2)
    with jz_izq:
        juicio_a = _form_juicio("A", c["id"] + "A")
    with jz_der:
        juicio_b = _form_juicio("B", c["id"] + "B")

    st.markdown("### Para cerrar")
    cual = st.radio("Cual usarias?",
                    ["Respuesta A", "Respuesta B", "Las dos", "Ninguna"],
                    key="cual_" + c["id"], horizontal=True, index=0)
    comentario = st.text_area("Comentario libre (opcional)",
                              key="com_" + c["id"], height=80)

    if st.button("Guardar evaluacion", type="primary"):
        # Las tres preguntas de tipo de error son obligatorias (arrancan en "—").
        sin_responder = [
            clave for juicio in (juicio_a, juicio_b)
            for clave, _, _ in _PREGUNTAS_TIPO_ERROR
            if juicio.get(clave) == "—"
        ]
        if sin_responder:
            st.warning("Te faltan preguntas de tipo de error por responder en "
                       "alguna de las dos respuestas.")
            st.stop()
        fila = {
            "codigo": CODIGO,
            "version_motor": VERSION_MOTOR,
            "consulta_id": c["id"],
            "momento_utc": datetime.now(timezone.utc).isoformat(),
            "pregunta": c["pregunta"],
            "articulos_recuperados": c["articulos_recuperados"],
            "contexto": c.get("contexto", ""),
            "orden": c["orden"],
            "respuesta_A": c["A"],
            "respuesta_B": c["B"],
            "juicio_A": juicio_a,
            "juicio_B": juicio_b,
            "cual_usaria": cual,
            "comentario": comentario.strip(),
        }
        ok, err = _guardar_fila(fila)
        if ok:
            st.success("Gracias, evaluacion guardada. Podes hacer otra consulta.")
            st.session_state.consulta = None
        else:
            st.error(f"No se pudo guardar: {err}")