# app.py - versión actualizada: evaluación por pregunta para el cuestionario
# Nota: elimina claves API en el código y usa variables de entorno (.env o exportadas).
#       Ejemplo local: crear .env con GENAI_API_KEY=tu_api_key y pip install python-dotenv
import re
import streamlit as st
import sqlite3
import pdfplumber
import requests
import pandas as pd
from io import BytesIO
import os
import json
from datetime import datetime, timezone

# Intentar cargar .env si está disponible (opcional, no obligatorio)
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# Inicializar cliente Gemeni/GenAI de forma segura (usar GENAI_API_KEY en el entorno)
client = None
gemini_model = "gemini-2.0-flash"
try:
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")  # lee de la variable de entorno
    if api_key:
        client = genai.Client(api_key=api_key)
    else:
        try:
            # intenta crear client sin api_key explícita (ADC u otra configuración)
            client = genai.Client()
        except Exception:
            client = None
except Exception:
    client = None

DB_PATH = "notas.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS badges (
        url TEXT PRIMARY KEY,
        id TEXT,
        nrc TEXT,
        nombre TEXT,
        curso TEXT,
        horas TEXT,
        fecha TEXT,
        valido INTEGER,
        mensaje TEXT,
        nota_taller REAL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS taller_resultados (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        id_estudiante TEXT,
        nrc_curso TEXT,
        nota_taller REAL,
        feedback TEXT,
        comentarios TEXT,
        fecha_registro TEXT
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS respuestas_cuestionario (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        id_estudiante TEXT,
        nrc_curso TEXT,
        respuestas TEXT,
        nota_cuestionario REAL,
        feedback TEXT,
        fecha_registro TEXT
    )
    """)
    conn.commit()
    conn.close()

# DB helpers
def guardar_en_db(id_est, nrc, nombre, curso, horas, fecha, url, valido, mensaje, nota_taller=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR REPLACE INTO badges (url, id, nrc, nombre, curso, horas, fecha, valido, mensaje, nota_taller)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (url, id_est, nrc, nombre, curso, horas, fecha, int(bool(valido)), mensaje, nota_taller))
    conn.commit()
    conn.close()

def obtener_registros():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM badges", conn)
    conn.close()
    return df

def obtener_calificaciones_badges():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT 
            id AS id_estudiante,
            nrc AS nrc_curso,
            SUM(CASE WHEN valido=1 THEN 1 ELSE 0 END) * 0.25 AS calificacion_badge
        FROM badges
        GROUP BY id, nrc
    """, conn)
    conn.close()
    return df

def guardar_resultado_taller(id_estudiante, nrc_curso, nota_taller, feedback, comentarios):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Guardar fecha como ISO string (timezone-aware) para evitar adaptadores de datetime
    fecha_iso = datetime.now(timezone.utc).isoformat()
    c.execute("""
        INSERT INTO taller_resultados (id_estudiante, nrc_curso, nota_taller, feedback, comentarios, fecha_registro)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (id_estudiante, nrc_curso, nota_taller, feedback, comentarios, fecha_iso))
    conn.commit()
    conn.close()

def obtener_aggregado_taller():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT 
            id_estudiante,
            nrc_curso,
            SUM(COALESCE(nota_taller,0)) AS calificacion_taller,
            COUNT(*) AS n_registros_taller
        FROM taller_resultados
        GROUP BY id_estudiante, nrc_curso
    """, conn)
    conn.close()
    return df

def guardar_respuestas_cuestionario(id_estudiante, nrc_curso, respuestas_dict, nota_cuestionario, feedback):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    fecha_iso = datetime.now(timezone.utc).isoformat()
    c.execute("""
        INSERT INTO respuestas_cuestionario (id_estudiante, nrc_curso, respuestas, nota_cuestionario, feedback, fecha_registro)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        id_estudiante,
        nrc_curso,
        json.dumps(respuestas_dict, ensure_ascii=False),
        nota_cuestionario,
        feedback,
        fecha_iso
    ))
    conn.commit()
    conn.close()

def obtener_aggregado_cuestionario():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT 
            id_estudiante,
            nrc_curso,
            SUM(COALESCE(nota_cuestionario,0)) AS calificacion_cuestionario,
            COUNT(*) AS n_registros_cuestionario
        FROM respuestas_cuestionario
        GROUP BY id_estudiante, nrc_curso
    """, conn)
    conn.close()
    return df

def obtener_ultimo_taller():
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT tr.id_estudiante, tr.nrc_curso, tr.nota_taller AS calificacion_taller, tr.fecha_registro
        FROM taller_resultados tr
        INNER JOIN (
            SELECT id_estudiante, nrc_curso, MAX(fecha_registro) AS max_fecha
            FROM taller_resultados
            GROUP BY id_estudiante, nrc_curso
        ) ult
        ON tr.id_estudiante = ult.id_estudiante
        AND tr.nrc_curso = ult.nrc_curso
        AND tr.fecha_registro = ult.max_fecha
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def obtener_ultimo_cuestionario():
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT rc.id_estudiante, rc.nrc_curso, rc.nota_cuestionario AS calificacion_cuestionario, rc.fecha_registro
        FROM respuestas_cuestionario rc
        INNER JOIN (
            SELECT id_estudiante, nrc_curso, MAX(fecha_registro) AS max_fecha
            FROM respuestas_cuestionario
            GROUP BY id_estudiante, nrc_curso
        ) ult
        ON rc.id_estudiante = ult.id_estudiante
        AND rc.nrc_curso = ult.nrc_curso
        AND rc.fecha_registro = ult.max_fecha
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# PDF extraction
def extraer_texto_y_urls(file_bytes):
    texto = ""
    urls = []
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        pages_text = []
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)
            texto += page_text + "\n"

        # Juntamos todo el texto para capturar URLs que puedan romperse en saltos de línea
        whole_text = "\n".join(pages_text)

        # Buscar coincidencias que empiecen en credly.com/go/ y que puedan tener saltos o espacios
        # Luego normalizaremos quitando espacios internos y signos terminales comunes.
        raw_matches = re.findall(r"(https?://(?:www\.)?credly\.com/go/[A-Za-z0-9\-\._~%/\\\n\r\t]+)", whole_text, flags=re.IGNORECASE)
        for m in raw_matches:
            # Normalizar: quitar saltos de línea/espacios internos
            url = re.sub(r"\s+", "", m)
            # Quitar puntuación terminal que suele venir pegada en textos (.,;:) o paréntesis/quotes
            url = url.rstrip(".,;:)\"]'")
            urls.append(url)

    # Deduplicar preservando el orden
    unique_urls = list(dict.fromkeys(urls))
    return texto.strip(), unique_urls

def procesar_texto(texto):
    lineas = [l.strip() for l in texto.splitlines() if l.strip()]

    curso_idx = None
    for i, linea in enumerate(lineas):
        if re.match(r"^AWS", linea, re.IGNORECASE):
            curso_idx = i
            break

    if curso_idx is None or curso_idx + 2 >= len(lineas):
        return None, None, None, None

    curso = lineas[curso_idx]
    horas = lineas[curso_idx + 1]
    fecha = lineas[curso_idx + 2]
    nombre = " ".join(lineas[:curso_idx])

    return nombre, curso, horas, fecha

def validar_badge_publico(url):
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return True, "✅ Badge válido"
        else:
            return False, f"❌ Badge no válido (status {resp.status_code})"
    except Exception as e:
        return False, f"❌ Error al validar: {e}"

# GenAI response extraction helper
def _extract_text_from_genai_response(response):
    try:
        if hasattr(response, "text") and isinstance(response.text, str):
            return response.text
        if hasattr(response, "candidates"):
            try:
                c = response.candidates
                if len(c) > 0:
                    cand = c[0]
                    if hasattr(cand, "content"):
                        cont = cand.content
                        if isinstance(cont, list) and len(cont) > 0:
                            first = cont[0]
                            if isinstance(first, dict) and "text" in first:
                                return first["text"]
                            return str(first)
                    return str(cand)
            except Exception:
                pass
        return str(response)
    except Exception:
        return str(response)

# Evaluación taller (igual que antes)
def evaluar_taller_redes_neuronales(contenido_estudiante: str):
    """
    Llama a Gemini para evaluar el taller. Devuelve (nota_taller_float_or_None, raw_text_response, parsed_json_or_None).
    - Si la respuesta incluye un JSON con la estructura esperada, lo parsea y extrae la nota.
    - Si no, intenta extraer un número del texto como fallback numérico.
    """
    if client is None:
        return None, "API de evaluación (Gemini) no configurada. Define la variable de entorno GENAI_API_KEY o configura las credenciales.", None

    prompt = f"""
Eres un profesor experto en ML y despliegue con Streamlit.
La nota máxima es: 1.50 puntos.
Objetivo: evaluar un app.py que implementa una interfaz para análisis de emociones (ver requisitos mínimos).
Devuelve SOLO un JSON con la siguiente estructura:

{{
  "nota_taller": 1.50,
  "detalle": [
    {{"criterio":"implementacion","score":0.00,"comentario":""}},
    {{"criterio":"ejecucion","score":0.00,"comentario":""}},
    {{"criterio":"preprocesamiento","score":0.00,"comentario":""}},
    {{"criterio":"evidencia","score":0.00,"comentario":""}}
  ],
  "resumen": "Texto corto con evidencias y fallos detectados"
}}

Revisa que el app.py tenga: área de texto, botón 'Predecir emoción', carga del modelo/tokenizer/label-encoder,
preprocesamiento (tokenize+pad) y uso de model.predict, y que no contenga resultados hardcodeados.
A continuación el código del estudiante:
{contenido_estudiante}
"""

    try:
        response = client.models.generate_content(model=gemini_model, contents=prompt)
        raw = _extract_text_from_genai_response(response)

        # Intentar extraer JSON completo de la respuesta
        parsed = None
        match_json = re.search(r"\{(?:.|\s)*\}", raw, flags=re.DOTALL)
        if match_json:
            try:
                parsed = json.loads(match_json.group(0))
            except Exception:
                parsed = None

        # Determinar nota (preferir parsed JSON, si no fallback numérico)
        nota_taller = None
        if isinstance(parsed, dict):
            if "nota_taller" in parsed:
                try:
                    nota_taller = float(parsed.get("nota_taller", 0.0))
                except Exception:
                    nota_taller = None
            elif "nota" in parsed:
                try:
                    nota_taller = float(parsed.get("nota", 0.0))
                except Exception:
                    nota_taller = None

        # Fallback: buscar primer número en el raw
        if nota_taller is None:
            m = re.search(r"(\d+(\.\d+)?)", raw)
            if m:
                try:
                    nota_taller = float(m.group(1))
                except Exception:
                    nota_taller = None

        if nota_taller is None:
            nota_taller = 0.0

        nota_taller = max(0.0, min(1.5, round(nota_taller, 2)))

        return nota_taller, raw, parsed

    except Exception as e:
        return None, f"Error en la llamada a la API: {e}", None

# ----------------------------
# NUEVA: Evaluación por pregunta (cada respuesta en su propio prompt)
# ----------------------------
def evaluar_respuestas_abiertas(respuestas_estudiante):
    """Evalúa cada respuesta individualmente enviando un prompt por pregunta y luego agrega la nota."""
    if client is None:
        return None, "API de evaluación (Gemini) no configurada. Define la variable de entorno GENAI_API_KEY o configura las credenciales.", None

    # Respuestas esperadas para 15 preguntas (idénticas a las definidas en la UI)
    respuestas_esperadas = {
        1: "LSTM tiene memoria a largo plazo mediante gates; RNN simple no.",
        2: "Permite retropropagar errores a través del tiempo para ajustar pesos.",
        3: "La multiplicación repetida de jacobianos reduce gradientes a casi cero.",
        4: "ht = tanh(Wx xt + Wh ht−1 + b).",
        5: "Decide qué parte de la memoria previa se olvida.",
        6: "Clasificador que distingue datos reales vs generados.",
        7: "Transformer usa solo atención y es paralelizable.",
        8: "Evitar que el modelo vea el futuro usando máscaras.",
        9: "Necesitan aprender relaciones complejas globales.",
        10: "La evolución fue de métodos basados en frecuencias (BoW, TF-IDF) hacia RNN y LSTM que capturaban dependencias secuenciales, pero con limitaciones. Los Transformers introdujeron self-attention, permitiendo paralelización, comprensión contextual profunda y modelos como BERT y GPT.",
        11: "Porque la calidad del texto de origen determina la calidad del análisis. Texto mal extraído introduce ruido, errores y pérdida de información. Técnicas como scraping, OCR y Textract son fundamentales para obtener datos limpios.",
        12: "El preprocesamiento normaliza y limpia el texto mediante tokenización, lematización, eliminación de ruido, manejo de puntuación y estandarización. Esto mejora la representatividad de los datos y el desempeño de los modelos.",
        13: "TF-IDF se basa en frecuencia y es útil para análisis simples sin semántica profunda. Los embeddings capturan relaciones semánticas y contexto, ideales para clasificación avanzada, similitud y tareas complejas.",
        14: "Los desafíos incluyen alta dimensionalidad, tokens raros, esparsidad, pérdida de contexto, desbalance de vocabulario y costos computacionales. Los embeddings también pueden requerir grandes recursos de memoria.",
        15: "Un agente puede consultar APIs internas, planificar tareas y responder automáticamente. Ejemplo: un agente que gestiona consultas de clientes verificando inventarios y fechas de entrega.",
}

    n_preg = len(respuestas_esperadas)
    per_q = round(3.0 / n_preg, 2)  # cada pregunta vale ~0.20

    results = []
    total = 0.0
    detalles_lines = []

    for k in range(1, n_preg + 1):
        expected = respuestas_esperadas[k]
        student = respuestas_estudiante.get(k, "").replace("\n", " ").strip()

        # Prompt específico por pregunta: pide SOLO un JSON con question, score, feedback
        prompt = f"""Eres un profesor experto en Ciencia de Datos y AWS.
Compara la respuesta esperada (texto) con la respuesta del estudiante y evalúa la calidad, además da una retroalimentación de Plagio. Si la detectección de plagio muestra superior al 70% la nota f[...]
La nota máxima es de 3.0 puntos, es decir debes calcular el valor con respecto a las preguntas con validez, consistentes y completitud de cada pregunta.


INSTRUCCIONES:
- Devuelve SOLO un JSON válido con EXACTAMENTE estos campos: {{
    "question": {k},
    "score": X,          # número decimal entre 0 y {per_q} con 2 decimales
    "feedback": "Texto corto de retroalimentación del examen con el analisis de plagio."
  }}
- No devuelvas texto adicional fuera del JSON.
- Sé conciso y objetivo.
- Ten en cuenta la respuesta esperada: {expected}

Respuesta del estudiante:
{student}
"""

        try:
            response = client.models.generate_content(model=gemini_model, contents=prompt)
            raw = _extract_text_from_genai_response(response).strip()
            # Extraer JSON si lo devuelve
            match_json = re.search(r"\{(?:.|\s)*\}", raw, flags=re.DOTALL)
            score = 0.0
            feedback = ""
            if match_json:
                try:
                    obj = json.loads(match_json.group(0))
                    score = float(obj.get("score", 0.0))
                    feedback = obj.get("feedback", "")
                except Exception:
                    # fallback: buscar número y tomar el resto como feedback
                    mnum = re.search(r"(\d+(\.\d+)?)", raw)
                    if mnum:
                        score = float(mnum.group(1))
                        feedback = raw
                    else:
                        score = 0.0
                        feedback = raw
            else:
                # si no hay JSON, intentar extraer un número y el texto como feedback
                mnum = re.search(r"(\d+(\.\d+)?)", raw)
                if mnum:
                    score = float(mnum.group(1))
                    feedback = raw
                else:
                    score = 0.0
                    feedback = raw

            # Asegurar límites por pregunta
            if score < 0.0:
                score = 0.0
            if score > per_q:
                score = per_q

            # Normalizar a 2 decimales
            score = round(score, 2)
            total += score
            results.append({"question": k, "score": score, "feedback": feedback})
            detalles_lines.append(f"Pregunta {k}: {score:.2f}/{per_q}\n{feedback}")

        except Exception as e:
            # En caso de error con la API, anotar 0 y el mensaje de error
            results.append({"question": k, "score": 0.0, "feedback": f"Error evaluación: {e}"})
            detalles_lines.append(f"Pregunta {k}: 0.00/{per_q}\nError evaluación: {e}")

    # Asegurar máximo 3.0 y redondeo final
    total = round(min(total, 3.0), 2)

    # Construir feedback legible
    detalles_lines.append(f"\nNota final: {total:.2f} / 3.00")
    feedback_text = "\n\n".join(detalles_lines)

    return total, feedback_text, results

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="Validador de Badges + Taller", layout="wide")
init_db()
st.title("Calificador de Badges de Credly y Evaluador de Talleres")

tabs = st.tabs([
    "📂 Cargar Badges",
    "📊 Validar Badges a DB",
    "📝 Evaluar Taller",
    "📝 Evaluar Cuestionario",
    "📈 Calificaciones"
])


# --- Reemplaza/inserta estas secciones en tu app.py ---

# ---------- Inicialización (antes de crear widgets) ----------
# Si ya tienes id_estudiante/nrc_curso en session_state, y quieres que
# aparezcan como valores por defecto en los text_input, copia esos valores
# a las keys del widget ANTES de crear los widgets.
if "id_estudiante" in st.session_state and "input_id" not in st.session_state:
    st.session_state["input_id"] = st.session_state["id_estudiante"]
if "nrc_curso" in st.session_state and "input_nrc" not in st.session_state:
    st.session_state["input_nrc"] = st.session_state["nrc_curso"]

# ----------------------------
# Pestaña 0 - Subida de archivo (usar sin reasignar input_id/input_nrc)
# ----------------------------
with tabs[0]:
    st.markdown("## Examen de Inteligencia Artificial 2025 UNAB")
    st.markdown("Sube el PDF del badge obtenido en [Credly](https://www.credly.com/) tras completar el ID y el NRC.")
    # Creamos los widgets (estos ya mantienen su valor en st.session_state["input_id"] / ["input_nrc"])
    id_input = st.text_input("ID del estudiante", key="input_id")
    nrc_input = st.text_input("NRC del curso", key="input_nrc")
    uploaded_file = st.file_uploader("Subir archivo PDF", type=["pdf"])

    if uploaded_file is not None and id_input and nrc_input:
        if st.button("Procesar y Guardar", key="procesar_guardar_pdf"):
            file_bytes = uploaded_file.read()
            texto, urls = extraer_texto_y_urls(file_bytes)
            urls_credly = [u for u in urls if u.startswith("https://www.credly.com/go/")]

            if not urls_credly:
                st.warning("⚠️ El PDF no tiene una URL de Credly para validar.")
            else:
                nombre, curso, horas, fecha = procesar_texto(texto)
                if not all([nombre, curso, horas, fecha]):
                    st.error("El PDF no tiene el formato esperado (nombre, curso, horas, fecha).")
                else:
                    for url in urls_credly:
                        valido, mensaje = validar_badge_publico(url)
                        # Guardar en la DB usando los valores ingresados (id_input, nrc_input)
                        guardar_en_db(id_input, nrc_input, nombre, curso, horas, fecha, url, valido, mensaje, None)

                        # NO reasignes las claves ligadas a widgets (input_id/input_nrc)
                        # st.session_state["input_id"] = id_input   # <-- eliminar (provoca excepción)
                        # st.session_state["input_nrc"] = nrc_input  # <-- eliminar

                        # Si quieres claves internas que uses en otras pestañas, sí crea otras claves:
                        st.session_state["id_estudiante"] = id_input
                        st.session_state["nrc_curso"] = nrc_input

                        st.success(f"Guardado en BD: {id_input}, {nrc_input}, {nombre}, {curso}, {horas}, {fecha}, {url}")
                        st.info(mensaje)

# Pestaña 2 - Ver registros
with tabs[1]:
    st.header("Validar que los Badge fueron Registrados")
    df = obtener_registros()
    if df.empty:
        st.info("No hay registros guardados todavía.")
    else:
        if st.session_state.get("input_id"):
            df = df[df["id"] == st.session_state["input_id"].strip()]
        if st.session_state.get("input_nrc"):
            df = df[df["nrc"] == st.session_state["input_nrc"].strip()]
        st.dataframe(df, width='stretch')

# ----------------------------
# Pestaña 3 - Evaluar Taller (actualizada para mostrar Resumen + Detalle)
# ----------------------------
with tabs[2]:
    st.header("Evaluar Taller de Redes Neuronales")
    st.markdown(
    """
    📥 [Abrir un taller de deployment (Examen_ID_NRC.ipynb)](https://colab.research.google.com/drive/163BFNTDSw6w99uOFpxfKCyjiqaIC89FL?usp=sharing)
    """,  unsafe_allow_html=True)

    archivos_taller = st.file_uploader(
        "Subir solución en Python (.py o .ipynb) - sube un solo archivo con la solución del taller o varios archivos si el taller tiene múltiples partes.", 
        type=["py", "ipynb"],
        accept_multiple_files=True,
        key="uploader_taller"
    )

    comentarios_taller = st.text_area(
        "Comentarios / Observaciones (opcional)", 
        height=120, 
        key="comentarios_taller"
    )

    if "taller_results" not in st.session_state:
        st.session_state["taller_results"] = []  # lista de dicts: {filename, nota, feedback_raw, feedback_json, timestamp}

    current_id = st.session_state.get("input_id", "").strip()
    current_nrc = st.session_state.get("input_nrc", "").strip()

    if not current_id or not current_nrc:
        st.warning("Debes ingresar ID y NRC en la pestaña 'Cargar PDF' antes de evaluar talleres.")
    else:
        if archivos_taller:
            for idx, archivo in enumerate(archivos_taller):
                st.markdown(f"**Archivo {idx+1}:** {archivo.name}")
                col1, col2 = st.columns([1, 3])

                with col1:
                    if st.button(f"Evaluar {archivo.name}", key=f"eval_file_{idx}"):
                        try:
                            raw_bytes = archivo.read()
                            nombre = archivo.name.lower()
                            if nombre.endswith(".py"):
                                contenido_estudiante = raw_bytes.decode("utf-8", errors="ignore")
                            elif nombre.endswith(".ipynb"):
                                try:
                                    nb_json = json.loads(raw_bytes.decode("utf-8"))
                                except Exception:
                                    nb_json = json.loads(raw_bytes.decode("latin-1"))
                                celdas = []
                                for celda in nb_json.get("cells", []):
                                    if celda.get("cell_type") == "code":
                                        celdas.append("".join(celda.get("source", [])))
                                contenido_estudiante = "\n".join(celdas)
                            else:
                                st.error("⚠️ Formato de archivo no soportado.")
                                contenido_estudiante = None

                            if not contenido_estudiante:
                                st.error("No se pudo extraer el contenido del archivo.")
                                continue

                            # Evaluar (Gemini)
                            nota_taller, feedback_raw, feedback_json = evaluar_taller_redes_neuronales(contenido_estudiante)

                            if nota_taller is None:
                                st.warning("⚠️ No se pudo calcular la nota automáticamente. Revisa el feedback.")
                                # guardar registro con nota None -> 0.0
                                nota_to_save = 0.0
                            else:
                                nota_to_save = float(nota_taller)

                            # Guardar en BD y en session_state
                            guardar_resultado_taller(
                                current_id,
                                current_nrc,
                                nota_to_save,
                                feedback_raw,
                                comentarios_taller or ""
                            )

                            registro = {
                                "filename": archivo.name,
                                "nota": nota_to_save,
                                "feedback_raw": feedback_raw,
                                "feedback_json": feedback_json,
                                "timestamp": datetime.now(timezone.utc).isoformat()
                            }
                            st.session_state["taller_results"].append(registro)

                            st.success(f"✅ Nota del taller guardada para {archivo.name}: {nota_to_save} / 1.5")
                            st.session_state["nota_taller"] = nota_to_save

                            # Mostrar feedback inmediato: abrimos dos subtabs Resumen / Detalle
                            st.markdown("### Resultado de la evaluación")
                            sub1, sub2 = st.tabs(["Resumen", "Detalle"])
                            with sub1:
                                # Resumen: si hay JSON mostrar campo resumen, sino mostrar primera línea raw
                                if isinstance(feedback_json, dict) and feedback_json.get("resumen"):
                                    st.write(feedback_json.get("resumen"))
                                else:
                                    # mostrar un fragmento del raw como resumen
                                    preview = feedback_raw.strip().splitlines()
                                    st.write(preview[0] if preview else "Sin resumen.")
                                st.write(f"Nota final: {nota_to_save} / 1.5")

                            with sub2:
                                # Detalle: si feedback_json tiene 'detalle' iterable, mostrarlo formateado
                                if isinstance(feedback_json, dict) and isinstance(feedback_json.get("detalle"), list):
                                    st.markdown("**Desglose por criterios**")
                                    for item in feedback_json.get("detalle", []):
                                        criterio = item.get("criterio", "criterio")
                                        score = item.get("score", 0.0)
                                        comentario = item.get("comentario", item.get("feedback", ""))
                                        st.write(f"- {criterio}: {score:.2f} — {comentario}")
                                else:
                                    # Si no hay detalle estructurado, mostramos raw con resaltado
                                    st.markdown("**Feedback (raw)**")
                                    st.code(feedback_raw, language="")
                                    # Intentar buscar líneas que parezcan puntuaciones
                                    st.markdown("**Fragmentos relevantes**")
                                    # Mostrar fragmentos donde aparezcan palabras clave
                                    keywords = ["implementacion", "ejecucion", "preprocesamiento", "evidencia", "nota"]
                                    for kw in keywords:
                                        for line in (feedback_raw or "").splitlines():
                                            if kw.lower() in line.lower():
                                                st.write(f"- {line.strip()}")

                        except Exception as e:
                            st.error(f"❌ Error al procesar el archivo: {e}")

                with col2:
                    st.write("")  # espacio o mostrar preview si quieres

        else:
            st.info("No hay archivos subidos para evaluar. Sube uno o varios y pulsa 'Evaluar' en el archivo deseado.")

        # Mostrar resumen de todas las evaluaciones de esta sesión (tabla y selector)
        st.markdown("---")
        st.subheader("Historial de evaluaciones en esta sesión")
        if not st.session_state["taller_results"]:
            st.info("Aún no se ha evaluado ningún archivo en esta sesión.")
        else:
            df_resumen = pd.DataFrame(st.session_state["taller_results"])
            display_df = df_resumen[["filename", "nota", "timestamp"]].sort_values(by="timestamp", ascending=False).reset_index(drop=True)
            st.dataframe(display_df, width='stretch')

            seleccion = st.selectbox("Selecciona un archivo para ver detalle:", display_df["filename"].tolist(), key="taller_res_selector")
            registro = next((r for r in st.session_state["taller_results"] if r["filename"] == seleccion), None)
            if registro:
                st.markdown("### Detalle seleccionado")
                st.write(f"- Archivo: {registro['filename']}")
                st.write(f"- Nota: {registro['nota']} / 1.5")
                st.write(f"- Fecha: {registro['timestamp']}")

                # Mostrar resumen y detalle como tabs
                rsub1, rsub2 = st.tabs(["Resumen", "Detalle"])
                with rsub1:
                    if isinstance(registro.get("feedback_json"), dict) and registro["feedback_json"].get("resumen"):
                        st.write(registro["feedback_json"]["resumen"])
                    else:
                        # fallback: mostrar primera línea del raw
                        first_line = (registro.get("feedback_raw") or "").strip().splitlines()
                        st.write(first_line[0] if first_line else "Sin resumen.")

                with rsub2:
                    if isinstance(registro.get("feedback_json"), dict) and isinstance(registro["feedback_json"].get("detalle"), list):
                        for item in registro["feedback_json"]["detalle"]:
                            criterio = item.get("criterio", "criterio")
                            score = item.get("score", 0.0)
                            comentario = item.get("comentario", item.get("feedback", ""))
                            st.write(f"- {criterio}: {score:.2f} — {comentario}")
                    else:
                        st.code(registro.get("feedback_raw", ""), language="")

                # Descargar feedback raw
                st.download_button(
                    label="📥 Descargar feedback (raw) como texto",
                    data=registro.get("feedback_raw", ""),
                    file_name=f"feedback_{registro['filename']}.txt",
                    mime="text/plain"
                )

# Pestaña 4 - Evaluar Cuestionario (ahora usa evaluación por pregunta)
with tabs[3]:
    st.header("Cuestionario de Preguntas Abiertas (Responde en 1-3 líneas cada una)")
    preguntas_abiertas = [
        "Explique brevemente la diferencia conceptual entre una RNN clásica-vainilla y una LSTM.",
        "En una RNN simple, ¿qué problema resuelve el cálculo de la derivada a lo largo del tiempo (BPTT)?",
        "¿Por qué las RNN sufren vanishing gradient? Explique en 2-3 líneas.",
        "Dé la ecuación del estado oculto ht en una RNN simple.",
        "En LSTM, describa brevemente el propósito de la puerta de olvido (forget gate).",
        "¿Qué es un discriminador en una GAN?",
        "¿Qué diferencia hay entre seq2seq con atención y Transformer?",
        "¿Qué es el entrenamiento causal? (causal masking)",
        "¿Por qué los Transformers requieren grandes datasets?",
        "Explica cómo ha evolucionado la arquitectura de NLP desde métodos tradicionales hasta modelos Transformer.",
        "¿Por qué la obtención de texto es una etapa crítica en el pipeline de NLP?",
        "Describe el propósito del preprocesamiento de texto en un proyecto de NLP.",
        "Compara brevemente TF-IDF con Word Embeddings y describe en qué casos es preferible cada enfoque.",
        "¿Qué desafíos suelen aparecer al vectorizar texto para NLP?",
        "Describe un escenario donde el uso de agentes (AI agents) en Amazon Bedrock mejore un proceso empresarial."
    ]

    respuestas_usuario = {}
    for i, pregunta in enumerate(preguntas_abiertas, start=1):
        st.markdown(f"**Pregunta {i}:** {pregunta}")
        respuesta = st.text_area(f"Respuesta {i}", key=f"resp_{i}", height=120)
        respuestas_usuario[i] = respuesta or ""

    if st.button("Evaluar Cuestionario", key="boton_evaluar_cuestionario"):
        if st.session_state.get("input_id") and st.session_state.get("input_nrc"):
            with st.spinner("Evaluando respuestas pregunta por pregunta..."):
                nota_cuestionario, feedback_text, parsed = evaluar_respuestas_abiertas(respuestas_usuario)

            if nota_cuestionario is not None:
                st.success(f"✅ Nota cuestionario: {nota_cuestionario:.2f} / 3.0")
                guardar_respuestas_cuestionario(
                    st.session_state["input_id"].strip(),
                    st.session_state["input_nrc"].strip(),
                    respuestas_usuario,
                    nota_cuestionario,
                    feedback_text
                )
                st.session_state["nota_cuestionario"] = nota_cuestionario
            else:
                st.warning(f"⚠️ No se pudo obtener una nota numérica del modelo. {feedback_text}")

            st.subheader("Retroalimentación del cuestionario")
            st.markdown(feedback_text)
        else:
            st.error("⚠️ Debes haber ingresado ID y NRC en la pestaña 'Cargar PDF' antes de evaluar.")

# Pestaña 5 - Calificaciones finales
with tabs[4]:
    st.header("📈 Resumen de Calificaciones")
    df_badges = obtener_calificaciones_badges()
    df_taller = obtener_ultimo_taller()
    df_cuestionario = obtener_ultimo_cuestionario()

    df_merge = (
        df_badges
        .merge(df_taller, on=["id_estudiante", "nrc_curso"], how="outer")
        .merge(df_cuestionario, on=["id_estudiante", "nrc_curso"], how="outer")
    )

    for col in ["calificacion_badge", "calificacion_taller", "calificacion_cuestionario"]:
        if col not in df_merge.columns:
            df_merge[col] = 0.0
        else:
            # Forzamos a numérico (convierte valores no numéricos a NaN), luego rellenamos
            df_merge[col] = pd.to_numeric(df_merge[col], errors="coerce").fillna(0.0)

    df_merge["nota_total"] = (
        df_merge["calificacion_badge"] +
        df_merge["calificacion_taller"] +
        df_merge["calificacion_cuestionario"]
    )

    id_filter = st.session_state.get("input_id", "").strip()
    nrc_filter = st.session_state.get("input_nrc", "").strip()

    if id_filter:
        df_merge = df_merge[df_merge["id_estudiante"] == id_filter]

    if nrc_filter:
        df_merge = df_merge[df_merge["nrc_curso"] == nrc_filter]

    if df_merge.empty:
        st.info("No hay calificaciones registradas aún para este estudiante y curso.")
    else:
        st.subheader("Resumen de calificaciones")
        st.dataframe(
            df_merge.reset_index(drop=True)[[
                "id_estudiante",
                "nrc_curso",
                "calificacion_badge",
                "calificacion_taller",
                "calificacion_cuestionario",
                "nota_total"
            ]],
            width='stretch'
        )
        st.info("⚖️ Nota máxima posible = 5.0 (Badges 0.5 + Taller 1.5 + Cuestionario 3.0)")

    try:
        with open(DB_PATH, "rb") as f:
            st.download_button(
                label="📥 Descargar base de datos SQLite",
                data=f,
                file_name="badges.db",
                mime="application/octet-stream"
            )
    except Exception as e:
        st.error(f"No se pudo preparar la descarga de la DB: {e}")

st.write("")  # espacio final
