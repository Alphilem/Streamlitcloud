import sys
import subprocess
import importlib

def install_required_packages():
    """Automatically install required packages if missing"""
    required_packages = [
        'streamlit',
        'openai', 
        'pandas',
        'numpy'
    ]
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} is already installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install packages before importing
install_required_packages()


import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from datetime import datetime
import io

spotify_df=pd.read_csv("Streamlitcloud/spotify-2023.csv")
spotify_df=spotify_df.sort_values("streams",ascending=False)
spotify_df=spotify_df.head(150)

# Configuración de la página
st.set_page_config(
    page_title="ChatBot Spotify - Análisis de Datos",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🎵 ChatBot Especializado - Análisis Spotify")
st.markdown("""
**Instrucciones:** Este chatbot está especializado únicamente en analizar la base de datos de Spotify. 
Solo responderá preguntas relacionadas con los datos cargados.
""")

# Inicialización del estado de la sesión
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "data_context" not in st.session_state:
        st.session_state.data_context = ""
    if "spotify_df" not in st.session_state:
        # Aquí cargas tu DataFrame existente
        st.session_state.spotify_df = spotify_df  # Tu DataFrame ya cargado
    if "api_key_configured" not in st.session_state:
        st.session_state.api_key_configured = False
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

initialize_session_state()

# Función para crear el contexto de datos
def create_data_context(df):
    """Crea un contexto comprehensivo sobre el DataFrame para la IA"""
    
    context = f"""
    CONTEXTO DEL DATASET DE SPOTIFY - ESTOS SON LOS ÚNICOS DATOS QUE PUEDO ANALIZAR:

    RESUMEN DEL DATASET:
    - Número de filas: {len(df)}
    - Número de columnas: {len(df.columns)}
    - Columnas disponibles: {', '.join(df.columns.tolist())}

    INFORMACIÓN DETALLADA DE COLUMNAS:
    """
    
    # Agregar información de cada columna
    for col in df.columns:
        context += f"\n- **{col}**: "
        context += f"Tipo: {df[col].dtype}, "
        
        if pd.api.types.is_numeric_dtype(df[col]):
            context += f"Rango: {df[col].min():.2f} a {df[col].max():.2f}, "
            context += f"Promedio: {df[col].mean():.2f}"
        else:
            unique_vals = df[col].nunique()
            context += f"Valores únicos: {unique_vals}"
            if unique_vals <= 15:
                sample_vals = df[col].unique()[:8]
                context += f", Ejemplos: {', '.join(map(str, sample_vals))}"
    
    # Agregar estadísticas básicas para columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        context += f"\n\nESTADÍSTICAS NUMÉRICAS:\n{df[numeric_cols].describe().to_string()}"
    
    context += f"""

    MUESTRA DE DATOS (primeras 5 filas):
    {df.head().to_string()}

    REGLAS IMPORTANTES:
    1. SOLO puedo responder preguntas sobre este dataset específico de Spotify
    2. Si me preguntan sobre otros temas, debo responder: "Soy un chatbot especializado únicamente en el análisis de datos de Spotify. No tengo información sobre otros temas."
    3. Debo ser preciso y basarme únicamente en los datos proporcionados
    4. Si no puedo encontrar información en los datos, debo decirlo claramente
    """
    
    return context

# Función para verificar si la pregunta está relacionada con los datos
def is_data_related_question(question, df):
    """Verifica si la pregunta está relacionada con el DataFrame"""
    
    # Palabras clave generales de datos
    data_keywords = [
        'spotify', 'música', 'canción', 'artista', 'álbum', 'género',
        'data', 'datos', 'dataset', 'base de datos', 'tabla',
        'columna', 'fila', 'estadística', 'análisis', 'analizar',
        'qué', 'cuántos', 'cuántas', 'mostrar', 'decir',
        'máximo', 'mínimo', 'promedio', 'media', 'mediana',
        'contar', 'suma', 'distribución', 'tendencia', 'patrón',
        'popularidad', 'bailable', 'energía', 'acústica',
        'duración', 'tempo', 'año', 'género'
    ]
    
    # Agregar nombres de columnas como palabras clave
    if df is not None:
        data_keywords.extend([col.lower() for col in df.columns])
    
    question_lower = question.lower()
    
    # Verificar si la pregunta contiene palabras clave relacionadas con datos
    return any(keyword in question_lower for keyword in data_keywords)

# Función para obtener respuesta especializada
def get_specialized_response(user_question, df, data_context, api_key):
    """Obtiene respuesta de OpenAI, especializada solo para el dataset"""
    
    if not is_data_related_question(user_question, df):
        return "Soy un chatbot especializado únicamente en el análisis de datos de Spotify. No tengo información sobre otros temas."
    
    try:
        client = OpenAI(api_key=api_key)
        
        system_prompt = f"""
        Eres un asistente especializado en análisis de datos de música de Spotify. 
        Tu ÚNICO conocimiento es sobre el siguiente dataset:

        {data_context}

        REGLAS ESTRICTAS:
        1. SOLO responde preguntas sobre este dataset específico de Spotify
        2. Si te preguntan sobre cualquier otro tema, responde: "Soy un chatbot especializado únicamente en el análisis de datos de Spotify. No tengo información sobre otros temas."
        3. Sé preciso y factual basándote únicamente en los datos proporcionados
        4. Si no puedes responder con la información del dataset, di "No puedo encontrar esa información en los datos de Spotify disponibles"
        5. No inventes información más allá de lo que está en el dataset
        6. Responde en el mismo idioma en que te hacen la pregunta
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            max_tokens=400,
            temperature=0.1  # Baja temperatura para respuestas más factuales
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error al conectar con el servicio de IA: {str(e)}"

# Barra lateral para configuración
with st.sidebar:
    st.header("🔑 Configuración de API")
    
    # Solicitar API Key
    api_key_input = st.text_input(
        "Ingresa tu API Key de OpenAI:",
        type="password",
        placeholder="sk-...",
        help="Puedes obtener tu API key en https://platform.openai.com/api-keys"
    )
    
    if api_key_input:
        st.session_state.api_key = api_key_input
        st.session_state.api_key_configured = True
        st.success("✅ API Key configurada correctamente")
    else:
        st.session_state.api_key_configured = False
        st.warning("⚠️ Ingresa tu API Key para usar el chatbot")
    
    st.header("📊 Información del Dataset")
    if st.session_state.spotify_df is not None:
        st.metric("Total de Canciones", len(st.session_state.spotify_df))
        st.metric("Columnas Disponibles", len(st.session_state.spotify_df.columns))
        
        # Mostrar columnas disponibles
        with st.expander("Ver Columnas Disponibles"):
            st.write(list(st.session_state.spotify_df.columns))
    
    # Controles del chat
    st.header("💬 Control del Chat")
    if st.button("🧹 Limpiar Historial de Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
    
    # Estadísticas
    st.header("📈 Estadísticas del Chat")
    st.metric("Mensajes Totales", len(st.session_state.chat_history))

# Inicializar el contexto de datos si no está cargado
if st.session_state.spotify_df is not None and not st.session_state.data_context:
    st.session_state.data_context = create_data_context(st.session_state.spotify_df)

# Mostrar información del dataset
if st.session_state.spotify_df is not None:
    with st.expander("📋 Vista Previa del Dataset Spotify", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎵 Canciones", len(st.session_state.spotify_df))
        with col2:
            st.metric("📊 Columnas", len(st.session_state.spotify_df.columns))
        with col3:
            numeric_cols = len(st.session_state.spotify_df.select_dtypes(include=[np.number]).columns)
            st.metric("🔢 Columnas Numéricas", numeric_cols)
        
        # Mostrar DataFrame
        st.dataframe(st.session_state.spotify_df.head(10), use_container_width=True)
        
        # Mostrar estadísticas rápidas
        st.subheader("📊 Estadísticas Rápidas")
        if numeric_cols > 0:
            st.dataframe(st.session_state.spotify_df.describe(), use_container_width=True)

# Mostrar mensajes del chat
st.header("💬 Conversación con el ChatBot")
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "timestamp" in message:
                st.caption(f"🕒 {message['timestamp']}")

# Entrada de chat
st.markdown("---")

# Verificar si la API key está configurada antes de permitir el chat
if not st.session_state.api_key_configured:
    st.warning("🔑 Por favor, ingresa tu API Key de OpenAI en la barra lateral para comenzar a chatear.")
    user_input = st.chat_input("Ingresa tu API Key primero...", disabled=True)
else:
    user_input = st.chat_input("Haz una pregunta sobre los datos de Spotify...")

# Procesar entrada del usuario
if user_input and st.session_state.spotify_df is not None and st.session_state.api_key_configured:
    # Agregar mensaje del usuario
    timestamp = datetime.now().strftime("%H:%M:%S")
    user_message = {
        "role": "user", 
        "content": user_input, 
        "timestamp": timestamp
    }
    st.session_state.messages.append(user_message)
    st.session_state.chat_history.append(user_message)
    
    # Obtener respuesta de la IA
    with st.spinner("🔍 Analizando datos de Spotify..."):
        response = get_specialized_response(
            user_input, 
            st.session_state.spotify_df, 
            st.session_state.data_context,
            st.session_state.api_key
        )
    
    # Agregar respuesta del asistente
    bot_timestamp = datetime.now().strftime("%H:%M:%S")
    bot_message = {
        "role": "assistant", 
        "content": response,
        "timestamp": bot_timestamp
    }
    st.session_state.messages.append(bot_message)
    st.session_state.chat_history.append(bot_message)
    
    st.rerun()

# Mensajes de advertencia
if st.session_state.spotify_df is None:
    st.error("⚠️ No se ha cargado el DataFrame de Spotify.")

# Pie de página
st.markdown("---")

st.caption("🎵 ChatBot Especializado en Spotify - Desarrollado con Streamlit y OpenAI")





