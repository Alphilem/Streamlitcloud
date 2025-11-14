import streamlit as st
from openai import OpenAI
import pandas as pd

# ✅ Obtener API key desde los secretos de Streamlit
api_key = st.secrets["OPENAI_API_KEY"]

# Inicializar cliente OpenAI
client = OpenAI(api_key=api_key)

# Importar dataset
df = pd.read_csv('spotify-2023.csv')

# Usar solo las primeras 100 filas
df_subset = df.head(150)

# Convertir a texto
df_string = df_subset.to_string()

# Título
st.title("🎵 Asistente de datos de Spotify 2023")

# Campo de texto para la pregunta
user_input = st.text_input("Escribe tu pregunta sobre estadisticas de Spotify en 2023:")

# Cuando el usuario escribe una pregunta
if user_input:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un asistente experto en las estadisticas de Spotify en 2023. "
                    "Usa ÚNICAMENTE la información del siguiente dataset para responder preguntas. "
                    "Si la pregunta no está relacionada con los datos, responde con: "
                    "'Lo siento, no fui entrenada para responder preguntas sobre la temática que me preguntaste.'\n\n"
                    "Aquí están los primeros 150 registros del dataset:\n" + df_string
                )
            },
            {"role": "user", "content": user_input}
        ]
    )

    # Mostrar respuesta
    answer = response.choices[0].message.content
    st.subheader("Respuesta:")
    st.write(answer)








