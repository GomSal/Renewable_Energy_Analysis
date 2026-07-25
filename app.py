import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq

# Configuración de la página
st.set_page_config(page_title="Dashboard Analítico & Asistente IA", layout="wide")

# 1. Carga de datos
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("energia_renovable.csv")
        if 'Fecha_Entrada_Operacion' in df.columns:
            df['Fecha_Entrada_Operacion'] = pd.to_datetime(df['Fecha_Entrada_Operacion'])
        return df
    except Exception as e:
        return pd.DataFrame() # Retorna dataframe vacío si no encuentra el archivo

df = load_data()

# 2. Sidebar
st.sidebar.title("Configuración")
st.sidebar.image("https://via.placeholder.com/150", caption="Logo de la Empresa")

# Input para API Key de Groq
api_key = st.sidebar.text_input("🔑 Groq API Key", type="password", help="Ingresa tu clave de API de Groq para usar el asistente LLaMA 3.3")

st.sidebar.header("Filtros Globales")
if not df.empty:
    categorias_seleccionadas = st.sidebar.multiselect(
        "Selecciona Tecnología",
        options=df["Tecnologia"].unique(),
        default=df["Tecnologia"].unique()
    )
    
    min_date = df["Fecha_Entrada_Operacion"].min()
    max_date = df["Fecha_Entrada_Operacion"].max()
    rango_fechas = st.sidebar.date_input("Rango de Fechas", [min_date, max_date])
    
    if len(rango_fechas) == 2:
        start_date = pd.to_datetime(rango_fechas[0])
        end_date = pd.to_datetime(rango_fechas[1])
        df_filtrado = df[(df["Tecnologia"].isin(categorias_seleccionadas)) & 
                         (df["Fecha_Entrada_Operacion"] >= start_date) & 
                         (df["Fecha_Entrada_Operacion"] <= end_date)]
    else:
        df_filtrado = df.copy()
else:
    st.error("No se pudo cargar el archivo 'energia_renovable.csv'. Asegúrate de que esté en la misma carpeta.")
    st.stop()

st.title("⚡ Dashboard Analítico con Asistente IA")

# 3. Pestañas (Tabs)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 EDA", 
    "📈 Variables Cuantitativas", 
    "📊 Variables Cualitativas", 
    "🎨 Reporte Avanzado Visual", 
    "🤖 Asistente de Datos (Groq)"
])

# --- TAB 1: EDA ---
with tab1:
    st.header("Análisis Exploratorio de Datos (EDA)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Filas", df_filtrado.shape[0])
    col2.metric("Total Columnas", df_filtrado.shape[1])
    col3.metric("Datos Faltantes", df_filtrado.isnull().sum().sum())
    
    st.subheader("Vista Previa de los Datos")
    st.dataframe(df_filtrado.head(10), use_container_width=True)
    
    st.subheader("Resumen Estadístico")
    st.dataframe(df_filtrado.describe(), use_container_width=True)

# --- TAB 2: Variables Cuantitativas ---
with tab2:
    st.header("Análisis de Variables Cuantitativas")
    num_cols = df_filtrado.select_dtypes(include=np.number).columns.tolist()
    
    if num_cols:
        col_sel = st.selectbox("Selecciona una variable numérica:", num_cols)
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig_hist = px.histogram(df_filtrado, x=col_sel, title=f"Distribución de {col_sel}", marginal="box")
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col_chart2:
            st.write(f"**Estadísticas descriptivas para {col_sel}:**")
            st.write(df_filtrado[col_sel].describe())
    else:
        st.info("No hay variables cuantitativas en el dataset.")

# --- TAB 3: Variables Cualitativas ---
with tab3:
    st.header("Análisis de Variables Cualitativas")
    cat_cols = df_filtrado.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    if cat_cols:
        col_sel_cat = st.selectbox("Selecciona una variable categórica:", cat_cols)
        
        fig_bar = px.bar(df_filtrado[col_sel_cat].value_counts().reset_index(), 
                         x=col_sel_cat, y='count', 
                         title=f"Conteo de categorías para {col_sel_cat}",
                         color=col_sel_cat)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No hay variables cualitativas en el dataset.")

# --- TAB 4: Reporte Avanzado Visual ---
with tab4:
    st.header("Reporte Visual Avanzado")
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.subheader("Evolución de Inversión en el Tiempo")
        df_fechas = df_filtrado.groupby(df_filtrado["Fecha_Entrada_Operacion"].dt.to_period("M"))["Inversion_Inicial_MUSD"].sum().reset_index()
        df_fechas["Fecha_Entrada_Operacion"] = df_fechas["Fecha_Entrada_Operacion"].dt.to_timestamp()
        fig_line = px.line(df_fechas, x="Fecha_Entrada_Operacion", y="Inversion_Inicial_MUSD", markers=True)
        st.plotly_chart(fig_line, use_container_width=True)
        
    with col_v2:
        st.subheader("Generación vs Inversión por Tecnología")
        fig_scatter = px.scatter(df_filtrado, x="Inversion_Inicial_MUSD", y="Generacion_Diaria_MWh", 
                                 color="Tecnologia", size="Capacidad_Instalada_MW", hover_name="ID_Proyecto")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    st.subheader("Matriz de Correlación")
    numeric_df = df_filtrado.select_dtypes(include=np.number)
    if not numeric_df.empty:
        fig_corr, ax_corr = plt.subplots(figsize=(8, 4))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)

# --- TAB 5: Asistente LLaMA 3.3 (Groq) ---
with tab5:
    st.header("Asistente Analítico de Datos")
    st.markdown("Haz preguntas sobre el conjunto de datos actual. El modelo **LLaMA 3.3 70B** responderá basándose en un resumen del dataset filtrado.")
    
    if api_key:
        try:
            client = Groq(api_key=api_key)
            
            # Construir contexto para el modelo basado en los datos filtrados
            contexto_datos = (
                f"El dataset contiene {df_filtrado.shape[0]} filas y las columnas: {', '.join(df_filtrado.columns)}. "
                f"Resumen numérico:\n{df_filtrado.describe().to_string()}\n"
            )
            
            # Inicializar historial de chat en session_state
            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {"role": "system", "content": f"Eres un analista de datos experto. Ayuda al usuario a entender sus datos. Utiliza este resumen estadístico como contexto:\n{contexto_datos}"}
                ]
            
            # Mostrar historial de chat (ocultando el system prompt)
            for msg in st.session_state.messages:
                if msg["role"] != "system":
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
            
            # Input de usuario
            if prompt := st.chat_input("Ej: ¿Cuál es la tecnología con mayor inversión promedio?"):
                # Agregar mensaje de usuario
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Llamada a Groq
                with st.spinner("LLaMA 3.3 está analizando..."):
                    chat_completion = client.chat.completions.create(
                        messages=st.session_state.messages,
                        model="llama-3.3-70b-versatile",
                        temperature=0.3,
                    )
                    
                    respuesta = chat_completion.choices[0].message.content
                    
                    # Agregar y mostrar respuesta
                    st.session_state.messages.append({"role": "assistant", "content": respuesta})
                    with st.chat_message("assistant"):
                        st.markdown(respuesta)
        except Exception as e:
            st.error(f"Error de conexión con Groq: {e}. Verifica tu API Key.")
    else:
        st.info("👈 Por favor, ingresa tu API Key de Groq en la barra lateral para habilitar el chat.")
