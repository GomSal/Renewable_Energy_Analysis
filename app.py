import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq

# Configuración de la página
st.set_page_config(page_title="Dashboard Analítico & Asistente IA", layout="wide")

# 2. Sidebar - Controles iniciales
st.sidebar.title("Configuración")
st.sidebar.image("https://via.placeholder.com/150", caption="Logo de la Empresa")

# Input para cargar archivo
uploaded_file = st.sidebar.file_uploader("📂 Sube tu dataset (CSV)", type=["csv"], help="Carga tu dataset de energías renovables o cualquier otro archivo con estructura similar.")

# Input para API Key de Groq
api_key = st.sidebar.text_input("🔑 Groq API Key", type="password", help="Ingresa tu clave de API de Groq para usar el asistente LLaMA 3.3")

# 1. Carga de datos dinámica
@st.cache_data
def load_data(file):
    if file is not None:
        try:
            df = pd.read_csv(file)
            # Intentar convertir la columna de fecha si existe en este nuevo dataset
            if 'Fecha_Entrada_Operacion' in df.columns:
                df['Fecha_Entrada_Operacion'] = pd.to_datetime(df['Fecha_Entrada_Operacion'])
            return df
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

df = load_data(uploaded_file)

st.title("⚡ Dashboard Analítico de Datos con IA")

# Verificamos si se ha subido un archivo
if df.empty:
    st.info("👈 Por favor, carga un archivo CSV en la barra lateral para comenzar el análisis interactivo.")
else:
    st.sidebar.header("Filtros Globales")
    df_filtrado = df.copy()
    
    # Filtro dinámico por Tecnología (si la columna existe en el dataset subido)
    if "Tecnologia" in df.columns:
        categorias_seleccionadas = st.sidebar.multiselect(
            "Selecciona Tecnología",
            options=df["Tecnologia"].unique(),
            default=df["Tecnologia"].unique()
        )
        df_filtrado = df_filtrado[df_filtrado["Tecnologia"].isin(categorias_seleccionadas)]
        
    # Filtro dinámico por Fecha (si la columna existe)
    if "Fecha_Entrada_Operacion" in df.columns:
        min_date = df["Fecha_Entrada_Operacion"].min()
        max_date = df["Fecha_Entrada_Operacion"].max()
        rango_fechas = st.sidebar.date_input("Rango de Fechas", [min_date, max_date])
        
        if len(rango_fechas) == 2:
            start_date = pd.to_datetime(rango_fechas[0])
            end_date = pd.to_datetime(rango_fechas[1])
            df_filtrado = df_filtrado[(df_filtrado["Fecha_Entrada_Operacion"] >= start_date) & 
                                      (df_filtrado["Fecha_Entrada_Operacion"] <= end_date)]

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
            st.info("No hay variables cuantitativas en el dataset cargado.")

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
            st.info("No hay variables cualitativas en el dataset cargado.")

    # --- TAB 4: Reporte Avanzado Visual ---
    with tab4:
        st.header("Reporte Visual Avanzado")
        
        # Lógica de adaptación: verificamos si las columnas específicas de energías renovables existen
        has_energy_cols = all(col in df_filtrado.columns for col in ["Fecha_Entrada_Operacion", "Inversion_Inicial_MUSD", "Tecnologia", "Generacion_Diaria_MWh", "Capacidad_Instalada_MW"])
        
        if has_energy_cols:
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.subheader("Evolución de Inversión en el Tiempo")
                df_fechas = df_filtrado.groupby(df_filtrado["Fecha_Entrada_Operacion"].dt.to_period("M"))["Inversion_Inicial_MUSD"].sum().reset_index()
                df_fechas["Fecha_Entrada_Operacion"] = df_fechas["Fecha_Entrada_Operacion"].dt.to_timestamp()
                fig_line = px.line(df_fechas, x="Fecha_Entrada_Operacion", y="Inversion_Inicial_MUSD", markers=True)
                st.plotly_chart(fig_line, use_container_width=True)
                
            with col_v2:
                st.subheader("Generación vs Inversión por Tecnología")
                # Incluir ID_Proyecto en el tooltip si la columna está disponible
                h_name = "ID_Proyecto" if "ID_Proyecto" in df_filtrado.columns else None
                fig_scatter = px.scatter(df_filtrado, x="Inversion_Inicial_MUSD", y="Generacion_Diaria_MWh", 
                                         color="Tecnologia", size="Capacidad_Instalada_MW", hover_name=h_name)
                st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("El dataset cargado no contiene las columnas estándar esperadas (Inversion_Inicial_MUSD, Tecnologia, etc.). Mostrando un gráfico de dispersión genérico.")
            # Respaldo seguro para cualquier otro set de datos
            if len(num_cols) >= 2:
                st.subheader(f"Dispersión de variables: {num_cols[0]} vs {num_cols[1]}")
                fig_scatter_gen = px.scatter(df_filtrado, x=num_cols[0], y=num_cols[1])
                st.plotly_chart(fig_scatter_gen, use_container_width=True)
        
        st.subheader("Matriz de Correlación")
        numeric_df = df_filtrado.select_dtypes(include=np.number)
        if not numeric_df.empty and len(numeric_df.columns) > 1:
            fig_corr, ax_corr = plt.subplots(figsize=(8, 4))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
            st.pyplot(fig_corr)
        else:
            st.info("No hay suficientes variables numéricas para calcular una correlación.")

    # --- TAB 5: Asistente LLaMA 3.3 (Groq) ---
    with tab5:
        st.header("Asistente Analítico de Datos")
        st.markdown("Haz preguntas sobre el conjunto de datos cargado. El modelo **LLaMA 3.3 70B** responderá basándose en un resumen del dataset filtrado.")
        
        if api_key:
            try:
                client = Groq(api_key=api_key)
                
                # Dinamismo: El contexto se actualiza automáticamente según el dataset subido y filtrado
                contexto_datos = (
                    f"El dataset actual contiene {df_filtrado.shape[0]} filas y las columnas: {', '.join(df_filtrado.columns)}. "
                    f"Resumen numérico:
{df_filtrado.describe().to_string()}
"
                )
                
                # Inicializar historial
                if "messages" not in st.session_state:
                    st.session_state.messages = [
                        {"role": "system", "content": f"Eres un analista de datos experto. Ayuda al usuario a entender sus datos. Utiliza este resumen estadístico como contexto:
{contexto_datos}"}
                    ]
                else:
                    # Garantizar que el prompt del sistema se actualiza si cambias de archivo o filtras
                    st.session_state.messages[0] = {"role": "system", "content": f"Eres un analista de datos experto. Ayuda al usuario a entender sus datos. Utiliza este resumen estadístico como contexto:
{contexto_datos}"}
                
                # Renderizar historial visualmente
                for msg in st.session_state.messages:
                    if msg["role"] != "system":
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])
                
                if prompt := st.chat_input("Ej: ¿Cuál es la tendencia principal que observas?"):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    with st.spinner("LLaMA 3.3 está analizando..."):
                        chat_completion = client.chat.completions.create(
                            messages=st.session_state.messages,
                            model="llama-3.3-70b-versatile",
                            temperature=0.3,
                        )
                        
                        respuesta = chat_completion.choices[0].message.content
                        st.session_state.messages.append({"role": "assistant", "content": respuesta})
                        with st.chat_message("assistant"):
                            st.markdown(respuesta)
            except Exception as e:
                st.error(f"Error de conexión con Groq: {e}. Verifica tu API Key.")
        else:
            st.info("👈 Por favor, ingresa tu API Key de Groq en la barra lateral para habilitar el chat.")
