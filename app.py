import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="Dashboard Energía Renovable", layout="wide")

# 1. Carga de datos con caché
@st.cache_data
def load_data():
    # Leer el archivo anexo
    df = pd.read_csv("energia_renovable.csv")
    # Asegurar que la fecha sea datetime
    df['Fecha_Entrada_Operacion'] = pd.to_datetime(df['Fecha_Entrada_Operacion'])
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    st.stop()

# 2. Sidebar
st.sidebar.title("Configuración")
st.sidebar.image("https://via.placeholder.com/150", caption="Logo de la Empresa")

st.sidebar.header("Filtros Globales")
categorias_seleccionadas = st.sidebar.multiselect(
    "Selecciona Tecnología",
    options=df["Tecnologia"].unique(),
    default=df["Tecnologia"].unique()
)

rango_fechas = st.sidebar.date_input(
    "Rango de Fechas (Operación)",
    [df["Fecha_Entrada_Operacion"].min(), df["Fecha_Entrada_Operacion"].max()]
)

# Filtrar datos
if len(rango_fechas) == 2:
    start_date = pd.to_datetime(rango_fechas[0])
    end_date = pd.to_datetime(rango_fechas[1])
    df_filtrado = df[(df["Tecnologia"].isin(categorias_seleccionadas)) & 
                     (df["Fecha_Entrada_Operacion"] >= start_date) & 
                     (df["Fecha_Entrada_Operacion"] <= end_date)]
else:
    df_filtrado = df.copy()

st.title("📊 Dashboard de Análisis - Energía Renovable")

# 3. Pestañas (Tabs)
tab1, tab2, tab3 = st.tabs(["Resumen Ejecutivo (KPIs)", "Análisis Exploratorio", "Datos Crudos"])

# 3.1 Tab 1: Resumen ejecutivo KPI's adaptado al dataset
with tab1:
    st.header("Indicadores Clave de Rendimiento")
    col1, col2, col3 = st.columns(3)
    
    total_generacion = df_filtrado["Generacion_Diaria_MWh"].sum()
    total_inversion = df_filtrado["Inversion_Inicial_MUSD"].sum()
    eficiencia_promedio = df_filtrado["Eficiencia_Planta_Pct"].mean() if not df_filtrado.empty else 0
    
    # Adaptación de las métricas anteriores (Ventas, Beneficios, Margen) al contexto actual
    col1.metric("Generación Total (MWh)", f"{total_generacion:,.2f}")
    col2.metric("Inversión Total (MUSD)", f"${total_inversion:,.2f}")
    col3.metric("Eficiencia Promedio", f"{eficiencia_promedio:.1f}%")

# 3.2 Tab 2: Análisis exploratorio (Gráficos)
with tab2:
    st.header("Visualizaciones de Datos")
    
    col_graf_1, col_graf_2 = st.columns(2)
    
    with col_graf_1:
        st.subheader("Generación por Tecnología (Plotly)")
        fig_plotly = px.bar(
            df_filtrado.groupby("Tecnologia")["Generacion_Diaria_MWh"].sum().reset_index(),
            x="Tecnologia", y="Generacion_Diaria_MWh", color="Tecnologia",
            title="Generación Acumulada"
        )
        st.plotly_chart(fig_plotly, use_container_width=True)
        
    with col_graf_2:
        st.subheader("Tendencia de Inversión (Pyplot)")
        fig_plt, ax = plt.subplots(figsize=(6, 4))
        # Agrupar por mes para que el gráfico de línea se vea más claro
        df_fechas = df_filtrado.groupby(df_filtrado["Fecha_Entrada_Operacion"].dt.to_period("M"))["Inversion_Inicial_MUSD"].sum().reset_index()
        df_fechas["Fecha_Entrada_Operacion"] = df_fechas["Fecha_Entrada_Operacion"].dt.to_timestamp()
        ax.plot(df_fechas["Fecha_Entrada_Operacion"], df_fechas["Inversion_Inicial_MUSD"], marker='o', linestyle='-', color='g')
        ax.set_title("Inversión Inicial en el tiempo")
        ax.set_xlabel("Fecha de Operación")
        ax.set_ylabel("Inversión (MUSD)")
        plt.xticks(rotation=45)
        st.pyplot(fig_plt)
        
    st.subheader("Generación vs Inversión (Seaborn)")
    fig_sns, ax_sns = plt.subplots(figsize=(10, 4))
    sns.scatterplot(
        data=df_filtrado, 
        x="Inversion_Inicial_MUSD", 
        y="Generacion_Diaria_MWh", 
        hue="Tecnologia", 
        size="Capacidad_Instalada_MW", 
        ax=ax_sns
    )
    ax_sns.set_title("Scatter Plot: Inversión vs Generación")
    st.pyplot(fig_sns)

# 3.3 Datos Crudos y Reportes
with tab3:
    st.header("Datos Crudos")
    st.dataframe(df_filtrado, use_container_width=True)
    
    # Generar Reportes
    csv = df_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar Reporte CSV",
        data=csv,
        file_name='reporte_datos_energia.csv',
        mime='text/csv',
    )
