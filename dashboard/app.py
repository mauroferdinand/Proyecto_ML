import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "models", "best_model_logreg.pkl")
SUMMARY_PATH = os.path.join(BASE_DIR, "artifacts", "reports", "summary.json")
SAMPLE_PRED_PATH = os.path.join(BASE_DIR, "artifacts", "reports", "sample_predictions.csv")

st.set_page_config(page_title="APAC - Riesgo Practicantes", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_summary():
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_sample_predictions():
    return pd.read_csv(SAMPLE_PRED_PATH)

def risk_band(prob: float):
    if prob < 0.33:
        return "Bajo"
    if prob < 0.66:
        return "Medio"
    return "Alto"

st.title("Dashboard – Riesgo de Incumplimiento de Practicantes")

# Panel superior: métricas del modelo
col1, col2, col3 = st.columns(3)

try:
    summary = load_summary()
    best_model = summary.get("best_model", "N/A")
    results = summary.get("all_results", {})
    col1.metric("Mejor modelo", best_model)
    if best_model in results:
        col2.metric("ROC-AUC", f"{results[best_model]['roc_auc']:.3f}")
        col3.metric("PR-AUC", f"{results[best_model]['pr_auc']:.3f}")
    else:
        col2.metric("ROC-AUC", "N/A")
        col3.metric("PR-AUC", "N/A")
except Exception:
    col1.metric("Mejor modelo", "N/A")
    col2.metric("ROC-AUC", "N/A")
    col3.metric("PR-AUC", "N/A")

st.divider()

# Cargar predicciones de ejemplo para vista de gestión
df_pred = load_sample_predictions()

# Filtros
with st.sidebar:
    st.header("Filtros")
    if "area_asignada" in df_pred.columns:
        areas = ["(Todas)"] + sorted(df_pred["area_asignada"].dropna().unique().tolist())
        area_sel = st.selectbox("Área", areas)
    else:
        area_sel = "(Todas)"

    if "supervisor_asignado" in df_pred.columns:
        sups = ["(Todos)"] + sorted(df_pred["supervisor_asignado"].dropna().unique().tolist())
        sup_sel = st.selectbox("Supervisor", sups)
    else:
        sup_sel = "(Todos)"

# Aplicar filtros
df_view = df_pred.copy()
if area_sel != "(Todas)" and "area_asignada" in df_view.columns:
    df_view = df_view[df_view["area_asignada"] == area_sel]
if sup_sel != "(Todos)" and "supervisor_asignado" in df_view.columns:
    df_view = df_view[df_view["supervisor_asignado"] == sup_sel]

# KPIs de gestión
c1, c2, c3, c4 = st.columns(4)
c1.metric("Practicantes (vista)", len(df_view))
c2.metric("Riesgo Alto", int((df_view["nivel_riesgo"] == "Alto").sum()))
c3.metric("Riesgo Medio", int((df_view["nivel_riesgo"] == "Medio").sum()))
c4.metric("Riesgo Bajo", int((df_view["nivel_riesgo"] == "Bajo").sum()))

# Tabla top alto riesgo
st.subheader("Top practicantes con mayor riesgo")
df_top = df_view.sort_values("prob_riesgo", ascending=False).head(20)
st.dataframe(df_top, use_container_width=True)

# Gráfico: distribución de probabilidad
st.subheader("Distribución de probabilidad de riesgo")
fig = plt.figure()
plt.hist(df_view["prob_riesgo"].values, bins=20)
plt.xlabel("Probabilidad de incumplimiento")
plt.ylabel("Cantidad")
st.pyplot(fig)

st.divider()

# Predicción manual (opcional) - si quieres probar el .pkl directo
st.subheader("Predicción manual (demo)")

model = load_model()

# Tomar las columnas del sample sin columnas de salida
cols_excluir = {"prob_riesgo", "nivel_riesgo", "estado_real"}
feature_cols = [c for c in df_pred.columns if c not in cols_excluir]

with st.expander("Ingresar un caso (usa valores numéricos / categorías existentes)"):
    input_data = {}
    # Mostramos unos pocos campos clave si existen
    campos_clave = [c for c in feature_cols if c in [
        "asistencia_pct", "tardanzas_30d", "inasistencias_injust_30d",
        "entregables_a_tiempo_pct", "eval_supervisor_promedio", "incidencias_registradas",
        "area_asignada", "supervisor_asignado"
    ]]

    for c in campos_clave:
        if df_pred[c].dtype == "object":
            opts = sorted(df_pred[c].dropna().unique().tolist())
            input_data[c] = st.selectbox(c, opts)
        else:
            val = float(df_pred[c].dropna().median())
            input_data[c] = st.number_input(c, value=val)

    # Completar el resto con medianas/modas
    for c in feature_cols:
        if c not in input_data:
            if df_pred[c].dtype == "object":
                input_data[c] = df_pred[c].dropna().mode().iloc[0]
            else:
                input_data[c] = float(df_pred[c].dropna().median())

    if st.button("Predecir riesgo"):
        X_new = pd.DataFrame([input_data])[feature_cols]
        proba = float(model.predict_proba(X_new)[:, 1][0])
        st.success(f"Probabilidad de incumplimiento: {proba:.3f}")
        st.info(f"Nivel de riesgo: {risk_band(proba)}")