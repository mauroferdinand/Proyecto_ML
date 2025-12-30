# crisp_dm_pipeline.py
# Implementación práctica (en código) de CRISP-DM para: Riesgo de incumplimiento de practicantes
# Dataset: practicantes_sintetico.csv
# Target:  estado_final (0 = cumplió, 1 = incumplió)

import os
import json
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# =========================
# 1) BUSINESS UNDERSTANDING
# =========================

@dataclass
class BusinessConfig:
    project_name: str = "APAC Practicantes - Riesgo Incumplimiento"
    target_col: str = "estado_final"   # 0 = cumplió, 1 = incumplió
    positive_class: int = 1
    # Umbrales para traducir probabilidad a "Bajo/Medio/Alto"
    low_thr: float = 0.33
    high_thr: float = 0.66

@dataclass
class DataConfig:
    data_path: str = "practicantes_sintetico.csv"
    date_col: str = "fecha_ingreso"
    id_cols_to_drop: tuple = ("id_practicante",)
    cutoff_date: str = "2024-12-31"    # Fecha de corte para calcular antigüedad

@dataclass
class OutputConfig:
    out_dir: str = "artifacts"
    model_dir: str = "artifacts/models"
    reports_dir: str = "artifacts/reports"

BUSINESS = BusinessConfig()
DATA = DataConfig()
OUT = OutputConfig()

os.makedirs(OUT.out_dir, exist_ok=True)
os.makedirs(OUT.model_dir, exist_ok=True)
os.makedirs(OUT.reports_dir, exist_ok=True)


# ======================
# 2) DATA UNDERSTANDING
# ======================

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def data_understanding_report(df: pd.DataFrame, target_col: str) -> dict:
    report = {}
    report["shape"] = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    report["columns"] = list(df.columns)
    report["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    report["missing_values"] = {c: int(df[c].isna().sum()) for c in df.columns}

    if target_col in df.columns:
        vc = df[target_col].value_counts(dropna=False)
        report["target_distribution"] = {str(k): int(v) for k, v in vc.items()}
        report["target_rate_positive"] = float((df[target_col] == BUSINESS.positive_class).mean())
    else:
        report["target_distribution"] = "TARGET_COLUMN_NOT_FOUND"

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    report["numeric_summary"] = df[numeric_cols].describe().to_dict() if numeric_cols else {}

    return report


# =====================
# 3) DATA PREPARATION
# =====================

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convertir fecha_ingreso a datetime y generar antiguedad_dias
    if DATA.date_col in df.columns:
        df[DATA.date_col] = pd.to_datetime(df[DATA.date_col], errors="coerce")
        cutoff = pd.to_datetime(DATA.cutoff_date)
        df["antiguedad_dias"] = (cutoff - df[DATA.date_col]).dt.days
        df = df.drop(columns=[DATA.date_col])

    # Eliminar IDs del modelado
    for c in DATA.id_cols_to_drop:
        if c in df.columns:
            df = df.drop(columns=[c])

    return df

def split_xy(df: pd.DataFrame, target_col: str):
    if target_col not in df.columns:
        raise ValueError(f"No existe la columna target '{target_col}' en el dataset.")
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y

def infer_feature_types(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_features, categorical_features

def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )


# ===========
# 4) MODELING
# ===========

def build_models():
    return {
        "logreg": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),
        "rf": RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    }

def train_model(preprocessor, model, X_train, y_train) -> Pipeline:
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)
    return pipe


# ==============
# 5) EVALUATION
# ==============

def evaluate(pipe: Pipeline, X_test, y_test) -> dict:
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, y_proba)

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr = auc(recall, precision)

    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "classification_report": rep,
        "confusion_matrix": cm
    }

def choose_best_model(results: dict) -> str:
    best_name = None
    best_score = (-1.0, -1.0)  # (roc_auc, pr_auc)
    for name, metrics in results.items():
        score = (metrics["roc_auc"], metrics["pr_auc"])
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


# ====================
# 6) DEPLOYMENT READY
# ====================

def risk_band(prob: float, low_thr: float, high_thr: float) -> str:
    if prob < low_thr:
        return "Bajo"
    if prob < high_thr:
        return "Medio"
    return "Alto"

def export_deployment_assets(best_pipe: Pipeline, X_sample: pd.DataFrame, out_model_path: str):
    joblib.dump(best_pipe, out_model_path)

    schema = {
        "expected_columns": list(X_sample.columns),
        "example_payload": X_sample.iloc[0].to_dict()
    }
    with open(os.path.join(OUT.model_dir, "input_schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)


def main():
    # 1) Business Understanding (metadatos de la corrida)
    business_meta = {
        "project": BUSINESS.project_name,
        "target": BUSINESS.target_col,
        "positive_class": BUSINESS.positive_class,
        "risk_thresholds": {"low": BUSINESS.low_thr, "high": BUSINESS.high_thr},
        "run_timestamp_utc": datetime.utcnow().isoformat()
    }

    # 2) Data Understanding
    df_raw = load_data(DATA.data_path)
    du = data_understanding_report(df_raw, BUSINESS.target_col)
    with open(os.path.join(OUT.reports_dir, "data_understanding.json"), "w", encoding="utf-8") as f:
        json.dump(du, f, ensure_ascii=False, indent=2)

    # 3) Data Preparation
    df = prepare_features(df_raw)
    X, y = split_xy(df, BUSINESS.target_col)

    numeric_features, categorical_features = infer_feature_types(X)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) Modeling + 5) Evaluation
    models = build_models()
    results = {}
    trained = {}

    for name, model in models.items():
        pipe = train_model(preprocessor, model, X_train, y_train)
        trained[name] = pipe

        metrics = evaluate(pipe, X_test, y_test)
        results[name] = metrics

        with open(os.path.join(OUT.reports_dir, f"evaluation_{name}.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    best_name = choose_best_model(results)
    best_pipe = trained[best_name]

    # Guardar resumen global
    summary = {
        "business": business_meta,
        "best_model": best_name,
        "all_results": {k: {"roc_auc": v["roc_auc"], "pr_auc": v["pr_auc"]} for k, v in results.items()}
    }
    with open(os.path.join(OUT.reports_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 6) Deployment assets
    model_path = os.path.join(OUT.model_dir, f"best_model_{best_name}.pkl")
    export_deployment_assets(best_pipe, X_train, model_path)

    # Crear muestra de predicciones para validar visualmente
    y_proba = best_pipe.predict_proba(X_test)[:, 1]
    df_pred = X_test.copy()
    df_pred["prob_riesgo"] = y_proba
    df_pred["nivel_riesgo"] = [risk_band(p, BUSINESS.low_thr, BUSINESS.high_thr) for p in y_proba]
    df_pred["estado_real"] = y_test.values

    df_pred.sample(min(200, len(df_pred)), random_state=42).to_csv(
        os.path.join(OUT.reports_dir, "sample_predictions.csv"),
        index=False
    )

    print("CRISP-DM pipeline finalizado.")
    print("Mejor modelo:", best_name)
    print("Modelo exportado en:", model_path)
    print("Reportes en:", OUT.reports_dir)


if __name__ == "__main__":
    main()