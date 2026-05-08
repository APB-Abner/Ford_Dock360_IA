"""
Feature Engineering — Caminho B (dataset real Ford)

Transforma o dataset de ordens de servico (1 linha = 1 evento) em uma tabela
agregada por VIN (1 linha = 1 veiculo) com features derivadas e targets.

Saidas:
  data/processed/vins_agregados.csv — features + target de churn por VIN

Targets:
  - churn (binario): 1 se VIN sem servico ha mais de 18 meses, 0 caso contrario
  - perfil_cluster: gerado em etapa posterior (clustering_real.py)

Decisoes de negocio:
  - Janela de churn: 18 meses (padrao industria automotiva)
  - Data de referencia: ServiceDate maximo do dataset
  - Outliers de KM acima de 500.000 sao tratados como NaN (erros de digitacao)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.config import RANDOM_STATE


INPUT_PATH = "data/raw/vin_share_Desafio_02.xlsx"
OUTPUT_PATH = "data/processed/vins_agregados.csv"
SHEET_NAME = "vin_share"
JANELA_CHURN_MESES = 18
KM_OUTLIER_THRESHOLD = 500_000


def carregar_ordens_servico(input_path=INPUT_PATH, sheet_name=SHEET_NAME):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset nao encontrado: {input_path}")

    print(f"Carregando {input_path}...")
    df = pd.read_excel(input_path, sheet_name=sheet_name)
    print(f"Carregado: {df.shape[0]:,} linhas, {df.shape[1]} colunas")
    return df


def parsear_datas(df):
    date_cols = [
        "ServiceDate", "ServiceOpenDate", "ServiceClosedDate",
        "InvoiceDate", "SalesDate", "DeliveryDate",
        "RegistrationDate", "WarrantyStartDate",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col + "_dt"] = pd.to_datetime(df[col], format="%m/%d/%Y", errors="coerce")
    return df


def limpar_outliers(df):
    df.loc[df["KM"] > KM_OUTLIER_THRESHOLD, "KM"] = np.nan
    df.loc[df["KM"] < 0, "KM"] = np.nan
    return df


def agregar_por_vin(df, ref_date):
    print(f"Data de referencia: {ref_date}")

    agg = df.groupby("VIN_Hash").agg(
        qtde_revisoes=("MaintenanceID", "count"),
        primeiro_servico=("ServiceDate_dt", "min"),
        ultimo_servico=("ServiceDate_dt", "max"),
        modelo=("ModelName", "first"),
        ano_modelo=("ModelYear", "first"),
        dealer_code=("DealerCode", "first"),
        n_dealers_usados=("DealerCode", "nunique"),
        km_max=("KM", "max"),
        sales_date=("SalesDate_dt", "first"),
        delivery_date=("DeliveryDate_dt", "first"),
        warranty_date=("WarrantyStartDate_dt", "first"),
        pct_agenda=("IsAgendaSchedule", "mean"),
    ).reset_index()

    # Features derivadas
    agg["dias_desde_ultimo_servico"] = (ref_date - agg["ultimo_servico"]).dt.days
    agg["meses_desde_ultimo_servico"] = (agg["dias_desde_ultimo_servico"] / 30.44).round(2)
    agg["meses_relacionamento"] = ((agg["ultimo_servico"] - agg["primeiro_servico"]).dt.days / 30.44).round(2)
    agg["dias_ate_primeira_revisao"] = (agg["primeiro_servico"] - agg["sales_date"]).dt.days
    agg["dias_ate_entrega"] = (agg["delivery_date"] - agg["sales_date"]).dt.days
    agg["idade_veiculo_meses"] = ((ref_date - agg["sales_date"]).dt.days / 30.44).round(2)

    # Intervalo medio entre revisoes
    agg["intervalo_medio_revisoes_dias"] = np.where(
        agg["qtde_revisoes"] > 1,
        ((agg["ultimo_servico"] - agg["primeiro_servico"]).dt.days / (agg["qtde_revisoes"] - 1)),
        np.nan,
    )

    # Target binario de churn
    agg["churn"] = (agg["meses_desde_ultimo_servico"] > JANELA_CHURN_MESES).astype(int)

    return agg


def validar_agregado(agg):
    print("\n=== VALIDACAO ===")
    print(f"VINs unicos: {len(agg):,}")
    print(f"\nDistribuicao de churn (janela {JANELA_CHURN_MESES}m):")
    print(agg["churn"].value_counts(normalize=True).round(4))

    print(f"\nDistribuicao por modelo (top 10):")
    print(agg["modelo"].value_counts().head(10))

    print(f"\nMissing values relevantes:")
    cols_check = ["km_max", "dias_ate_primeira_revisao", "dias_ate_entrega",
                  "intervalo_medio_revisoes_dias", "ano_modelo"]
    for col in cols_check:
        n_missing = agg[col].isna().sum()
        pct = n_missing / len(agg) * 100
        print(f"  {col}: {n_missing:,} ({pct:.2f}%)")


def main():
    os.makedirs("data/processed", exist_ok=True)

    df = carregar_ordens_servico()
    df = parsear_datas(df)
    df = limpar_outliers(df)

    ref_date = df["ServiceDate_dt"].max()
    agg = agregar_por_vin(df, ref_date)

    validar_agregado(agg)

    agg.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSalvo em: {OUTPUT_PATH}")
    print(f"Shape: {agg.shape}")
    return agg


if __name__ == "__main__":
    main()
