"""
Feature engineering pos-venda - dados reais Ford.

Gera uma linha por VIN usando somente historico de servicos ate DATA_CORTE.
O target observa retorno futuro dentro da janela operacional de churn.
"""

import os

import numpy as np
import pandas as pd

from src.pipeline.config import DATA_CORTE, JANELA_CHURN_MESES, TARGET_CHURN


INPUT_PATH = "data/raw/vin_share_Desafio_02.xlsx"
OUTPUT_PATH = "data/processed/snapshots_pos_venda.csv"
CHURN_DATASET_PATH = "data/processed/dataset_churn_pos_venda.csv"
SHEET_NAME = "vin_share"
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
            parsed = pd.to_datetime(df[col], format="%m/%d/%Y", errors="coerce")
            if parsed.isna().mean() > 0.90:
                parsed = pd.to_datetime(df[col], errors="coerce")
            df[col + "_dt"] = parsed
    return df


def limpar_outliers(df):
    if "KM" in df.columns:
        df.loc[df["KM"] > KM_OUTLIER_THRESHOLD, "KM"] = np.nan
        df.loc[df["KM"] < 0, "KM"] = np.nan
    return df


def _validar_janela_observacao(df, data_corte, janela_churn_meses):
    fim_janela = data_corte + pd.DateOffset(months=janela_churn_meses)
    data_max = df["ServiceDate_dt"].max()
    janela_observavel = bool(pd.notna(data_max) and data_max >= fim_janela)

    print(f"Data de corte: {data_corte.date()}")
    print(f"Fim da janela futura ({janela_churn_meses}m): {fim_janela.date()}")
    print(f"Maior ServiceDate na base: {data_max.date() if pd.notna(data_max) else 'NaT'}")

    if not janela_observavel:
        print(
            "ALERTA: a base nao cobre a janela futura completa para esta data de corte. "
            "O target fica censurado; use uma DATA_CORTE mais antiga para avaliacao final."
        )
    return fim_janela, data_max, janela_observavel


def build_snapshot(df, data_corte, janela_churn_meses=JANELA_CHURN_MESES):
    data_corte = pd.Timestamp(data_corte)
    df = parsear_datas(df.copy())
    df = limpar_outliers(df)
    df = df[df["ServiceDate_dt"].notna()].copy()

    fim_janela, data_max, janela_observavel = _validar_janela_observacao(
        df, data_corte, janela_churn_meses
    )

    historico = df[df["ServiceDate_dt"] <= data_corte].copy()
    futuro = df[
        (df["ServiceDate_dt"] > data_corte)
        & (df["ServiceDate_dt"] <= fim_janela)
    ].copy()

    if historico.empty:
        raise ValueError("Nenhum evento de servico encontrado ate a DATA_CORTE.")

    agg_spec = {
        "qtde_revisoes_ate_corte": ("MaintenanceID", "count"),
        "primeiro_servico_ate_corte": ("ServiceDate_dt", "min"),
        "ultimo_servico_ate_corte": ("ServiceDate_dt", "max"),
        "modelo": ("ModelName", "first"),
        "ano_modelo": ("ModelYear", "first"),
        "dealer_code_ate_corte": ("DealerCode", "first"),
        "n_dealers_usados_ate_corte": ("DealerCode", "nunique"),
        "km_max_ate_corte": ("KM", "max"),
        "sales_date": ("SalesDate_dt", "first"),
        "delivery_date": ("DeliveryDate_dt", "first"),
        "warranty_date": ("WarrantyStartDate_dt", "first"),
    }
    if "IsAgendaSchedule" in historico.columns:
        agg_spec["pct_agenda_ate_corte"] = ("IsAgendaSchedule", "mean")

    snapshot = historico.groupby("VIN_Hash").agg(**agg_spec).reset_index()
    if "pct_agenda_ate_corte" not in snapshot.columns:
        snapshot["pct_agenda_ate_corte"] = np.nan

    snapshot["dias_desde_ultimo_servico_ate_corte"] = (
        data_corte - snapshot["ultimo_servico_ate_corte"]
    ).dt.days
    snapshot["meses_desde_ultimo_servico_ate_corte"] = (
        snapshot["dias_desde_ultimo_servico_ate_corte"] / 30.44
    ).round(2)
    snapshot["meses_relacionamento_ate_corte"] = (
        (snapshot["ultimo_servico_ate_corte"] - snapshot["primeiro_servico_ate_corte"]).dt.days / 30.44
    ).round(2)
    snapshot["dias_ate_primeira_revisao"] = (
        snapshot["primeiro_servico_ate_corte"] - snapshot["sales_date"]
    ).dt.days
    snapshot["dias_ate_entrega"] = (snapshot["delivery_date"] - snapshot["sales_date"]).dt.days
    snapshot["idade_veiculo_meses_ate_corte"] = (
        (data_corte - snapshot["sales_date"]).dt.days / 30.44
    ).round(2)
    snapshot["intervalo_medio_revisoes_dias_ate_corte"] = np.where(
        snapshot["qtde_revisoes_ate_corte"] > 1,
        (
            (snapshot["ultimo_servico_ate_corte"] - snapshot["primeiro_servico_ate_corte"]).dt.days
            / (snapshot["qtde_revisoes_ate_corte"] - 1)
        ),
        np.nan,
    )

    retorno = futuro.groupby("VIN_Hash").agg(
        qtd_servicos_pos_corte=("MaintenanceID", "count"),
        primeiro_servico_pos_corte=("ServiceDate_dt", "min"),
        ultimo_servico_pos_corte=("ServiceDate_dt", "max"),
    ).reset_index()

    snapshot = snapshot.merge(retorno, on="VIN_Hash", how="left")
    snapshot["qtd_servicos_pos_corte"] = snapshot["qtd_servicos_pos_corte"].fillna(0).astype(int)
    snapshot["voltou_pos_corte"] = snapshot["qtd_servicos_pos_corte"] > 0
    snapshot[TARGET_CHURN] = (~snapshot["voltou_pos_corte"]).astype(int)
    snapshot["data_corte"] = data_corte
    snapshot["fim_janela_churn"] = fim_janela
    snapshot["data_max_observada"] = data_max
    snapshot["janela_futura_observavel"] = janela_observavel

    # Alias temporario para notebooks antigos; nao usar em novos treinos.
    snapshot["churn"] = snapshot[TARGET_CHURN]
    return snapshot


def validar_snapshot(snapshot):
    print("\n=== VALIDACAO SNAPSHOT POS-VENDA ===")
    print(f"VINs com historico ate corte: {len(snapshot):,}")
    print(f"\nDistribuicao de {TARGET_CHURN}:")
    print(snapshot[TARGET_CHURN].value_counts(normalize=True).round(4))

    print("\nDistribuicao por modelo (top 10):")
    print(snapshot["modelo"].value_counts().head(10))

    print("\nMissing values relevantes:")
    cols_check = [
        "km_max_ate_corte",
        "dias_ate_primeira_revisao",
        "intervalo_medio_revisoes_dias_ate_corte",
        "ano_modelo",
    ]
    for col in cols_check:
        n_missing = snapshot[col].isna().sum()
        pct = n_missing / len(snapshot) * 100
        print(f"  {col}: {n_missing:,} ({pct:.2f}%)")


def main():
    os.makedirs("data/processed", exist_ok=True)

    df = carregar_ordens_servico()
    snapshot = build_snapshot(df, DATA_CORTE, JANELA_CHURN_MESES)
    validar_snapshot(snapshot)

    snapshot.to_csv(OUTPUT_PATH, index=False)
    snapshot.to_csv(CHURN_DATASET_PATH, index=False)
    print(f"\nSalvo em: {OUTPUT_PATH}")
    print(f"Salvo em: {CHURN_DATASET_PATH}")
    print(f"Shape: {snapshot.shape}")
    return snapshot


if __name__ == "__main__":
    main()
