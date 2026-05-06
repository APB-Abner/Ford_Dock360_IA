import os

import numpy as np
import pandas as pd

from src.pipeline.config import LEAKAGE_COLUMNS


RANDOM_STATE = 42


def _make_rng(random_state):
    return np.random.default_rng(random_state)


def _exact_labels(rng, labels, probs, n_rows):
    counts = np.floor(np.array(probs) * n_rows).astype(int)
    counts[-1] += n_rows - counts.sum()
    values = np.concatenate([np.repeat(label, count) for label, count in zip(labels, counts)])
    rng.shuffle(values)
    return values


def _add_missing_values(df, rng):
    missing_rates = {
        "renda_mensal": 0.0247,
        "score_credito": 0.0204,
        "tempo_decisao_dias": 0.0120,
    }

    for col, rate in missing_rates.items():
        missing_count = int(len(df) * rate)
        missing_idx = rng.choice(df.index, missing_count, replace=False)
        df.loc[missing_idx, col] = np.nan


def _make_churn(df, rng):
    renda = df["renda_mensal"].fillna(df["renda_mensal"].median())
    score = df["score_credito"].fillna(df["score_credito"].median())
    decisao = df["tempo_decisao_dias"].fillna(df["tempo_decisao_dias"].median())

    risk = (
        1.45 * ((df["distancia_concessionaria_km"] - df["distancia_concessionaria_km"].mean()) / df["distancia_concessionaria_km"].std())
        + 1.25 * ((df["km_estimado_ano"] - df["km_estimado_ano"].mean()) / df["km_estimado_ano"].std())
        + 1.05 * ((decisao - decisao.mean()) / decisao.std())
        + 0.95 * ((score.mean() - score) / score.std())
        + 0.85 * ((renda.mean() - renda) / renda.std())
        + 0.70 * (df["cidade_tier"] == "interior").astype(int)
        + 0.60 * (df["perfil_digital"] == "baixo").astype(int)
        + 0.45 * (df["forma_pagamento"] == "financiado").astype(int)
        + rng.normal(0, 0.55, len(df))
    )
    cutoff = np.quantile(risk, 0.86)
    return (risk >= cutoff).astype(int)


def generate_base2(n_rows=500000, random_state=RANDOM_STATE):
    rng = _make_rng(random_state)

    estados = np.array(["SP", "RJ", "MG", "PR", "RS", "SC", "BA", "PE", "GO", "DF"])
    modelos = np.array(["Ka", "Fiesta", "EcoSport", "Ranger", "Territory", "Bronco", "Maverick"])
    segmentos = {
        "Ka": "hatch",
        "Fiesta": "hatch",
        "EcoSport": "suv",
        "Ranger": "pickup",
        "Territory": "suv",
        "Bronco": "suv",
        "Maverick": "pickup",
    }
    modelo_veiculo = rng.choice(modelos, n_rows, p=[0.16, 0.14, 0.22, 0.18, 0.14, 0.08, 0.08])
    preco_base = {
        "Ka": 62000,
        "Fiesta": 70000,
        "EcoSport": 105000,
        "Ranger": 210000,
        "Territory": 190000,
        "Bronco": 260000,
        "Maverick": 240000,
    }
    preco_veiculo = np.array([preco_base[modelo] for modelo in modelo_veiculo])
    preco_veiculo = np.round(preco_veiculo * rng.normal(1.0, 0.09, n_rows), 2)
    entrada_pct = np.round(rng.beta(2.2, 5.0, n_rows) * 0.65 + 0.05, 3)
    forma_pagamento = rng.choice(["financiado", "a_vista", "consorcio"], n_rows, p=[0.66, 0.18, 0.16])
    valor_financiado = np.where(
        forma_pagamento == "a_vista",
        0,
        np.round(preco_veiculo * (1 - entrada_pct), 2),
    )

    df = pd.DataFrame(
        {
            "id_cliente": np.arange(1, n_rows + 1),
            "idade": np.clip(rng.normal(42, 12, n_rows).round(), 18, 80).astype(int),
            "genero": rng.choice(["F", "M", "Outro"], n_rows, p=[0.48, 0.50, 0.02]),
            "estado": rng.choice(estados, n_rows, p=[0.28, 0.12, 0.12, 0.09, 0.08, 0.07, 0.08, 0.05, 0.06, 0.05]),
            "cidade_tier": rng.choice(["capital", "metropolitana", "interior"], n_rows, p=[0.34, 0.28, 0.38]),
            "renda_mensal": np.round(rng.lognormal(mean=8.85, sigma=0.55, size=n_rows), 2),
            "score_credito": np.clip(rng.normal(670, 95, n_rows).round(), 300, 950).astype(int),
            "estado_civil": rng.choice(["solteiro", "casado", "divorciado", "viuvo"], n_rows, p=[0.34, 0.49, 0.13, 0.04]),
            "filhos": np.clip(rng.poisson(1.1, n_rows), 0, 5),
            "profissao": rng.choice(["CLT", "autonomo", "empresario", "servidor", "aposentado"], n_rows, p=[0.45, 0.23, 0.14, 0.12, 0.06]),
            "canal_compra": rng.choice(["loja", "site", "telefone", "parceiro"], n_rows, p=[0.62, 0.18, 0.07, 0.13]),
            "modelo_veiculo": modelo_veiculo,
            "segmento_veiculo": np.array([segmentos[modelo] for modelo in modelo_veiculo]),
            "ano_modelo": rng.choice([2020, 2021, 2022, 2023, 2024, 2025], n_rows, p=[0.08, 0.12, 0.18, 0.24, 0.25, 0.13]),
            "preco_veiculo": preco_veiculo,
            "valor_financiado": valor_financiado,
            "prazo_financiamento_meses": rng.choice([0, 24, 36, 48, 60], n_rows, p=[0.18, 0.12, 0.22, 0.27, 0.21]),
            "entrada_pct": entrada_pct,
            "forma_pagamento": forma_pagamento,
            "km_estimado_ano": np.clip(rng.normal(14500, 5200, n_rows).round(), 3000, 45000).astype(int),
            "tempo_habilitacao_anos": np.clip(rng.normal(18, 11, n_rows).round(), 0, 60).astype(int),
            "distancia_concessionaria_km": np.round(rng.gamma(2.1, 7.5, n_rows), 1),
            "tempo_decisao_dias": np.clip(rng.gamma(2.4, 5.5, n_rows).round(), 1, 90).astype(int),
            "perfil_digital": rng.choice(["baixo", "medio", "alto"], n_rows, p=[0.25, 0.47, 0.28]),
        }
    )

    _add_missing_values(df, rng)

    return df


def generate_base1(n_rows=500000, random_state=RANDOM_STATE):
    rng = _make_rng(random_state)
    df = generate_base2(n_rows=n_rows, random_state=random_state)

    perfil = _exact_labels(
        rng,
        ["fiel", "economico", "abandono", "esquecido"],
        [0.30, 0.26, 0.24, 0.20],
        n_rows,
    )
    perfil_idx = pd.Series(perfil).map({"fiel": 0, "economico": 1, "abandono": 2, "esquecido": 3}).to_numpy()

    prob_revisao = np.array([0.84, 0.63, 0.38, 0.52])[perfil_idx]
    fez_revisao = rng.binomial(1, prob_revisao)
    meses_revisao = np.where(
        fez_revisao == 1,
        np.clip(rng.normal(np.array([5.8, 7.2, 9.8, 10.5])[perfil_idx], 2.1, n_rows).round(), 1, 24),
        np.nan,
    )
    perdeu_revisao = (fez_revisao == 0).astype(int)
    share_rede = np.where(
        fez_revisao == 1,
        np.clip(rng.beta(np.array([8.0, 4.0, 2.0, 3.0])[perfil_idx], 2.5), 0, 1),
        np.clip(rng.beta(1.2, 6.0, n_rows), 0, 1),
    )
    qtde_revisoes = np.clip(
        rng.poisson(np.array([2.7, 1.8, 0.8, 1.2])[perfil_idx], n_rows),
        0,
        8,
    )
    satisfacao = np.clip(
        rng.normal(np.array([8.6, 7.3, 5.6, 6.3])[perfil_idx], 1.25, n_rows).round(1),
        0,
        10,
    )

    churn = _make_churn(df, rng)

    df["perfil_latente"] = perfil
    df["fez_primeira_revisao_rede"] = fez_revisao
    df["meses_ate_primeira_revisao"] = meses_revisao
    df["perdeu_primeira_revisao"] = perdeu_revisao
    df["voltou_tarde_revoltado"] = ((meses_revisao > 12) | ((fez_revisao == 0) & (rng.random(n_rows) < 0.18))).astype(int)
    df["trouxe_oleo_externo"] = rng.binomial(1, np.array([0.07, 0.35, 0.23, 0.18])[perfil_idx])
    df["pede_desconto_revisao"] = rng.binomial(1, np.array([0.12, 0.58, 0.32, 0.30])[perfil_idx])
    df["sensibilidade_desconto_pos"] = np.clip(rng.normal(np.array([3.2, 8.2, 6.5, 6.8])[perfil_idx], 1.4, n_rows).round(1), 0, 10)
    df["qtde_revisoes_24m"] = qtde_revisoes
    df["share_revisoes_rede_24m"] = np.round(share_rede, 3)
    df["gasto_manutencao_rede_24m"] = np.round(qtde_revisoes * rng.normal(980, 260, n_rows) * share_rede, 2)
    df["satisfacao_marca_24m"] = satisfacao
    df["churn_rede_24m"] = churn

    return df


def check_leakage(x):
    vazamentos = [col for col in LEAKAGE_COLUMNS if col in x.columns]
    if vazamentos:
        raise ValueError(f"Colunas com data leakage em X: {vazamentos}")


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    base1 = generate_base1()
    base2 = generate_base2()
    base1.to_csv("data/raw/ford_clientes_historico_completo.csv", index=False)
    base2.to_csv("data/raw/ford_clientes_operacional_compra.csv", index=False)
    print(f"Base 1: {base1.shape} -> data/raw/ford_clientes_historico_completo.csv")
    print(f"Base 2: {base2.shape} -> data/raw/ford_clientes_operacional_compra.csv")
