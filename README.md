# Ford VinGuard

Projeto academico de Machine Learning para prever risco de churn na rede Ford e apoiar a ficha de abordagem comercial com perfil de cliente, probabilidade de churn e historico de problemas por modelo.

O repositorio contem geracao de bases sinteticas, segmentacao com K-Means, treinamento de modelos, analises com graficos, rastreamento com MLflow e uma API FastAPI protegida por JWT.

## Objetivo

O Ford VinGuard simula um pipeline de dados para identificar clientes com maior probabilidade de abandonar a rede de manutencao. A API retorna:

- predicao de churn: `churn` ou `no_churn`;
- probabilidade calibrada de churn;
- nivel de risco: `low`, `medium` ou `high`;
- perfil previsto, quando o modelo opcional estiver disponivel;
- acao recomendada para abordagem;
- top 3 problemas historicos por modelo, quando informado `modelo_veiculo`.

## Estrutura do Projeto

```text
.
|-- data/
|   |-- raw/                  # CSVs brutos e base de complaints
|   `-- processed/            # Artefatos intermediarios
|-- models/                   # Modelos joblib e checksums sha256
|-- notebooks/                # Notebooks de EDA e pipeline completo
|-- reports/                  # Graficos, CSVs e relatorios gerados
|-- scripts/                  # Scripts auxiliares
|-- src/
|   |-- api/                  # API FastAPI
|   `-- pipeline/             # Geracao, treino, clustering, MLflow e analises
|-- tests/                    # Testes unitarios
|-- requirements.txt
|-- requirements-dev.txt
`-- README.md
```

## Principais Arquivos

### Pipeline

- `src/pipeline/generate_data.py`: gera as bases sinteticas `Base 1` e `Base 2`.
- `src/pipeline/clustering.py`: executa K-Means, escolhe `k` por silhouette e salva `cluster_labels.csv`.
- `src/pipeline/train_classifier.py`: treina e salva o classificador opcional de perfil.
- `src/pipeline/train_churn.py`: treina o modelo calibrado de churn e salva checksum SHA256.
- `src/pipeline/visualizations.py`: gera arvore, importancia de features e matriz de confusao.
- `src/pipeline/bias_analysis.py`: avalia F1-Macro por subgrupos sensiveis.
- `src/pipeline/mlflow_tracking.py`: registra experimentos no MLflow.
- `src/pipeline/complaints_loader.py`: carrega top 3 complaints por modelo Ford.

### API

- `src/api/main.py`: instancia FastAPI e registra rotas.
- `src/api/routers/health.py`: healthcheck de chave e artefatos.
- `src/api/routers/predict.py`: endpoints `/predict` e `/predict/batch`.
- `src/api/security/auth.py`: criacao e validacao de JWT com roles.
- `src/api/services/predictor.py`: carregamento dos modelos e predicao.
- `src/api/models/schemas.py`: schemas Pydantic de entrada e saida.

## Bases de Dados

### Base 1: historico completo

Arquivo esperado:

```text
data/raw/ford_clientes_historico_completo.csv
```

Contem dados historicos e colunas pos-compra, incluindo target de churn e variaveis usadas apenas para segmentacao. Essa base nao deve ser usada diretamente como `X` do classificador de churn.

### Base 2: operacional de compra

Arquivo esperado:

```text
data/raw/ford_clientes_operacional_compra.csv
```

Contem somente informacoes disponiveis no momento operacional da compra. Essa e a base de features usada pelo modelo de churn em producao.

### Complaints por modelo

Arquivo esperado:

```text
data/raw/ford_complaints_top3_por_modelo.csv
```

Colunas:

- `modelo`
- `rank`
- `componente`
- `total_reclamacoes`

Esse arquivo nao entra no treinamento de ML. Ele e usado apenas como contexto adicional na ficha de abordagem retornada pela API.

## Regra Critica: Data Leakage

As colunas abaixo nunca podem aparecer no `X` de treino ou teste do classificador de churn:

```text
fez_primeira_revisao_rede
meses_ate_primeira_revisao
perdeu_primeira_revisao
voltou_tarde_revoltado
trouxe_oleo_externo
pede_desconto_revisao
sensibilidade_desconto_pos
qtde_revisoes_24m
share_revisoes_rede_24m
gasto_manutencao_rede_24m
satisfacao_marca_24m
churn_rede_24m
```

A lista oficial fica em `src/pipeline/config.py`. Os scripts de treino implementam `check_leakage()` e os testes em `tests/test_leakage.py` validam essa protecao automaticamente.

## Requisitos

Python recomendado: 3.11.

Dependencias principais:

- scikit-learn 1.4.2
- pandas 2.2.2
- numpy 1.26.4
- matplotlib 3.8.4
- seaborn 0.13.2
- mlflow 2.11.3
- fastapi 0.111.0
- uvicorn 0.29.0
- pydantic 2.7.1
- pydantic-settings 2.2.1
- python-jose 3.3.0
- joblib 1.3.2

## Instalacao

Crie e ative um ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependencias:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Configure a chave JWT antes de importar ou subir a API:

```bash
export SECRET_KEY="troque-por-uma-chave-com-pelo-menos-32-caracteres"
```

Variaveis opcionais:

```bash
export JWT_ISSUER="ford-vinguard-api"
export JWT_AUDIENCE="ford-vinguard-api"
```

## Como Reproduzir o Pipeline

Execute os comandos a partir da raiz do projeto.

### 1. Gerar dados sinteticos

```bash
python -m src.pipeline.generate_data
```

Saidas:

- `data/raw/ford_clientes_historico_completo.csv`
- `data/raw/ford_clientes_operacional_compra.csv`

### 2. Gerar labels de cluster

```bash
python -m src.pipeline.clustering
```

Saidas:

- `data/processed/cluster_labels.csv`
- `reports/elbow_silhouette.png`
- `reports/clusters_pca.png`

### 3. Treinar classificador de perfil

```bash
python -m src.pipeline.train_classifier
```

Saidas:

- `models/perfil_rf_classifier.joblib`
- `models/perfil_rf_classifier.sha256`
- `reports/model_comparison`

Esse modelo e opcional para a API. Se ele nao existir, a API ainda retorna churn.

### 4. Treinar modelo de churn

```bash
python -m src.pipeline.train_churn
```

Saidas:

- `models/churn_rf_calibrated.joblib`
- `models/churn_rf_calibrated.sha256`
- `reports/precision_recall_churn.png`
- `reports/confusion_matrix_churn.png`

O modelo de churn e obrigatorio para `/predict` e `/predict/batch`.

### 5. Gerar visualizacoes auxiliares

```bash
python -m src.pipeline.visualizations
```

Saidas principais:

- `reports/decision_tree.png`
- `reports/feature_importance.png`
- `reports/feature_importance.csv`
- `reports/confusion_matrix_rf.png`

### 6. Rodar analise de vies

```bash
python -m src.pipeline.bias_analysis
```

Saidas:

- `reports/bias_analysis.csv`
- `reports/bias_genero.png`
- `reports/bias_renda.png`
- `reports/bias_score_credito.png`

## MLflow

O projeto usa MLflow para registrar parametros, metricas e artefatos.

Para abrir a interface local:

```bash
mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5000
```

Depois acesse:

```text
http://localhost:5000
```

## API FastAPI

### Subir a API

```bash
export SECRET_KEY="troque-por-uma-chave-com-pelo-menos-32-caracteres"
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Documentacao interativa:

```text
http://localhost:8000/docs
```

### Healthcheck

```bash
curl http://localhost:8000/health
```

Retorna `200` quando a chave JWT e o modelo obrigatorio existem. Retorna `503` com status `degraded` quando faltar configuracao ou artefato.

### Autenticacao

Os endpoints de predicao exigem Bearer Token JWT com role `analyst` ou `admin`.

Para gerar um token local:

```bash
python -c "from src.api.security.auth import create_access_token; print(create_access_token('dev-user', 'analyst'))"
```

Use o token no header:

```text
Authorization: Bearer <TOKEN>
```

### Predicao individual

Endpoint:

```text
POST /predict
```

Exemplo de payload:

```json
{
  "features": {
    "idade": 42,
    "genero": "M",
    "estado": "SP",
    "cidade_tier": "capital",
    "renda_mensal": 8500,
    "score_credito": 720,
    "estado_civil": "casado",
    "filhos": 1,
    "profissao": "CLT",
    "canal_compra": "loja",
    "modelo_veiculo": "Ranger",
    "segmento_veiculo": "pickup",
    "ano_modelo": 2024,
    "preco_veiculo": 210000,
    "valor_financiado": 150000,
    "prazo_financiamento_meses": 48,
    "entrada_pct": 0.28,
    "forma_pagamento": "financiado",
    "km_estimado_ano": 18000,
    "tempo_habilitacao_anos": 20,
    "distancia_concessionaria_km": 12.5,
    "tempo_decisao_dias": 14,
    "perfil_digital": "medio"
  },
  "modelo_veiculo": "Ranger"
}
```

Exemplo com `curl`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d @payload.json
```

Resposta:

```json
{
  "prediction": "no_churn",
  "churn_probability": 0.231421,
  "risk_level": "low",
  "perfil_previsto": "fiel",
  "probabilidades_perfil": {
    "abandono": 0.05,
    "economico": 0.12,
    "esquecido": 0.08,
    "fiel": 0.75
  },
  "acao_recomendada": "Nenhuma acao ativa. Registrar para agradecimento apos proxima revisao.",
  "historico_problemas": [
    {
      "rank": 1,
      "componente": "ENGINE",
      "total_reclamacoes": 10
    }
  ]
}
```

Os valores acima sao apenas exemplo. O retorno real depende dos artefatos em `models/` e dos dados em `data/raw/`.

### Predicao em lote

Endpoint:

```text
POST /predict/batch
```

Payload:

```json
{
  "items": [
    {
      "features": {
        "idade": 42
      },
      "modelo_veiculo": "Ranger"
    }
  ]
}
```

O limite configurado no schema e de 500 itens por requisicao. Cada item deve conter todas as features esperadas pelo modelo treinado; se faltar alguma, a API retorna `422` com `missing_features`.

## Niveis de Risco

A API classifica o risco a partir da probabilidade de churn:

- `high`: probabilidade maior ou igual a `0.70`;
- `medium`: probabilidade maior ou igual a `0.40` e menor que `0.70`;
- `low`: probabilidade menor que `0.40`.

A predicao `churn` e retornada quando a probabilidade e maior ou igual a `0.50`.

## Modelos e Checksums

A API verifica integridade dos modelos antes de carregar:

- `models/churn_rf_calibrated.joblib`
- `models/churn_rf_calibrated.sha256`
- `models/perfil_rf_classifier.joblib`
- `models/perfil_rf_classifier.sha256`

O modelo de churn e obrigatorio. O modelo de perfil e opcional.

Durante o carregamento, o servico tenta deixar `models/` e arquivos `.joblib` como read-only. Se o checksum estiver ausente ou invalido, a API retorna erro `503`.

## Testes

Rode a suite:

```bash
pytest -q
```

Testes existentes:

- `tests/test_leakage.py`: bloqueia colunas proibidas no treino.
- `tests/test_health.py`: valida healthcheck sem modelos e com modelos.
- `tests/test_predict_batch.py`: valida predicao vetorizada em lote.

## Notebooks

Notebooks disponiveis:

- `notebooks/00_pipeline_completo.ipynb`
- `notebooks/01_eda_base1.ipynb`
- `notebooks/02_eda_base2.ipynb`

Para abrir:

```bash
jupyter lab --ip 0.0.0.0 --port 8888
```

## Portas Usadas

- `8000`: FastAPI
- `8888`: Jupyter
- `5000`: MLflow UI

## Boas Praticas Aplicadas

- `random_state=42` em rotinas com aleatoriedade.
- `stratify=y` em splits de classificacao.
- `class_weight="balanced"` nos classificadores.
- Pre-processamento dentro de `sklearn.Pipeline`.
- Uso de F1-Macro, AUC-ROC e PR-AUC em vez de accuracy como metrica principal.
- Bloqueio automatico de colunas com data leakage.
- `matplotlib.use("Agg")` em scripts que geram graficos.
- Artefatos de modelo com checksum SHA256.

## Troubleshooting

### `SECRET_KEY` invalida

Erro comum:

```text
SECRET_KEY invalida: configure uma chave com pelo menos 32 caracteres
```

Solucao:

```bash
export SECRET_KEY="troque-por-uma-chave-com-pelo-menos-32-caracteres"
```

### API retorna `503` no `/health`

Verifique:

- se `models/churn_rf_calibrated.joblib` existe;
- se `models/churn_rf_calibrated.sha256` existe e corresponde ao modelo;
- se `SECRET_KEY` foi configurada;
- se o diretorio `models/` esta acessivel.

### API retorna `422` em `/predict`

O modelo foi treinado com uma lista fixa de features. Se alguma feature esperada estiver ausente no payload, a resposta inclui `missing_features`.

### Modelo de perfil nao aparece na resposta

O modelo `models/perfil_rf_classifier.joblib` e opcional. Se ele nao existir, os campos `perfil_previsto`, `probabilidades_perfil` e `acao_recomendada` ficam `null`.

## Observacoes

Este projeto usa dados sinteticos e tem finalidade academica. Para uso em producao real, seriam necessarios dados historicos validados, split temporal por safra, monitoramento de drift, revisao juridica/privacidade e avaliacao de impacto em clientes.
