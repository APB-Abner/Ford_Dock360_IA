# Ford VinGuard

Projeto de Machine Learning para prever risco de churn na rede Ford Brasil e apoiar a ficha de abordagem comercial (Dock 360) com perfil de cliente, probabilidade de churn e histórico de problemas.


## Objetivo

O Ford VinGuard identifica clientes com maior probabilidade de abandonar a rede de manutenção (churn definido como 18 meses sem serviços). A API retorna:

- Predição de churn: `churn` ou `no_churn`;
- Probabilidade calibrada de churn;
- Nível de risco: `low`, `medium` ou `high`;
- Perfil previsto via clustering comportamental;
- Ação recomendada para abordagem comercial;
- Top 3 problemas históricos por modelo Ford.

## Estrutura do Projeto

```text
.
|-- data/
|   |-- raw/                  # Dataset real (.xlsx) e complaints (.csv)
|   `-- processed/            # VINs agregados e labels de cluster
|-- models/                   # Modelos joblib e checksums sha256
|-- notebooks/                # Notebooks de EDA e Pipeline Completo
|-- reports/                  # Gráficos e relatórios de métricas
|-- src/
|   |-- api/                  # API FastAPI 
|   `-- pipeline/             # Scripts de engenharia e treino (Real)
|-- tests/                    # Testes de leakage, saúde e schema
|-- requirements.txt
`-- README.md
```

## Principais Arquivos

### Pipeline Real

- `src/pipeline/feature_engineering_real.py`: Agrega 600k+ OS em 175k+ VINs únicos.
- `src/pipeline/clustering_real.py`: Executa K-Means em features comportamentais.
- `src/pipeline/train_classifier_real.py`: Treina o classificador de perfil de cliente.
- `src/pipeline/train_churn_real.py`: Treina o modelo calibrado de churn.
- `src/pipeline/mlflow_tracking.py`: Registra experimentos no MLflow.
- `src/pipeline/config.py`: Centraliza constantes e lista de anti-leakage.

### API

- `src/api/main.py`: Ponto de entrada FastAPI.
- `src/api/services/predictor.py`: Lógica de inferência.
- `src/api/models/schemas.py`: Contratos Pydantic com exemplos reais.

## Regra Crítica: Data Leakage

As seguintes features representam comportamento **posterior** à compra e são proibidas no treinamento dos classificadores:
`qtde_revisoes`, `meses_desde_ultimo_servico`, `meses_relacionamento`, `n_dealers_usados`, `km_max`, `pct_agenda`, `intervalo_medio_revisoes_dias`, `dias_ate_primeira_revisao`, `primeiro_servico`, `ultimo_servico`.

O modelo de produção utiliza apenas features disponíveis no momento da compra:
`modelo`, `ano_modelo`, `dias_ate_entrega`, `idade_veiculo_meses`.

## Como Reproduzir o Pipeline

Execute os scripts na ordem abaixo a partir da raiz:

### 1. Engenharia de Features
```bash
python -m src.pipeline.feature_engineering_real
```
Gera `data/processed/vins_agregados.csv`.

### 2. Segmentação (Clustering)
```bash
python -m src.pipeline.clustering_real
```
Gera `data/processed/cluster_labels.csv`.

### 3. Treinamento
```bash
python -m src.pipeline.train_churn_real
python -m src.pipeline.train_classifier_real
```
Gera os arquivos `.joblib` em `models/`.

## API FastAPI

### Subir a API
```bash
export SECRET_KEY="sua-chave-secreta-de-pelo-menos-32-caracteres"
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Exemplo de Request (/predict)
```json
{
  "features": {
    "ano_modelo": 2023,
    "dias_ate_entrega": 11,
    "idade_veiculo_meses": 18.5,
    "modelo": "RANGER"
  },
  "modelo_veiculo": "Ranger"
}
```

## Deploy no Render

O projeto esta preparado para deploy como Web Service Python no Render via
`render.yaml`.

Configuracao usada:

- Build Command: `pip install --upgrade pip && pip install -r requirements-api.txt`
- Start Command: `python scripts/fetch_model_artifacts.py && uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
- Health Check Path: `/`
- Python: `3.11.11`

Variaveis criadas/configuradas no Render:

- `SECRET_KEY`: gerada pelo Render, obrigatoria para JWT
- `JWT_ISSUER`: `ford-vinguard-api`
- `JWT_AUDIENCE`: `ford-vinguard-api`
- `MODELS_DIR`: `models`
- `CHURN_MODEL_FILENAME`: `churn_pos_venda_rf_calibrated.joblib`
- `PERFIL_MODEL_FILENAME`: `segmento_pos_venda_classifier_experimental.joblib`
- `CHURN_MODEL_URL`: URL privada/publica para baixar o `.joblib` de churn
- `PERFIL_MODEL_URL`: URL privada/publica para baixar o `.joblib` de perfil
- `CHURN_MODEL_SHA256`: checksum esperado do modelo de churn
- `PERFIL_MODEL_SHA256`: checksum esperado do modelo de perfil

Observacao importante: `models/` e `data/` nao sao versionados. Por isso, o
deploy tem dois niveis:

- Sem URLs de modelo: `/` e `/docs` sobem para demonstrar a API; `/health`
  retorna `503 degraded` porque os artefatos ainda nao existem.
- Com `CHURN_MODEL_URL` e `PERFIL_MODEL_URL`: o script
  `scripts/fetch_model_artifacts.py` baixa os `.joblib` para `MODELS_DIR` antes
  do Uvicorn iniciar. Com os checksums corretos, `/health`, `/predict` e
  `/predict/batch` ficam prontos para demo.

As URLs dos artefatos devem ser configuradas no painel do Render ou preenchidas
no fluxo inicial do Blueprint. Nao commitar URLs assinadas, tokens ou datasets
reais no repositorio.

## Testes
```bash
pytest tests/ -v
```
