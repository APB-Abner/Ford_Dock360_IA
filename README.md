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

O modelo de produção publicado usa features de snapshot calculadas somente até
`DATA_CORTE`, sem usar colunas futuras ou target:
`ano_modelo`, `qtde_revisoes_ate_corte`,
`meses_desde_ultimo_servico_ate_corte`, `meses_relacionamento_ate_corte`,
`n_dealers_usados_ate_corte`, `km_max_ate_corte`, `pct_agenda_ate_corte`,
`intervalo_medio_revisoes_dias_ate_corte`, `dias_ate_primeira_revisao`,
`idade_veiculo_meses_ate_corte`, `modelo`.

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
    "ano_modelo": 2020,
    "qtde_revisoes_ate_corte": 2,
    "meses_desde_ultimo_servico_ate_corte": 14.2,
    "meses_relacionamento_ate_corte": 48.0,
    "n_dealers_usados_ate_corte": 1,
    "km_max_ate_corte": 48200,
    "pct_agenda_ate_corte": 0.65,
    "intervalo_medio_revisoes_dias_ate_corte": 220.0,
    "dias_ate_primeira_revisao": 180,
    "idade_veiculo_meses_ate_corte": 54.0,
    "modelo": "KA"
  },
  "modelo_veiculo": "Ka"
}
```

### Token de demo/servico

Os endpoints `/predict` e `/predict/batch` exigem Bearer JWT com role
`analyst` ou `admin`.

Para gerar um token local usando o mesmo `SECRET_KEY` da API:

```bash
python scripts/create_demo_token.py --role analyst
```

Para gerar token no ambiente publicado, configure `DEMO_TOKEN_SECRET` no Render
e chame:

```bash
curl -X POST "https://ford-vinguard-api.onrender.com/auth/demo-token?role=analyst" \
  -H "X-Demo-Token-Secret: <DEMO_TOKEN_SECRET>"
```

Use o `access_token` retornado como `Authorization: Bearer <token>` na FastAPI
ou como header `X-ML-Demo-Token` no BFF Java.

## Deploy no Render

O projeto esta preparado para deploy como Web Service Python no Render via
`render.yaml`.

Configuracao usada:

- Build Command: `pip install --upgrade pip && pip install -r requirements-api.txt && python scripts/fetch_model_artifacts.py`
- Start Command: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
- Health Check Path: `/`
- Python: `3.11.11`

Variaveis criadas/configuradas no Render:

- `SECRET_KEY`: gerada pelo Render, obrigatoria para JWT
- `JWT_ISSUER`: `ford-vinguard-api`
- `JWT_AUDIENCE`: `ford-vinguard-api`
- `ACCESS_TOKEN_EXPIRE_MINUTES`: validade dos tokens de demo/servico
- `DEMO_TOKEN_SECRET`: segredo opcional para habilitar `/auth/demo-token`
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

## Deploy no Azure Container Apps via Dockerfile

Para evitar o build automatico da Azure/Oryx, o projeto tambem possui um
`Dockerfile` explicito na raiz. A imagem roda a FastAPI na porta `8000` por
padrao e respeita a variavel `PORT` quando a plataforma definir outro valor.

Fluxo do container:

```text
pip install -r requirements-api.txt
python scripts/fetch_model_artifacts.py
uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

Arquivos e pastas pesadas ou sensiveis ficam fora do contexto Docker via
`.dockerignore`, incluindo `.env`, `data/`, `models/`, `notebooks/`,
`reports/`, `tests/`, `__pycache__` e `.git`.

Variaveis obrigatorias/recomendadas no Azure Container Apps:

```env
PORT=8000
SECRET_KEY=<chave-com-pelo-menos-32-caracteres>
DEMO_TOKEN_SECRET=<segredo-para-gerar-token-demo>
MODELS_DIR=models
CHURN_MODEL_FILENAME=churn_pos_venda_rf_calibrated.joblib
PERFIL_MODEL_FILENAME=segmento_pos_venda_classifier_experimental.joblib
CHURN_MODEL_URL=<url-do-artefato-churn>
PERFIL_MODEL_URL=<url-do-artefato-perfil>
CHURN_MODEL_SHA256=<sha256-do-artefato-churn>
PERFIL_MODEL_SHA256=<sha256-do-artefato-perfil>
JWT_ISSUER=ford-vinguard-api
JWT_AUDIENCE=ford-vinguard-api
```

Exemplo de build local:

```bash
docker build -t ford-vinguard-ml-api .
```

Exemplo de execucao local sem gravar segredos no repositorio:

```bash
docker run --rm -p 8000:8000 \
  -e PORT=8000 \
  -e SECRET_KEY="troque-por-uma-chave-com-32-caracteres" \
  -e DEMO_TOKEN_SECRET="troque-por-um-segredo-de-demo" \
  -e MODELS_DIR=models \
  -e CHURN_MODEL_FILENAME=churn_pos_venda_rf_calibrated.joblib \
  -e PERFIL_MODEL_FILENAME=segmento_pos_venda_classifier_experimental.joblib \
  -e CHURN_MODEL_URL="<url-do-artefato-churn>" \
  -e PERFIL_MODEL_URL="<url-do-artefato-perfil>" \
  -e CHURN_MODEL_SHA256="<sha256-do-artefato-churn>" \
  -e PERFIL_MODEL_SHA256="<sha256-do-artefato-perfil>" \
  -e JWT_ISSUER=ford-vinguard-api \
  -e JWT_AUDIENCE=ford-vinguard-api \
  ford-vinguard-ml-api
```

No Azure Container Apps, configure ingress HTTP externo para a porta alvo
`8000`. O endpoint `/` valida liveness, `/health` valida a disponibilidade dos
modelos e `/docs` abre o Swagger.

Nao commitar `.env`, datasets reais, arquivos em `data/raw`,
`data/processed`, `models/*.joblib`, URLs assinadas ou segredos usados no Azure.

## Testes
```bash
pytest tests/ -v
```
