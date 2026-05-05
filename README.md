# Ford VinGuard

Projeto de machine learning para o pipeline Ford VinGuard.

## Setup

### Dev Container

No Codium/VS Code com a extensao Dev Containers instalada:

1. Abra o projeto.
2. Execute `Dev Containers: Rebuild Container Without Cache`.
3. Aguarde o `postCreateCommand` criar a estrutura local do projeto.

Portas expostas pelo container:

- `8000`: FastAPI
- `8888`: Jupyter
- `5000`: MLflow UI

Comandos uteis dentro do container:

```bash
python -m pip check
python -m py_compile src/pipeline/*.py ford-ml-api/app/**/*.py
```

Para subir a API localmente:

```bash
cd ford-ml-api
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Ambiente virtual local

Crie e ative um ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependencias:

```bash
pip install -r requirements.txt
```

## Estrutura

- `notebooks/`: notebooks Jupyter
- `src/pipeline/`: scripts Python do pipeline
- `data/raw/`: CSVs brutos
- `data/processed/`: dados processados
- `models/`: modelos serializados
- `reports/`: graficos e relatorios
- `tests/`: testes unitarios simples
- `ford-ml-api/app/`: aplicacao FastAPI
