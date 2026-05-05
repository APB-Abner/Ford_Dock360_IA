#!/usr/bin/env bash
# Aplicar todas as correcoes identificadas na auditoria
# Executar de dentro de /workspace: bash scripts/apply_fixes.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "Raiz do projeto: $ROOT"
echo ""

# US-001: Remover .codex/ do historico git
echo "=== US-001: Removendo .codex/ do git ==="
if git ls-files --error-unmatch .codex/ >/dev/null 2>&1; then
  git rm -r --cached .codex/
  echo "  .codex/ removido do rastreamento git"
else
  echo "  .codex/ ja nao esta rastreado"
fi

# US-002: Permissao /predict de viewer para analyst
echo ""
echo "=== US-002: Corrigindo permissao /predict ==="
FILE="ford-ml-api/app/routers/predict.py"
if grep -q "require_role(RoleEnum.viewer)" "$FILE"; then
  sed -i 's/require_role(RoleEnum\.viewer)/require_role(RoleEnum.analyst)/g' "$FILE"
  echo "  Corrigido: viewer -> analyst"
else
  echo "  Ja corrigido"
fi
python3 -m py_compile "$FILE" && echo "  py_compile OK"

# US-003: ValueError -> aviso
echo ""
echo "=== US-003: Corrigindo ValueError em visualizations.py ==="
python3 - << 'PYEOF'
path = "src/pipeline/visualizations.py"
with open(path, "r") as f:
    content = f.read()
old = '        raise ValueError(f"plano_manutencao nao ficou no top 3: {top3}")'
new = '        print(f"AVISO: plano_manutencao nao ficou no top 3: {top3}")'
if old in content:
    with open(path, "w") as f:
        f.write(content.replace(old, new))
    print("  ValueError -> print de aviso OK")
else:
    print("  Ja corrigido ou nao encontrado")
PYEOF
python3 -m py_compile src/pipeline/visualizations.py && echo "  py_compile OK"

# US-005: Comentario n_estimators
echo ""
echo "=== US-005: Comentario n_estimators em mlflow_tracking.py ==="
python3 - << 'PYEOF'
path = "src/pipeline/mlflow_tracking.py"
with open(path, "r") as f:
    content = f.read()
marker = "# Hiperparametros reduzidos intencionalmente"
old = '        "RandomForest": RandomForestClassifier(\n            n_estimators=50,'
new = '        # Hiperparametros reduzidos intencionalmente para velocidade de tracking.\n        # Modelo de producao usa n_estimators=200 (ver train_classifier.py)\n        "RandomForest": RandomForestClassifier(\n            n_estimators=50,'
if marker not in content and old in content:
    with open(path, "w") as f:
        f.write(content.replace(old, new))
    print("  Comentario adicionado")
else:
    print("  Ja presente ou nao encontrado")
PYEOF
python3 -m py_compile src/pipeline/mlflow_tracking.py && echo "  py_compile OK"

# US-006: complaints_loader.py
echo ""
echo "=== US-006: Criando src/pipeline/complaints_loader.py ==="
cat > src/pipeline/complaints_loader.py << 'PYEOF'
from pathlib import Path
import pandas as pd

_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "ford_complaints_top3_por_modelo.csv"
_cache = None


def load_complaints_top3(csv_path=None):
    global _cache
    if _cache is not None:
        return _cache
    path = Path(csv_path) if csv_path else _CSV_PATH
    if not path.exists():
        return pd.DataFrame(columns=["modelo", "rank", "componente", "total_reclamacoes"])
    _cache = pd.read_csv(path)
    return _cache


def get_top3_por_modelo(modelo):
    if not modelo:
        return []
    df = load_complaints_top3()
    if df.empty:
        return []
    match = df[df["modelo"].str.lower() == modelo.strip().lower()]
    if match.empty:
        return []
    return match.sort_values("rank")[["rank", "componente", "total_reclamacoes"]].to_dict(orient="records")
PYEOF
python3 -m py_compile src/pipeline/complaints_loader.py && echo "  py_compile OK"

# US-007/008: schemas.py
echo ""
echo "=== US-007/008: Atualizando schemas.py ==="
cat > ford-ml-api/app/models/schemas.py << 'PYEOF'
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class RoleEnum(str, Enum):
    viewer = "viewer"
    analyst = "analyst"
    admin = "admin"


class ChurnLabelEnum(str, Enum):
    churn = "churn"
    no_churn = "no_churn"


class RiskLevelEnum(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class PredictRequest(BaseModel):
    features: dict
    modelo_veiculo: Optional[str] = None


class PredictResponse(BaseModel):
    prediction: ChurnLabelEnum
    churn_probability: float
    risk_level: RiskLevelEnum
    perfil_previsto: Optional[str] = None
    probabilidades_perfil: Optional[dict] = None
    acao_recomendada: Optional[str] = None
    historico_problemas: Optional[list] = None


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest]
PYEOF
python3 -m py_compile ford-ml-api/app/models/schemas.py && echo "  schemas.py OK"

# US-007/008: predictor.py
echo ""
echo "=== US-007/008: Atualizando predictor.py ==="
cat > ford-ml-api/app/services/predictor.py << 'PYEOF'
import os
import sys
from pathlib import Path
import joblib
import pandas as pd
from fastapi import HTTPException
from app.models.schemas import ChurnLabelEnum, PredictResponse, RiskLevelEnum

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from src.pipeline.complaints_loader import get_top3_por_modelo
    _COMPLAINTS_OK = True
except ImportError:
    _COMPLAINTS_OK = False

_ACOES = {
    "abandono": "Ativar Pulse Loop imediato. Oferecer plano de manutencao com desconto antes da primeira revisao.",
    "esquecido": "Configurar lembrete 45 dias antes do vencimento. Verificar disponibilidade de agenda.",
    "economico": "Apresentar tabela comparativa custo oficial vs externo. Oferta de pacote economico com preco fixo.",
    "fiel": "Nenhuma acao ativa. Registrar para agradecimento apos proxima revisao.",
}


class PredictorService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model_churn = None
            cls._instance.model_perfil = None
            cls._instance.feature_names = None
        return cls._instance

    def _model_path(self, filename):
        env_key = "MODEL_PATH_" + filename.upper().replace(".", "_").replace("-", "_")
        return Path(os.environ.get(env_key, _PROJECT_ROOT / "models" / filename))

    def _load_churn(self):
        if self.model_churn is None:
            path = self._model_path("churn_rf_calibrated.joblib")
            if not path.exists():
                raise HTTPException(status_code=503, detail=f"Modelo churn nao encontrado: {path}")
            self.model_churn = joblib.load(path)
            self.feature_names = list(getattr(self.model_churn, "feature_names_in_", []))
        return self.model_churn

    def _load_perfil(self):
        if self.model_perfil is None:
            path = self._model_path("perfil_rf_classifier.joblib")
            if path.exists():
                self.model_perfil = joblib.load(path)
        return self.model_perfil

    def _build_frame(self, features):
        self._load_churn()
        if not self.feature_names:
            estimator = getattr(self.model_churn, "estimator", None)
            preprocessor = getattr(estimator, "named_steps", {}).get("preprocessor")
            self.feature_names = list(getattr(preprocessor, "feature_names_in_", []))
        if self.feature_names:
            missing = [c for c in self.feature_names if c not in features]
            if missing:
                raise HTTPException(status_code=422, detail={"missing_features": missing})
            features = {c: features[c] for c in self.feature_names}
        return pd.DataFrame([features])

    def _predict_churn(self, frame):
        model = self._load_churn()
        prob = float(model.predict_proba(frame)[0][1]) if hasattr(model, "predict_proba") else float(model.predict(frame)[0])
        label = ChurnLabelEnum.churn if prob >= 0.5 else ChurnLabelEnum.no_churn
        risk = RiskLevelEnum.high if prob >= 0.70 else (RiskLevelEnum.medium if prob >= 0.40 else RiskLevelEnum.low)
        return label, round(prob, 6), risk

    def _predict_perfil(self, frame):
        model = self._load_perfil()
        if model is None:
            return None, None
        try:
            probs = model.predict_proba(frame)[0]
            classes = model.classes_
            perfil = str(classes[probs.argmax()])
            return perfil, {str(c): round(float(p), 4) for c, p in zip(classes, probs)}
        except Exception:
            return None, None

    def predict(self, features, modelo_veiculo=None):
        frame = self._build_frame(features)
        label, prob_churn, risk = self._predict_churn(frame)
        perfil, prob_perfil = self._predict_perfil(frame)
        historico = []
        if _COMPLAINTS_OK and modelo_veiculo:
            try:
                historico = get_top3_por_modelo(modelo_veiculo)
            except Exception:
                pass
        return PredictResponse(
            prediction=label,
            churn_probability=prob_churn,
            risk_level=risk,
            perfil_previsto=perfil,
            probabilidades_perfil=prob_perfil,
            acao_recomendada=_ACOES.get(perfil) if perfil else None,
            historico_problemas=historico,
        )


predictor_service = PredictorService()
PYEOF
python3 -m py_compile ford-ml-api/app/services/predictor.py && echo "  predictor.py OK"

# Router
echo ""
echo "=== Atualizando router predict.py ==="
cat > ford-ml-api/app/routers/predict.py << 'PYEOF'
from fastapi import APIRouter, Depends
from app.models.schemas import BatchPredictRequest, PredictRequest, PredictResponse, RoleEnum
from app.security.auth import require_role
from app.services.predictor import predictor_service

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, role: str = Depends(require_role(RoleEnum.analyst))):
    return predictor_service.predict(request.features, modelo_veiculo=request.modelo_veiculo)


@router.post("/predict/batch", response_model=list[PredictResponse])
def predict_batch(request: BatchPredictRequest, role: str = Depends(require_role(RoleEnum.analyst))):
    return [predictor_service.predict(i.features, modelo_veiculo=i.modelo_veiculo) for i in request.items]
PYEOF
python3 -m py_compile ford-ml-api/app/routers/predict.py && echo "  router OK"

# US-009: AGENTS.md
echo ""
echo "=== US-009: Atualizando AGENTS.md ==="
cat >> AGENTS.md << 'MDEOF'

## Complaints e Ficha de Abordagem

- data/raw/ford_complaints_top3_por_modelo.csv — top 3 problemas historicos por modelo Ford
  - Colunas: modelo, rank, componente, total_reclamacoes
  - Fonte: reclamacoes NHTSA 2010-2024 filtradas para modelos Brasil
  - Nao e dado de ML — e contexto para a Ficha de Abordagem

- src/pipeline/complaints_loader.py
  - load_complaints_top3() -> DataFrame completo
  - get_top3_por_modelo(modelo) -> list[dict] com rank, componente, total_reclamacoes
  - Match case-insensitive. Retorna [] se nao encontrado ou CSV ausente.

- PredictResponse campos:
  - prediction, churn_probability, risk_level (churn — obrigatorio)
  - perfil_previsto, probabilidades_perfil, acao_recomendada (perfil — opcional)
  - historico_problemas (ficha de abordagem — opcional)

- models/: churn_rf_calibrated.joblib (obrigatorio), perfil_rf_classifier.joblib (opcional)
- Endpoints /predict e /predict/batch exigem role analyst ou admin
MDEOF
echo "  AGENTS.md OK"

# Validacao final
echo ""
echo "=== Validacao final ==="
python3 -m py_compile \
  src/pipeline/complaints_loader.py \
  src/pipeline/visualizations.py \
  src/pipeline/mlflow_tracking.py \
  ford-ml-api/app/models/schemas.py \
  ford-ml-api/app/services/predictor.py \
  ford-ml-api/app/routers/predict.py \
  && echo "  Todos os arquivos OK"

echo ""
echo "=== Concluido ==="
echo "  Falta apenas: mkdir -p data/raw && cp ford_complaints_top3_por_modelo.csv data/raw/"
