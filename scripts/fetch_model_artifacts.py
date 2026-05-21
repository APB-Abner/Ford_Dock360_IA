"""Baixa modelos para o deploy do Render quando URLs forem configuradas."""

import hashlib
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
CHUNK_SIZE = 1024 * 1024 * 8


def _model_specs():
    return [
        {
            "label": "churn",
            "filename": os.getenv("CHURN_MODEL_FILENAME", "churn_pos_venda_rf_calibrated.joblib"),
            "url": os.getenv("CHURN_MODEL_URL", ""),
            "sha256": os.getenv("CHURN_MODEL_SHA256", ""),
        },
        {
            "label": "perfil",
            "filename": os.getenv("PERFIL_MODEL_FILENAME", "kmeans_segmentador_pos_venda.joblib"),
            "url": os.getenv("PERFIL_MODEL_URL", ""),
            "sha256": os.getenv("PERFIL_MODEL_SHA256", ""),
        },
    ]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url, destination):
    tmp_path = destination.with_suffix(destination.suffix + ".partial")
    request = Request(url, headers={"User-Agent": "ford-vinguard-render-bootstrap"})

    with urlopen(request, timeout=120) as response, tmp_path.open("wb") as output:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            output.write(chunk)

    tmp_path.replace(destination)


def _ensure_artifact(spec):
    filename = spec["filename"]
    url = spec["url"]
    expected_sha = spec["sha256"].strip().lower()
    path = MODELS_DIR / filename

    if path.exists():
        if expected_sha and _sha256(path) != expected_sha:
            raise SystemExit(f"Checksum local invalido para {filename}")
        print(f"Artefato ja existe: {filename}")
        return

    if not url:
        print(f"Artefato sem URL configurada, pulando: {filename}")
        return

    print(f"Baixando artefato: {filename}")
    try:
        _download(url, path)
    except URLError as exc:
        raise SystemExit(f"Falha ao baixar {filename}: {exc}") from exc

    if expected_sha:
        actual_sha = _sha256(path)
        if actual_sha != expected_sha:
            path.unlink(missing_ok=True)
            raise SystemExit(f"Checksum baixado invalido para {filename}")
        path.with_suffix(".sha256").write_text(expected_sha + "\n", encoding="utf-8")
        print(f"Checksum validado: {filename}")
    else:
        print(f"Sem checksum configurado para {filename}; configure *_SHA256 para validar o artefato.")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for spec in _model_specs():
        _ensure_artifact(spec)


if __name__ == "__main__":
    main()
