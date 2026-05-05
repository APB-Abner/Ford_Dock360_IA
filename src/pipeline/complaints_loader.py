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
