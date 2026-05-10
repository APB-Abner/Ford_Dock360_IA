"""Gera um JWT local de demo para chamar /predict.

Use as mesmas variaveis SECRET_KEY, JWT_ISSUER e JWT_AUDIENCE do ambiente da API.
Nao grave o token em arquivos versionados.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jose import jwt

from src.api.config import settings
from src.api.security.auth import ALGORITHM


def parse_args():
    parser = argparse.ArgumentParser(description="Gera JWT Bearer para demo da API ML.")
    parser.add_argument("--subject", default=settings.DEMO_TOKEN_SUBJECT)
    parser.add_argument("--role", default="analyst", choices=["viewer", "analyst", "admin"])
    parser.add_argument("--minutes", type=int, default=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = {
        "sub": args.subject,
        "role": args.role,
        "iss": settings.JWT_ISSUER,
        "aud": settings.JWT_AUDIENCE,
        "exp": datetime.utcnow() + timedelta(minutes=args.minutes),
    }
    print(jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGORITHM))


if __name__ == "__main__":
    main()
