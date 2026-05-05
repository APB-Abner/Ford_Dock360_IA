# PRD: Module Resolution and Package Structure Fix

## 1. Overview
The current project suffers from module resolution errors and relies on an anti-pattern (runtime modification of `sys.path`) to allow the API to import components from the ML pipeline.

## 2. Problem Statement
In `ford-ml-api/app/services/predictor.py`, the code dynamically appends the project root to `sys.path` (`sys.path.insert(0, str(_PROJECT_ROOT))`) so it can perform `from src.pipeline.complaints_loader import get_top3_por_modelo`. 
This is an anti-pattern that leads to:
- Static analysis tools like `mypy` failing to resolve imports (`Cannot find implementation or library stub for module`).
- The same source file being loaded under different namespaces (e.g., `src.pipeline.complaints_loader` and `complaints_loader`), causing `mypy` and potential runtime collisions.

## 3. Acceptance Criteria
- [ ] Remove `sys.path.insert` from `predictor.py` and any other application code.
- [ ] Add empty `__init__.py` files to `src/`, `src/pipeline/`, and appropriate directories in `ford-ml-api/app/` to establish standard Python packages.
- [ ] Ensure that `mypy .` runs without complaining about "Source file found twice under different module names".
- [ ] Ensure the API can import `src.pipeline` modules without path hacking (this may require setting `PYTHONPATH` during execution or converting `src` and `ford-ml-api` into installed packages via `pip install -e .`).

## 4. Technical Details
- File to modify: `/workspace/ford-ml-api/app/services/predictor.py`
- Directories requiring `__init__.py`: `src/`, `src/pipeline/`, `ford-ml-api/app/`, `ford-ml-api/app/routers/`, `ford-ml-api/app/models/`, `ford-ml-api/app/security/`, `ford-ml-api/app/services/`.
- Ensure correct `PYTHONPATH` handling in deployment/execution scripts instead of relying on inline hacks.
