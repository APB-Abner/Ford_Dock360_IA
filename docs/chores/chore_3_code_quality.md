# PRD: Code Quality and Linting Cleanup

## 1. Overview
The codebase currently contains several linting, formatting, and typing errors flagged by `ruff` and `mypy`. Maintaining code quality is critical for long-term readability and stability.

## 2. Problem Statement
Running `ruff check .` returns 12 errors, primarily concerning:
- Unused imports (`sys`, `numpy`) in the Jupyter notebooks.
- Imports not placed at the top of the file in Jupyter notebooks.
- Multiple statements on a single line (using semicolons) in `src/pipeline/clustering.py` and `src/pipeline/visualizations.py` (e.g., `import matplotlib; matplotlib.use("Agg")`).

Running `mypy .` fails due to:
- Missing library stubs for the `jose` package (`types-python-jose`).
- Untyped modules in `ford-ml-api/app` (partially caused by module resolution issues, see Chore 2).

## 3. Acceptance Criteria
- [ ] `ruff check .` passes with 0 errors. Unused imports are removed, module-level imports are moved to the top in notebooks (or suppressed if explicitly required mid-cell), and single-line multiple statements are broken into multiple lines.
- [ ] `types-python-jose` is added to `requirements-dev.txt` and installed.
- [ ] `mypy .` passes without missing stub errors or critical typing violations.

## 4. Technical Details
- Run `ruff check --fix .` to auto-resolve basic issues.
- Manually edit `src/pipeline/clustering.py` and `src/pipeline/visualizations.py` to fix `E702`.
- Install stub packages using `python3 -m pip install types-python-jose`.
