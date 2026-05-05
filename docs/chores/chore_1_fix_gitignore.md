# PRD: Fix Gitignore Overreach

## 1. Overview
The global `.gitignore` file currently includes a rule `models/` which broadly ignores any directory named `models` anywhere in the project hierarchy. This mistakenly ignores critical source code located in `ford-ml-api/app/models/` (such as `schemas.py`).

## 2. Problem Statement
Because `ford-ml-api/app/models/` is ignored, changes to database or Pydantic models in the API are not tracked by source control. This is a critical structural issue that could lead to lost work or deployed systems breaking due to missing schema files. Furthermore, this causes tooling (like file reading or `glob` matching) to ignore these paths.

## 3. Acceptance Criteria
- [ ] The global `.gitignore` is updated to scope the `models` ignore rule to only the root directory (i.e., `/models/`).
- [ ] The API models directory (`ford-ml-api/app/models/`) is properly tracked by git.
- [ ] No extraneous machine learning binary files (e.g., `.pkl`, `.joblib`) are accidentally tracked as a result of this change.

## 4. Technical Details
- File to modify: `/workspace/.gitignore`
- Current rule: `models/`
- Target rule: `/models/`
