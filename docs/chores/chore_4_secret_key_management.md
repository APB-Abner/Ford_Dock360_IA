# PRD: Secret Key Configuration Management

## 1. Overview
The API currently loads the `SECRET_KEY` for JWT decoding by calling `os.environ.get("SECRET_KEY")` dynamically within the request context. 

## 2. Problem Statement
In `ford-ml-api/app/security/auth.py`, the `_secret_key()` function reads the environment variable during every request that requires authentication. If the environment variable is missing, it throws an `HTTPException(500)`. 
This approach is flawed because:
- It fails at runtime (during a request) rather than at startup (Fail Fast principle).
- It prevents easy caching and configuration validation.
- It relies on scattered `os.environ` lookups rather than a centralized, typed configuration schema.

## 3. Acceptance Criteria
- [ ] Replace the dynamic `os.environ.get` lookup in `auth.py` with a centralized Pydantic Settings configuration (`BaseSettings`).
- [ ] The API should fail to start if the `SECRET_KEY` is not provided in the environment.
- [ ] The JWT decoding function should securely access the cached `SECRET_KEY` from the settings instance.

## 4. Technical Details
- Utilize `pydantic-settings` (add to `requirements.txt` if necessary).
- Create a `config.py` in `ford-ml-api/app/` containing a `Settings(BaseSettings)` class.
- Instantiate the settings once and import this instance into `auth.py`.
