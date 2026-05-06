# Relatorio de Cybersecurity

## US-019 - SECRET_KEY no startup

A API valida a `SECRET_KEY` durante o startup. A chave precisa ter pelo menos
32 caracteres e nao pode ser o valor de exemplo `changeme-local-only`.

Se a validacao falhar, o processo encerra com `SystemExit` e uma mensagem clara
para impedir que tokens JWT HS256 sejam aceitos com uma chave fraca ou copiada
do `.env.example`.

## Roadmap de arquitetura alvo

Para producao, a arquitetura alvo deve remover a responsabilidade de emitir e
gerenciar credenciais da API academica:

- Autenticacao federada via OIDC, com emissor externo confiavel e validacao de
  `iss`, `aud`, `sub` e `exp` nos tokens.
- Rotacao e armazenamento de segredos em Vault, evitando segredos persistidos em
  arquivos `.env` nos ambientes de homologacao e producao.
- Uso de chaves assimetricas publicadas por JWKS no provedor OIDC, reduzindo o
  risco operacional de uma `SECRET_KEY` compartilhada para HS256.
