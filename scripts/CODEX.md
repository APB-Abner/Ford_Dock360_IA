# Instrucoes para o Codex

Voce e um agente autonomo de desenvolvimento. Sua tarefa e implementar a user story descrita acima.

## Regras

1. Implemente exatamente o que os acceptance criteria pedem
2. Crie todos os arquivos necessarios no repositorio
3. Use Python 3.11, scikit-learn 1.4, pandas 2.2, mlflow 2.11, fastapi 0.111
4. Codigo deve ser funcional e executavel
5. Nao adicione dependencias fora do requirements.txt
6. Ao finalizar, responda APENAS com um resumo do que foi criado, sem comentarios extras

## Estrutura do projeto

- src/pipeline/ : scripts Python do pipeline de ML
- notebooks/ : notebooks Jupyter
- data/raw/ : dados brutos CSV
- data/processed/ : dados processados
- models/ : modelos serializados
- reports/ : graficos e relatorios
- tests/ : testes unitarios
- ford-ml-api/ : API FastAPI
