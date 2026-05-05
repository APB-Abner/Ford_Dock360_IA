# Instrucoes para o Agente

## Filosofia geral

- Prefira sempre a solucao mais simples e direta
- Menos arquivos e melhor. Menos linhas e melhor
- Nao use Clean Architecture, DDD, CQRS, Repository Pattern ou similares
- Sem camadas extras de abstracao (services, repositories, factories) sem necessidade clara
- Se funciona e e legivel, esta bom. Nao refatore sem motivo
- Nao adicione logging framework, telemetria ou observabilidade a menos que pedido

## Estilo Python

- Funcoes simples no lugar de classes sempre que possivel
- Scripts executaveis com if __name__ == '__main__' em vez de modulos importaveis
- Sem type hints excessivos — apenas onde ajudam a entender
- Sem docstrings longas — comentarios curtos e diretos
- Sem __init__.py desnecessarios
- Sem **kwargs e configuracao dinamica — parametros explicitos sao mais auditaveis
- print() basta para scripts academicos — sem loguru, structlog ou logging framework

## Machine Learning — regras criticas

- Todo pre-processamento DEVE estar dentro de um sklearn.Pipeline
  - Nunca fite o scaler fora do pipeline — causa data leakage silencioso
  - O pipeline garante que treino e inferencia usam a mesma transformacao
- random_state=42 e obrigatorio em tudo que tem aleatoriedade (KMeans, train_test_split, RandomForest, etc.)
- Sempre use stratify=y no train_test_split quando o problema for classificacao
- Nunca use accuracy como metrica principal em classes desbalanceadas — use F1 Macro ou AUC-ROC
- class_weight='balanced' e obrigatorio em todos os classificadores deste projeto

## Data Leakage — regra absoluta deste projeto

As seguintes colunas existem na Base 1 e JAMAIS podem aparecer no X de treino ou teste do classificador:
fez_primeira_revisao_rede, meses_ate_primeira_revisao, perdeu_primeira_revisao,
voltou_tarde_revoltado, trouxe_oleo_externo, pede_desconto_revisao,
sensibilidade_desconto_pos, qtde_revisoes_24m, share_revisoes_rede_24m,
gasto_manutencao_rede_24m, satisfacao_marca_24m, churn_rede_24m

Se qualquer dessas colunas aparecer no X, o modelo esta invalido.
Sempre implemente a funcao check_leakage() para verificar isso automaticamente.

## Graficos e visualizacoes

- matplotlib.use('Agg') DEVE ser a primeira linha de qualquer script que gera graficos
  - Sem isso o script quebra no container por falta de display
- Sempre salve com plt.savefig('reports/nome.png', dpi=150, bbox_inches='tight') antes de plt.show()
- Sempre crie o diretorio reports/ com os.makedirs('reports', exist_ok=True) antes de salvar
- Feche cada figura com plt.close() para liberar memoria em loops

## Estrutura do projeto Ford VinGuard

- src/pipeline/ : scripts Python do pipeline
- notebooks/ : notebooks Jupyter
- data/raw/ : CSVs brutos
- data/processed/ : CSVs processados (ex: cluster_labels.csv)
- models/ : modelos serializados com joblib
- reports/ : graficos PNG e relatorios
- tests/ : testes unitarios simples
- ford-ml-api/ : API FastAPI

## Dependencias disponiveis

scikit-learn==1.4.2, pandas==2.2.2, numpy==1.26.4, matplotlib==3.8.4,
seaborn==0.13.2, mlflow==2.11.3, fastapi==0.111.0, uvicorn==0.29.0,
pydantic==2.7.1, python-jose==3.3.0, joblib==1.3.2

Nao adicione dependencias fora dessa lista sem perguntar.
