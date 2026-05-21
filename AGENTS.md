# Instrucoes para o Agente — Ford VinGuard (pos-venda com dados reais)

## Filosofia geral

- Solucao mais simples e direta. Funcoes em vez de classes.
- Sem Clean Architecture, DDD, Repository Pattern, ou camadas de abstracao.
- Sem logging framework, telemetria ou observabilidade alem do basico.

## Estilo Python

- Funcoes simples, scripts executaveis com if __name__ == '__main__'
- Type hints apenas onde ajudam
- Sem docstrings longas — comentarios curtos
- print() basta para scripts academicos

## Machine Learning — regras criticas

- Todo pre-processamento DENTRO de sklearn.Pipeline
- random_state=42 obrigatorio em tudo que aceita
- stratify=y no train_test_split de classificacao
- AUC-ROC como metrica principal para churn — nunca accuracy isolada
- class_weight='balanced' obrigatorio em classificadores

## Dataset (Caminho B — dados reais Ford)

Fonte unica: data/raw/vin_share_Desafio_02.xlsx (aba 'vin_share')
- 602.788 ordens de servico reais Ford Brasil 2020-2026
- 175.554 VINs unicos (anonimizados via hash)
- Granularidade: 1 linha = 1 evento de manutencao

## Problema correto

O projeto e POS-VENDA. O objetivo e prever risco de abandono da rede autorizada
Ford a partir do historico parcial de servicos/manutencoes por VIN.

Fluxo metodologico:
historico de servico ate uma data de corte -> features comportamentais ate corte
-> abandono futuro -> segmentacao comportamental pos-venda -> acao de retencao.

Nao tratar o projeto como previsao no ato da compra.

## Data Leakage temporal — regra absoluta

As features do X devem ser calculadas apenas com ServiceDate <= DATA_CORTE.
O target deve ser calculado apenas com ServiceDate > DATA_CORTE.

Nunca usar no X colunas pos-corte, futuro, target, churn ou voltou, por exemplo:
  churn_futuro_18m, voltou_pos_corte, primeiro_servico_pos_corte,
  ultimo_servico_pos_corte, qtd_servicos_pos_corte

Features comportamentais permitidas quando calculadas ate a data de corte:
  qtde_revisoes_ate_corte, meses_desde_ultimo_servico_ate_corte,
  meses_relacionamento_ate_corte, n_dealers_usados_ate_corte,
  km_max_ate_corte, pct_agenda_ate_corte,
  intervalo_medio_revisoes_dias_ate_corte, dias_ate_primeira_revisao,
  idade_veiculo_meses_ate_corte, modelo, ano_modelo

## Targets

- churn_futuro_18m: 1 se VIN nao tiver servico em (DATA_CORTE, DATA_CORTE + 18 meses], senao 0
- segmento_pos_venda: derivado por K-Means nas features comportamentais ate corte

## Pipeline (ordem de execucao)

1. python -m src.pipeline.feature_engineering_real  # snapshot temporal por VIN
2. python -m src.pipeline.clustering_real           # segmentacao pos-venda
3. python -m src.pipeline.train_churn_real          # risco de abandono pos-venda

## Estrutura

- src/pipeline/feature_engineering_real.py : snapshot pos-venda por VIN
- src/pipeline/clustering_real.py          : K-Means + segmentos pos-venda neutros
- src/pipeline/train_churn_real.py         : RF calibrado para abandono pos-venda
- data/raw/vin_share_Desafio_02.xlsx       : dataset real Ford
- data/processed/snapshots_pos_venda.csv    : 1 linha por VIN com features ate corte
- data/processed/dataset_churn_pos_venda.csv: dataset de treino de churn pos-venda
- data/processed/segmentos_pos_venda.csv    : VIN_Hash + segmento_pos_venda
- models/kmeans_segmentador_pos_venda.joblib: pipeline de segmentacao
- models/churn_pos_venda_rf_calibrated.joblib: modelo de churn pos-venda

## Graficos

- matplotlib.use('Agg') na primeira linha de scripts que geram graficos
- Salvar com plt.savefig(..., dpi=150, bbox_inches='tight') antes de plt.close()
- Sempre os.makedirs('reports', exist_ok=True) antes
