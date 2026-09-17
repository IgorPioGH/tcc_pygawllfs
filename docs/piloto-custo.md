# Piloto cronometrado do PyGAwLLfs

**Data da medição:** 13 de setembro de 2026
**Objetivo:** estimar o custo computacional de uma execução completo do
PyGAwLLfs antes de comprometer o cronograma experimental


## Configuração Medida
**Hardware:** Intel Core i7, 16 núcleos lógicos (12 alocados ao pool de 
avaliação paralela, conforme `CPU_USAGE_FRACTION = 0.8`),
NVIDIA RTX 4050 (não utilizada - RandomForest CPU-bound)

**Dados:** subamostra estratificada de 50.000 voos do VRA 2024 - 2025,
semente 42, taxa de atraso de 18.5%. **20 atributos rústicos** (derivados
temporais e codificação ordinal de categóricas), não são features definitivas do estudo, apenas um conjunto com número e perfil de custo representativos.

**Algoritmo genético:** população 50, 10 gerações, taxa de cruzamento 0.8,
probabilidade de mutação 1/N = 0.05, `tau_reset` 50, linkage learning ativo.

**Modelo:** RandomForest com `n_estimators=100` e `max_depth=10` (ambos
fixos no wrapper do repositório original), `n_jobs=1` por processo, o paralelismo
acontece no nível de população, não dentro do modelo.

**Métodos comparativos desativados** (--no_compare) é para isolar o custo do GA.

## Resultados

| Medida | Valor |
|---|---|
| Tempo do GA (execução 1) | 108,08 s |
| Tempo do GA (execução 2) | 107,70 s |
| Tempo total de parede (inclui I/O) | 1 min 56 s |
| Tempo de CPU acumulado | 13 min 40 s |
| Paralelismo efetivo | 7,1× |
| Pico de RAM | 256 MB |
| Custo estimado por treino de modelo | ~1,5 s |

As duas execuções independentes diferem em 0.4%, indicando medição estável.

## Extrapolação

| Cenário | Estimativa |
|---|---|
| 1 execução, 100 gerações | **~18 min** |
| 5 sementes × RandomForest | ~1,5 h |
| Protocolo completo (3 modelos × 5 sementes) | ~3 h |

## Decisão

Abaixo do limiar de 20 minutos por execução: Não foi necessário acionar nenhuma das 
mitigações previstas (reduzir a subamostra para 25.000, reduzir o número de árvores, ou cortar o XGBoost).

A estimativa preliminar do projeto era de 2.5~4.5 horas por execução e 2~3 dias de
máquina para o protocolo completo. A diferença se explica por três fatores verificáveis:
(i) o limite `max_depth=10` no RandomForest, ausente da estimativa original;
(ii) paralelismo efetivo de 7.1x no pool de processos;
(iii) custo por treino no piso da faixa estimada.

## Ressalvas
1. **Apenas o RandomForest foi medido**
2. **O piloto usou 20 atributos, o dataset final pode chegar a 40**
3. **Não houve teste de throttling térmico**
4. **O número de treinos de modelo não foi instrumentado**
5. **A medição usou o PyGAwLLfs sem as correções metodológicas**

## Como reproduzir
1. Gerar o arquivo de entrada: `python src/data/exportar_dat.py` 
    (produz `PyGAwLLfs/data/vra_piloto.dat`)
2. Registrar o dataset em `datasets_config.py` e os 20 nomes de variável,
    na mesma ordem das colunas, em `variables_config.py`
3. Executar, de dentro de `PyGAwLLfs/`:

    time python src/main.py --datasets vra_piloto --models rf --max_gen 10 --n_runs 1 --pop_size 50 --no_compare

Os tempos ficam registrados em `PyGAwLLfs/results/time_statistics.db`
(tabela `run_times`, coluna `time_gawll`).