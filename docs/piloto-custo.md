# Custo computacional do PyGAwLLfs

Este documento tem duas medições. A primeira (5 out) é a definitiva: o
protocolo completo, rodado com o wrapper corrigido. A segunda (13 set) é o
piloto original, mantido como registro histórico de como o cronograma foi
dimensionado.

---

## 1. Protocolo completo medido (5 de outubro de 2026)

**O que rodou:** `python src/importance/gawll.py` (padrões), ou seja,
3 modelos × 5 sementes (42–46) × 100 gerações, população 50, `pc` 0,8,
`pm` = 1/N, `tau_reset` 50, linkage learning ativo. Wrapper corrigido
(fitness = F1 na validação, mutação e elitismo corrigidos; ver docs/metodologia.md §4.12).

**Dados:** subamostra estratificada de 50.000 voos (45.336 de treino,
4.664 de validação), **29 atributos finais** (dataset congelado da Etapa 5).

**Hardware e paralelismo:** Intel Core i7 (16 núcleos lógicos) + RTX 4050.
LogReg e RF: 12 processos via `joblib`, um modelo por processo com `n_jobs=1`.
XGBoost: um modelo por vez na GPU (`device="cuda"`).

**Hiperparâmetros:** os escolhidos na Etapa 6 (`results/tables/baselines.csv`):
LogReg `C=1.0`; RF `n_estimators=100, max_depth=10`; XGBoost `n_estimators=300,
max_depth=8`.

### Resultados (média das 5 sementes)

| Modelo | Tempo por execução | Faixa | Treinos | Avaliações pedidas | Economia do cache | Tempo efetivo por treino |
|---|---|---|---|---|---|---|
| LogReg | 5,4 min | 5,2–5,8 | 3.216 | 4.144 | 22% | 0,10 s |
| RF | 26,4 min | 24,3–27,7 | 4.041 | 4.971 | 19% | 0,39 s |
| XGBoost | 33,0 min | 32,1–34,2 | 4.400 | 5.321 | 17% | 0,45 s |

**Protocolo inteiro: 5,4 h** de parede (5 out, 16:56 → 22:20), sem nenhuma
interrupção e sem erro no log.

- "Treinos" são ajustes de modelo realmente feitos; "avaliações pedidas"
  incluem os acertos de cache (o linkage learning reavalia cromossomos que o
  GAwLL avalia de novo logo depois). O número de treinos é a moeda de custo
  independente da máquina (docs/metodologia.md §7).
- No RF, o tempo por execução caiu de 27,7 para 24,3 min ao longo das
  sementes: **nenhum sinal de thermal throttling** em 5,4 h contínuas.

### Comparação com as estimativas

| Fonte | Protocolo completo |
|---|---|
| Estimativa inicial (docs/metodologia.md §2) | 2–3 dias |
| Piloto de 13 set (só RF, 20 atributos, código com bugs) | ~3 h |
| **Medido em 5 out** | **5,4 h** |

O medido ficou acima do piloto por três motivos esperados: 29 atributos em vez
de 20; a mutação corrigida gera mais cromossomos novos (o piloto, com a
mutação inoperante, acertava muito mais o cache); e o XGBoost, que não tinha
sido medido, é o mais caro por execução porque roda em sequência na GPU.
Nenhum gatilho do §9 disparou (o limite era 30 min por piloto de 10 gerações).

---

## 2. Piloto original (13 de setembro de 2026)

**Objetivo:** estimar o custo computacional de uma execução completa do
PyGAwLLfs antes de comprometer o cronograma experimental.

### Configuração medida

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

### Resultados

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

### Extrapolação

| Cenário | Estimativa |
|---|---|
| 1 execução, 100 gerações | **~18 min** |
| 5 sementes × RandomForest | ~1,5 h |
| Protocolo completo (3 modelos × 5 sementes) | ~3 h |

### Decisão

Abaixo do limiar de 20 minutos por execução: Não foi necessário acionar nenhuma das 
mitigações previstas (reduzir a subamostra para 25.000, reduzir o número de árvores, ou cortar o XGBoost).

A estimativa preliminar do projeto era de 2.5~4.5 horas por execução e 2~3 dias de
máquina para o protocolo completo. A diferença se explica por três fatores verificáveis:
(i) o limite `max_depth=10` no RandomForest, ausente da estimativa original;
(ii) paralelismo efetivo de 7.1x no pool de processos;
(iii) custo por treino no piso da faixa estimada.

### Ressalvas (todas resolvidas pela medição de 5 out)

1. ~~Apenas o RandomForest foi medido~~ → os três modelos medidos
2. ~~O piloto usou 20 atributos, o dataset final pode chegar a 40~~ → 29 atributos finais
3. ~~Não houve teste de throttling térmico~~ → 5,4 h contínuas sem degradação
4. ~~O número de treinos de modelo não foi instrumentado~~ → `n_treinos` em cada checkpoint
5. ~~A medição usou o PyGAwLLfs sem as correções metodológicas~~ → wrapper corrigido.
   Atenção: este piloto rodou com os dois bugs de implementação (mutação
   inoperante e elitismo perdido), o que reduziu a diversidade da população e
   inflou os acertos de cache

### Como reproduzir este piloto

1. Gerar o arquivo de entrada: `python src/data/exportar_dat.py` 
    (produz `PyGAwLLfs/data/vra_piloto.dat`)
2. Registrar o dataset em `datasets_config.py` e os 20 nomes de variável,
    na mesma ordem das colunas, em `variables_config.py`
3. Executar, de dentro de `PyGAwLLfs/`:

    time python src/main.py --datasets vra_piloto --models rf --max_gen 10 --n_runs 1 --pop_size 50 --no_compare

Os tempos ficam registrados em `PyGAwLLfs/results/time_statistics.db`
(tabela `run_times`, coluna `time_gawll`).
