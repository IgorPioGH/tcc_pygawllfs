# Metodologia do projeto — regras e decisões

Referência metodológica do TCC. **As citações "§x.y" no código e nos demais
documentos apontam para as seções deste arquivo.** Decisões provisórias e como
revertê-las estão em `docs/decisoes.md`; números da preparação dos dados em
`docs/preparacao-dados.md`.

---

## 1. Contexto

**Título:** Avaliação comparativa de métodos de importância e interação de
atributos em modelos de aprendizado de máquina aplicados à previsão de atrasos de voos.

**Autor:** Igor Pio — Bacharelado em Ciência da Computação, FFCLRP/USP.
**Orientador:** Prof. Dr. Renato Tinós.

**Pergunta de pesquisa:** como diferentes métodos de estimação de importância e
interação de atributos se comparam em termos de **concordância, estabilidade,
fidelidade e custo computacional** no problema de previsão de atrasos de voos?

**Método central:** PyGAwLLfs (Guimarães et al., 2026) — algoritmo genético
com linkage learning para seleção de atributos, que produz como subproduto um
vetor de importância e uma matriz de interação. Repositório:
https://github.com/dgabrl/PyGAwLLfs

**Métodos de comparação:** Permutation Feature Importance; SHAP (TreeSHAP,
§4.6); importâncias intrínsecas (coeficientes padronizados da regressão
logística, MDI do Random Forest, ganho do XGBoost); estatística H de Friedman
e Popescu (2008), para as interações.

**Dados:** VRA (Voo Regular Ativo) da ANAC, 2024–2025.
**Tarefa:** classificação binária — atraso na partida ≥ 15 min é a classe positiva.
**Modelos:** regressão logística, Random Forest e XGBoost.

## 2. Ambiente e custo

Notebook com Intel Core i7 (16 núcleos lógicos) e NVIDIA RTX 4050 (6 GB).
A GPU só acelera o XGBoost (`device="cuda"`, `tree_method="hist"`); o
scikit-learn é todo CPU. Execuções longas: notebook na tomada, suspensão
inibida (`systemd-inhibit`) e checkpoint a cada execução.

Custo medido (detalhes em `docs/piloto-custo.md`): o protocolo completo do
PyGAwLLfs (3 modelos × 5 sementes × 100 gerações, 29 atributos) levou **5,4 h**;
por semente, Permutation levou 17–28 s e TreeSHAP 2–140 s.

## 3. Decisões de escopo

1. **Subamostra estratificada de 50.000 voos para o PyGAwLLfs.** O artigo
   original rodou o dataset Boson com 52.035 exemplos e levou 7,2 h por
   execução; com a base completa, o protocolo seria inviável. A subamostra é
   estratificada por período (treino/validação) e classe, com semente fixa, e
   é usada **só** dentro do wrapper evolutivo. Modelos finais, métricas,
   demais métodos e fidelidade usam todas as linhas de cada período do split
   (§4.4) — nunca a base inteira sem particionar.
2. **Entre 20 e 40 atributos (N = 29).** O GA opera sobre N bits; one-hot de
   aeroportos geraria centenas de colunas (espaço de busca e matriz de
   interação intratáveis, e H de Friedman impossível). Aeroportos e empresas
   entram por target encoding (§4.10) e por porte. O artigo testou até N = 93.
3. **Dados meteorológicos ficam fora** — trabalho futuro.

## 4. Correções metodológicas

Correções em relação à proposta original e ao código do PyGAwLLfs.

### 4.1 Fitness na validação, não no teste

O Algoritmo 4 do artigo calcula o termo de desempenho do fitness no conjunto
de **teste**: a busca otimizaria sobre o próprio conjunto de avaliação. Aqui o
GA treina no treino, calcula o fitness na **validação**, e o teste só é usado
na avaliação final. Implementado em `src/importance/gawll.py`
(`AvaliadorFitness`); `tests/test_gawll.py` falha se a busca abrir o teste.
**Consequência:** o F1 de validação do melhor indivíduo é otimista (o
subconjunto foi escolhido olhando esse número) e não é reportado como
desempenho; o desempenho do subconjunto sai do teste (fidelidade).

### 4.2 F1 no lugar da acurácia

Atrasos são ~18% da base: responder sempre "não atrasa" dá ~76% de acurácia
com F1 = 0. O termo de desempenho do fitness é o **F1**; o termo de
parcimônia `(N − |x|)/N` e os pesos 0,98 / 0,02 do artigo são mantidos.

### 4.3 Só atributos conhecidos antes da partida

**Proibidos** (pós-hoc): `Partida Real`, `Chegada Real`, `Situação Voo`,
`Código Justificativa` — além do próprio alvo (`atraso_min`, `atrasado`).
**Permitidos:** empresa, aeroportos de origem e destino, horários previstos e
seus derivados (hora, dia da semana, mês, período, fim de semana, feriado e
véspera), duração prevista, tipo de linha e congestionamento programado
(voos previstos no mesmo aeroporto e hora). A guarda é
`src/data/clean.py::assert_sem_vazamento`. Propagação de atraso (voo anterior
da mesma aeronave) não foi implementada: usa dados realizados.
*Limitação conhecida:* o congestionamento foi contado sem os voos cancelados;
impacto medido desprezível (`docs/dicionario-features.md`, limitação 4).

### 4.4 Divisão temporal, nunca aleatória

Treino, validação e teste separados por datas absolutas, com o invariante
`max(treino) < min(validação)` e `max(validação) < min(teste)`. Cortes:
treino jan/2024–ago/2025, validação set–out/2025, teste nov–dez/2025.
Nenhum `train_test_split` embaralhado nos dados reais.

### 4.5 Métricas de comparação de rankings

Spearman, Kendall **tau-b** (trata empates), **RBO** com `p = 0,9` (pesa o
topo do ranking, onde a decisão acontece) e sobreposição top-k (k = 5, 10, 15).

### 4.6 SHAP: TreeSHAP

`shap.TreeExplainer` para RF e XGBoost (exato e rápido); nunca KernelSHAP
nesta escala. Na regressão logística, os coeficientes padronizados cumprem o
papel da importância intrínseca. SHAP global = `mean(|SHAP|)` sobre ~10 mil
linhas da **validação**.

### 4.7 H de Friedman: escopo restrito

O custo cresce com o quadrado do número de atributos. O H é calculado sobre a
**união** de (a) o top-15 de atributos por consenso entre os métodos de
importância e (b) os atributos presentes no grafo de interação reduzido do
PyGAwLLfs (5% das arestas mais fortes). Usar só (a) excluiria, por
construção, os atributos que o PyGAwLLfs aponta como mais interativos.
Background de ~500 linhas. Se o custo ficar inviável, corta-se por (a),
preservando sempre os atributos das 5 arestas mais fortes do PyGAwLLfs.
*Implementado em `src/interaction/friedman_h.py`:* a união ficou com 24 (RF)
e 25 (XGBoost) atributos, sem corte (~7 min no total); escala logit, 5
amostras de 500 linhas da validação; H não normalizado como versão principal.
A LogReg fica de fora: é aditiva no logit, então H = 0 por construção.

### 4.8 Comparação por rank, não por valor

As escalas dos métodos não se comparam (diferença de fitness, queda de F1,
unidades de probabilidade ou de margem). A comparação primária é sempre por
rank. Quando uma figura precisa de magnitude: valor absoluto + min-max, nunca
soma-1 (há valores com sinal); valores negativos da Permutation viram 0 nas
figuras ("sem evidência de importância"), mas o valor bruto fica nos CSVs.

### 4.9 Isolamento do conjunto de teste

Nenhum método de importância ou interação usa o teste em nenhuma etapa —
nem para estimar, nem para escolher parâmetros. Todos usam a validação. O
teste aparece em dois lugares só: a avaliação final dos modelos (Etapa 6) e
a fidelidade por retenção top-k (Etapa 12), sempre como última coisa.

### 4.10 Target encoding sem vazamento dentro do treino

No treino, o valor de cada linha vem de validação cruzada **out-of-fold**
(5 folds): a linha nunca usa o próprio rótulo. Depois, o mapa ajustado no
treino inteiro é aplicado sem reajuste em validação e teste. Suavização
`m = 20`. Testado por `features.py::verifica_vazamento`.

### 4.11 Comparar dentro do mesmo modelo antes de comparar entre modelos

Toda comparação de concordância, estabilidade ou fidelidade entre métodos
acontece primeiro **dentro do mesmo modelo** (ex.: MDI × Permutation × SHAP ×
PyGAwLLfs no RF). Comparar métodos aplicados a modelos diferentes mistura
divergência de método com divergência de modelo. A comparação entre modelos
é uma pergunta separada: os métodos concordam mais entre si no mesmo modelo
do que em modelos diferentes?

### 4.12 Bugs de implementação do PyGAwLLfs, corrigidos no wrapper

Dois comportamentos do código do repositório divergem do método descrito,
reproduzidos em `tests/test_gawll.py`:

1. **Mutação inoperante:** `Mutation.bit_flip_mutation` devolve um array novo
   que o `GAwLL` ignora; nenhum bit é invertido (nem no tau-reset).
   Correção: `MutacaoInPlace`.
2. **Elitismo perdido:** `GAwLL._apply_elitism` grava o elite na população
   antiga, descartada em seguida. Correção: `GAwLLCorrigido._apply_elitism`.

As correções são subclasses em `src/importance/gawll.py`; o código do
PyGAwLLfs não é editado. No texto: correções 4 e 5, ao lado de §4.1, §4.2 e §4.4.

## 5. Diferenciais

### Diferencial 1 — dataset sintético com gabarito

Sair de "os métodos concordam?" para "**qual método recupera a verdade?**".
Estrutura conhecida por construção (`src/synthetic/generate.py`): par
(x1, x2) com interação pura; x3, x4 informativas aditivas; x5, x6
redundantes (cópias correlacionadas de x3 e x4); x7–x10 ruído. O gabarito
tem **três categorias**: diretas, redundantes (proxies — apontá-las como
importantes não é erro) e ruído. Avalia-se a recuperação de importância
(relevantes × ruído), a distinção entre diretas e redundantes e a
recuperação dos pares de interação, com o **mesmo código** usado nos dados reais.

### Diferencial 4 — validação de domínio das interações

Os pares mais fortes do grafo de interação do PyGAwLLfs são confrontados com
estatística descritiva do VRA: a taxa de atraso no cruzamento das duas
variáveis difere do que os efeitos isolados de cada uma preveem? Cada par é
classificado como **corroborado descritivamente** ou **não corroborado** —
nunca "confirmado": é evidência observacional, e um confundidor omitido
poderia produzir o mesmo padrão. Os pares não corroborados também são discutidos.

**Fora do escopo:** fidelidade por remoção (ROAR), atributos-sombra
(Boruta), testes de Friedman-Nemenyi, fronteira de Pareto custo × informação.
A fidelidade por **retenção** (top-5/10/15) faz parte do objetivo original.

## 6. Estrutura do repositório

```
data/            raw/ e interim/ (não versionados), processed/ (dataset congelado)
src/data/        aquisição, limpeza, alvo, atributos, auditorias e figuras descritivas
src/models/      baselines e métricas
src/importance/  PyGAwLLfs corrigido e métodos de comparação
src/synthetic/   Diferencial 1
src/compare/     concordância, estabilidade e fidelidade
src/domain/      Diferencial 4
tests/           testes (scripts com assert)
results/         tabelas (CSV) e figuras
docs/            documentação
notebooks/       só exploração; a lógica fica em src/
```

## 7. Convenções

- Python 3.13, ambiente conda `tcc_pygawllfs`, dependências em `requirements.txt`.
- **Sementes fixas** em toda operação estocástica: `42, 43, 44, 45, 46`.
- Todo experimento salva importância, ranking, subconjunto selecionado,
  matriz de interação, tempo **e número de treinos de modelo** (o tempo
  depende da máquina; o número de treinos é a medida honesta de custo).
- **Formato padrão de importância:** CSV `feature,importance,rank,method,model,seed`
  (rank 1 = mais importante).
- Parquet para datasets intermediários; `logging` em vez de `print`; nenhuma
  lógica em notebook.

## 8. (Cronograma)

Mantido fora do repositório.

## 9. Riscos e gatilhos

| Risco | Sinal de alerta | Resposta |
|---|---|---|
| Vazamento silencioso | na validação: ROC-AUC > 0,90, **ou** F1 > 0,85, **ou** PR-AUC acima de 3× a taxa de atraso | parar e auditar os atributos contra o §4.3 antes de tocar o teste (`baselines.py::auditar_vazamento`) |
| Custo do GA | piloto de 10 gerações > 30 min | subamostra de 25 mil; 50 árvores; menos modelos (não foi necessário) |
| H de Friedman inviável | estimativa > 6 h | top-10 por importância (mantendo as arestas mais fortes do PyGAwLLfs) e background de 300 |
| Rankings sem concordância | Spearman ~0 entre todos | é resultado, não falha (o artigo observou o mesmo em grafos densos) — discutir |

## 10. Referências centrais

- GUIMARÃES, D. G.; CARVALHO, S. S.; NASCIMENTO, R. H.; TINÓS, R. Estimating
  variable importance and interaction in machine learning via genetic
  algorithms. In: *Intelligent Systems: BRACIS 2025*. Springer, 2026.
  p. 478–492. (LNAI 16179). DOI: 10.1007/978-3-032-15987-8_32.
- TINÓS, R.; PRZEWOZNICZEK, M. W.; WHITLEY, D.; CHICANO, F. Genetic algorithm
  with linkage learning. *GECCO 2023*, p. 981–989. DOI: 10.1145/3583131.3590349.
- FRIEDMAN, J. H.; POPESCU, B. E. Predictive learning via rule ensembles.
  *Annals of Applied Statistics*, v. 2, n. 3, p. 916–954, 2008. DOI: 10.1214/07-AOAS148.
- WEBBER, W.; MOFFAT, A.; ZOBEL, J. A similarity measure for indefinite
  rankings. *ACM Transactions on Information Systems*, v. 28, n. 4, p. 1–38,
  2010. DOI: 10.1145/1852102.1852106. (RBO)
- INGLIS, A.; PARNELL, A.; HURLEY, C. B. Visualizing variable importance and
  variable interaction effects in machine learning models. *JCGS*, v. 31,
  n. 3, p. 766–778, 2022. DOI: 10.1080/10618600.2021.2007935.
- BREIMAN, L. Random forests. *Machine Learning*, v. 45, p. 5–32, 2001.
- CHEN, T.; GUESTRIN, C. XGBoost: a scalable tree boosting system. *KDD 2016*, p. 785–794.
- ANAC. Histórico de voos: Voo Regular Ativo (VRA). Brasília, DF, 2026.

## 11. Pipeline — ordem de execução

Rodar da raiz do repositório, com o ambiente ativado. Cada script lê o que o
anterior produziu. O `main.py` da raiz roda tudo nesta ordem (`python main.py
--listar` mostra as etapas e o tempo esperado; ver o `README.md`).

| Ordem | Script | Produz |
|---|---|---|
| 1 | `src/data/download.py` | `data/raw/VRA_{ano}_{mes}.csv` |
| 2 | `src/data/ler_dados.py` | `data/interim/vra_bruto.parquet` |
| 3 | `src/data/clean.py` | `data/interim/vra_limpo.parquet` |
| 4 | `src/data/target.py` | `data/interim/vra_target.parquet` |
| 5 | `src/data/features.py` | `data/processed/{train,val,test,subamostra}.parquet` |
| 6 | `src/models/baselines.py` | `results/tables/baselines*.csv`, modelos em `experiments/runs/baselines/` |
| 7 | `src/importance/gawll.py` | checkpoints em `experiments/runs/gawll/` (~5,4 h) |
| 8 | `src/importance/consolidar_gawll.py` | `importancia_gawll.csv`, `interacao_gawll.csv`, `selecao_gawll.csv` |
| 9 | `src/importance/{intrinsic,permutation,shap_global}.py` | `importancia_{intrinseca,permutation,shap}.csv` |
| 10 | `src/synthetic/{generate,aplicar_metodos,recovery}.py` | `sintetico_*.csv`, `recuperacao_sintetico*.csv` (~30 min) |
| 11 | `src/compare/{rankings,stability,figuras}.py` | `concordancia*.csv`, `estabilidade*.csv`, figuras |
| 12 | `src/compare/fidelity.py` | `fidelidade.csv`, `fidelidade_resumo.csv`, figura (~50 min; toca o teste) |
| 13a | `src/interaction/friedman_h.py` | `interacao_h.csv` (~7 min) |
| 13b | `src/compare/interacoes.py` (depois da 15) | `interacoes*.csv`, figura PyGAwLLfs × H |
| 15 | `src/domain/validate_pairs.py` | `dominio_pares.csv`, `dominio_celulas.csv`, grafo e heatmaps (~1 min) |
| — | `src/data/{figuras,auditar_congestionamento}.py` | histograma do atraso; auditoria do congestionamento |

Testes (scripts com `assert`): `tests/test_gawll.py`, `tests/test_importancia.py`,
`tests/test_sintetico.py`, `tests/test_comparacao.py`, `tests/test_dominio.py`,
`tests/test_interacao.py`, `tests/test_fidelidade.py`. (O `aplicar_metodos.py` do sintético calcula também o
H, então depende do `friedman_h.py`.)

O projeto e o clone `PyGAwLLfs/` (não versionado) têm, cada um, uma pasta
`src/` sem `__init__.py`; elas se fundem no `sys.path`. Nunca criar
`__init__.py` em nenhuma das duas.
