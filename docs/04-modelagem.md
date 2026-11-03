# 4. Modelagem

Quarta fase do CRISP-DM: os modelos preditivos e os métodos que os explicam.
Código em `src/modelagem/` (etapas `baselines`, `pygawllfs` e `importancias`).

## Modelos preditivos

`baselines.py` treina quatro modelos no treino, escolhe a configuração de cada um
pela **acurácia na validação** e avalia a escolhida uma única vez no teste.

| Modelo | Configuração | Grade testada |
|---|---|---|
| trivial | sempre a classe mais frequente | — |
| regressão logística | `StandardScaler` + `LogisticRegression` (até 1.000 iterações) | C ∈ {0,1; 1; 10} |
| Random Forest | 100 árvores | profundidade máxima 10 (fixa) |
| XGBoost | 300 árvores, taxa de aprendizado 0,1, `tree_method="hist"` na GPU | profundidade máxima ∈ {4, 6, 8} |

- Todos com semente 42 e sem peso de classe (os conjuntos já são 50/50).
- O Random Forest é fixo em 100 árvores e profundidade 10, o mesmo do código do
  PyGAwLLfs, para que todos os métodos expliquem o mesmo modelo.
- `baselines_tuning.csv` guarda as métricas de cada configuração na validação e no
  treino (colunas `*_treino`), para o diagnóstico de ajuste.
- **Alarme de vazamento:** antes de tocar o teste, o script para se alguma
  configuração escolhida tiver, na validação, ROC-AUC acima de 0,90, F1 acima de
  0,85 ou acurácia acima de 0,85. Com atributos conhecidos antes da partida, um
  desempenho assim indicaria um atributo vazando o alvo.
- Os modelos treinados ficam em `experiments/runs/baselines/` e são os mesmos que
  todos os métodos de importância explicam.

## Métodos de importância

Todos produzem uma tabela no mesmo formato (`feature`, `importance`, `rank`,
`method`, `model`, `seed`; rank 1 = mais importante), calculada sem o conjunto de
teste.

### Importância intrínseca (`intrinsic.py`)

O que o próprio modelo carrega depois do treino, como no tutorial de importância
de atributos do Machine Learning Mastery (Brownlee, 2020):

- **regressão logística:** o coeficiente de cada atributo padronizado
  (`coef_[0]` com o `StandardScaler`), com sinal; o rank usa o valor absoluto;
- **Random Forest:** a redução média de impureza (MDI, `feature_importances_`);
- **XGBoost:** o ganho médio das divisões que usam o atributo (*gain*,
  `feature_importances_`).

É determinística: um valor por modelo.

### Permutation Importance (`permutation.py`)

Para cada atributo, embaralha a coluna na validação e mede a queda da acurácia do
modelo já treinado; repete 10 vezes e tira a média:

```
PI(j) = acurácia(validação) − média_r acurácia(validação com a coluna j embaralhada, repetição r)
```

O valor pode ser negativo (embaralhar pode melhorar a acurácia por acaso); o rank
usa o valor bruto. Roda com 5 sementes (42 a 46), que mudam só o embaralhamento.

### TreeSHAP (`shap_global.py`)

Valores SHAP exatos para árvores (`shap.TreeExplainer`,
`feature_perturbation="tree_path_dependent"`), no Random Forest e no XGBoost. A
importância global de um atributo é a média de |SHAP| em 10 mil voos sorteados da
validação, com 5 sementes de sorteio. No Random Forest o SHAP está na escala de
probabilidade; no XGBoost, na de log-odds. A regressão logística não tem SHAP:
seus coeficientes padronizados cumprem esse papel.

### PyGAwLLfs (`gawll.py`, `consolidar_gawll.py`)

Algoritmo genético com *linkage learning* (Tinós et al., 2023) aplicado à seleção
de atributos (Guimarães et al., 2026). Cada indivíduo é uma máscara x de N = 29
bits (atributo usado ou não). O fitness é

```
f(x) = 0,98 · acurácia(x) + 0,02 · (1 − |x| / N)
```

em que acurácia(x) é a acurácia, na validação da subamostra, do modelo treinado na
parte de treino da subamostra só com os atributos de x, e |x| é o número de
atributos usados. O motor do GA é o do repositório original (commit `3be28a6`),
usado por importação e subclasses, sem edição.

**Uma geração** (população de 50, taxa de cruzamento 0,8):

1. 40 filhos por cruzamento uniforme entre pais escolhidos por torneio de 3, cada
   um com mutação bit a bit de probabilidade 1/N;
2. **3 sondagens de *linkage learning***: a partir de um pai x escolhido por
   torneio, sorteiam-se dois atributos g e h e avaliam-se x⊕g, x⊕h e x⊕gh
   (x com o bit g, o bit h e os dois invertidos), 3 filhos por sondagem;
3. 1 filho de preenchimento (cópia mutada de um pai);
4. elitismo: o melhor indivíduo da geração anterior substitui o pior da nova.

Se o melhor fitness não melhora por 50 gerações, a população é perturbada a partir
do melhor indivíduo (mutação de probabilidade 0,3 em cada bit).

**Importância e interação.** Em cada sondagem do par (g, h):

```
δ_g    = |f(x⊕g) − f(x)|                         efeito de inverter g
δ_h    = |f(x⊕h) − f(x)|                         efeito de inverter h
ω_gh   = |f(x⊕gh) − f(x⊕g) − f(x⊕h) + f(x)|      não aditividade do par
```

Quando ω_gh > 0, entram também os efeitos condicionais |f(x⊕gh) − f(x⊕h)| (de g)
e |f(x⊕gh) − f(x⊕g)| (de h). Ao longo da execução, **v_imp(g)** guarda o maior
efeito já observado para g, e **v_int(g, h)** a maior não aditividade já observada
para o par. O resultado de uma execução é o melhor subconjunto x*, o vetor v_imp e
a matriz v_int.

**Orçamento e cobertura.** Com 3 sondagens por geração, 200 gerações dão 600
sondagens; com 406 pares possíveis, a fração esperada de pares sondados ao menos
uma vez é 1 − (1 − 1/406)^600 ≈ 77%. Um contador de sondagens
(`LinkageLearningComContador`) registra quantas vezes cada par foi sondado, o que
separa "par não sondado" de "par sondado, interação nula" (coluna `sondada` de
`interacao_gawll.csv`). O checkpoint de cada execução guarda também o melhor
fitness de cada geração e as gerações com perturbação.

**Execuções:** 3 modelos × 5 sementes (42 a 46), uma por vez, com retomada pelo
checkpoint. O número de gerações vem da variável de ambiente `TCC_GERACOES`
(padrão 200), e a consolidação recusa checkpoints com outro número.

### H de Friedman (`friedman_h.py`)

Mede a interação de um par na **função** do modelo (Friedman e Popescu, 2008), a
partir das dependências parciais (PD) centradas, avaliadas em 500 voos sorteados
da validação (que servem de pontos e de fundo):

```
resíduo(j, k) = PD_jk − PD_j − PD_k
H²(j, k)      = Σ resíduo² / Σ PD_jk²          (normalizado)
H_abs(j, k)   = raiz(média de resíduo²)        (não normalizado)
```

- Escala logit: "sem interação" quer dizer efeitos que se somam no log-odds.
- **H_abs é a versão principal** (Inglis et al., 2022): o normalizado fica instável
  quando o efeito conjunto do par é pequeno e passa de 1 em pares de *dummies* da
  mesma variável, cuja combinação "as duas ligadas" não existe nos dados. As duas
  versões são salvas.
- Calculado nos **406 pares** das 29 features, o mesmo universo que o PyGAwLLfs
  pode sondar, no Random Forest e no XGBoost, com 5 sementes de amostra.
- A regressão logística fica de fora: no logit ela é aditiva, então H = 0 por
  construção (conferido pelos testes).

## Artigo, código e este trabalho

| Item | Artigo | Código PyGAwLLfs | Este trabalho |
|---|---|---|---|
| Divisão dos dados | 70/30 aleatória | 70/30 aleatória | temporal: treino, validação e teste |
| Classes | não balanceia | não balanceia | 50/50 em cada conjunto |
| Métrica do fitness | acurácia | acurácia | acurácia |
| Onde o fitness é medido | teste | teste | validação (o teste fica fora da busca) |
| Fitness | 0,98 · desempenho + 0,02 · parcimônia | igual | igual |
| População / gerações | 50 / 100 | 50 / 100 | 50 / 200 |
| p_c / p_m | 0,4 / 1/\|P\| | 0,8 / 1/N | 0,8 / 1/N (Adendo B) |
| Seleção | torneio de 3 e elitismo | igual, com o elitismo inoperante | igual, com o elitismo corrigido (Adendo A) |
| Mutação | bit a bit | inoperante | bit a bit, corrigida (Adendo A) |
| Modelos | árvore de decisão, KNN, MLP, RF | — | regressão logística, RF (100 árvores, profundidade 10, como no código) e XGBoost |
| Métodos comparados | intrínseca (árvore e RF), Permutation | — | intrínseca (LogReg, RF, XGBoost), Permutation, SHAP, H de Friedman |
| Comparação | correlação de Pearson, matriz e dispersão | — | Spearman (as escalas diferem), no mesmo formato visual |
| Execuções reportadas | a primeira | média | média de 5 sementes |
| Grafo de interação reduzido | 5% das arestas | — | 5% das arestas |

### Adendo A — dois erros de implementação do código original

1. **Mutação inoperante.** `Mutation.bit_flip_mutation` devolve um vetor novo com
   os bits invertidos, mas o motor do GA ignora o retorno: nenhum bit é invertido,
   nem na mutação comum nem na perturbação por estagnação. Sem mutação, o GA quase
   não explora o espaço de subconjuntos.
2. **Elitismo perdido.** `GAwLL._apply_elitism` grava o melhor indivíduo na
   população antiga, que é descartada logo em seguida; o melhor da geração pode se
   perder.

Os dois são reproduzidos por `tests/test_gawll.py` (`teste_mutacao` e
`teste_elitismo`). A versão usada é a corrigida: `MutacaoInPlace` inverte os bits
no próprio vetor, e `GAwLLCorrigido._apply_elitism` grava o elite na população
nova. Com o elitismo corrigido, o melhor fitness nunca cai de uma geração para a
outra, o que o teste também confere.

### Adendo B — taxas de cruzamento e de mutação

O artigo reporta taxa de cruzamento de 0,4 e mutação de 1/|P|; o código do
repositório usa 0,8 e 1/N. Este trabalho segue o código, que é o que de fato roda.
Com 0,8, cada geração tem 40 filhos de cruzamento e espaço para 3 sondagens de
*linkage learning*, o que dá as 600 sondagens em 200 gerações e a cobertura de
cerca de 77% da matriz de interação.

## Referências

- BROWNLEE, J. How to calculate feature importance with Python. *Machine Learning
  Mastery*, 2020. Disponível em:
  https://machinelearningmastery.com/calculate-feature-importance-with-python/.
- BREIMAN, L. Random forests. *Machine Learning*, v. 45, p. 5–32, 2001.
- CHEN, T.; GUESTRIN, C. XGBoost: a scalable tree boosting system. In: *KDD 2016*.
  p. 785–794.
- FRIEDMAN, J. H.; POPESCU, B. E. Predictive learning via rule ensembles.
  *Annals of Applied Statistics*, v. 2, n. 3, p. 916–954, 2008.
- GUIMARÃES, D. G.; CARVALHO, S. S.; NASCIMENTO, R. H.; TINÓS, R. Estimating
  variable importance and interaction in machine learning via genetic
  algorithms. In: *Intelligent Systems: BRACIS 2025*. Springer, 2026.
  p. 478–492. (LNAI 16179). DOI: 10.1007/978-3-032-15987-8_32.
- INGLIS, A.; PARNELL, A.; HURLEY, C. B. Visualizing variable importance and
  variable interaction effects in machine learning models. *Journal of
  Computational and Graphical Statistics*, v. 31, n. 3, p. 766–778, 2022.
- LUNDBERG, S. M. et al. From local explanations to global understanding with
  explainable AI for trees. *Nature Machine Intelligence*, v. 2, p. 56–67, 2020.
- TINÓS, R.; PRZEWOZNICZEK, M. W.; WHITLEY, D.; CHICANO, F. Genetic algorithm
  with linkage learning. In: *GECCO 2023*. p. 981–989.
  DOI: 10.1145/3583131.3590349.
