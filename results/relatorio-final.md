# Métodos de importância e interação de atributos na previsão de atrasos de voos

**Relatório final do projeto** · Igor Pio · orientador: Prof. Dr. Renato Tinós · FFCLRP/USP · outubro de 2026

## Resumo

Este trabalho compara cinco formas de medir **quais atributos importam** e
**quais interagem** num modelo de aprendizado de máquina, usando como problema
a previsão de atrasos de voos no Brasil (base VRA da ANAC, 2024–2025, 1,86
milhão de voos). O método central é o **PyGAwLLfs**, um algoritmo genético que
seleciona atributos e, estima a importância de cada um e a interação
entre pares. Ele foi comparado com **Permutation Importance**, **TreeSHAP**, as
**importâncias intrínsecas** dos modelos e, para interações, a **estatística H
de Friedman**, em três modelos (regressão logística, Random Forest e XGBoost) e
por quatro critérios: **concordância, estabilidade, fidelidade e custo**.

Principais resultados:

- **Os métodos concordam sobre o que importa**: aeroporto de origem, empresa e
  hora lideram quase todos os rankings. A concordância é maior entre métodos
  aplicados ao mesmo modelo (Spearman 0,70) do que entre modelos diferentes
  (0,57–0,60).
- **Todos apontam atributos que carregam a informação**: só com as 5 features
  mais importantes de cada método, o modelo retém 87%–99% do PR-AUC do modelo
  completo no teste, contra 72%–81% de 5 features sorteadas — vantagem que se
  mantém com intervalo de confiança de 95% em todos os casos.
- **Nem toda diferença pequena é real**: com intervalos de confiança por
  bootstrap de dias do teste, Random Forest e XGBoost empatam no PR-AUC, e no
  XGBoost os quatro métodos empatam na fidelidade do top-5.
- **Os métodos discordam sobre o que interage**: PyGAwLLfs e H de Friedman
  quase não concordam (Spearman de 0,13 a 0,21). Num dataset sintético com gabarito, os
  dois encontram a interação verdadeira, mas só o PyGAwLLfs trata **pares
  redundantes** como interação.
- **O custo difere em duas ordens de grandeza**: 4,9 h e milhares de treinos
  para o PyGAwLLfs, contra segundos a minutos e nenhum treino para os demais.
- Ao adaptar o PyGAwLLfs, foram corrigidos três problemas metodológicos do
  protocolo original e **dois bugs de implementação** do repositório.

## 1. Objetivo e perguntas de pesquisa

**Pergunta principal.** Como diferentes métodos de estimação de importância e
interação de atributos se comparam em termos de **concordância**,
**estabilidade**, **fidelidade** e **custo computacional** no problema de
previsão de atrasos de voos?

Ela foi desdobrada em seis perguntas específicas, respondidas na seção 9:

| | Pergunta | Como foi medida |
|---|---|---|
| P1 | Os métodos apontam os mesmos atributos? A concordância depende mais do método ou do modelo explicado? | correlação entre rankings (Spearman, Kendall, RBO, top-k) |
| P2 | Os rankings se repetem quando muda a semente aleatória? | as mesmas métricas entre 5 sementes |
| P3 | Os atributos apontados como importantes realmente carregam a informação? | retreino só com o top-k, avaliado no teste |
| P4 | Quanto custa cada método? | tempo e número de treinos de modelo |
| P5 | PyGAwLLfs e H concordam sobre as interações? Qual recupera a estrutura verdadeira? | correlação entre rankings de pares; dataset sintético com gabarito |
| P6 | As interações do PyGAwLLfs correspondem a padrões reais de atraso? | teste descritivo nos dados de voo |

## 2. Dados e preparação

**Fonte.** VRA (Voo Regular Ativo) da ANAC: um registro por voo, com empresa,
aeroportos, horários previstos e realizados. Foram usados os 24 meses de
2024–2025.

**Limpeza.** Dos 1.992.832 registros, saíram os cancelados (69.496), os sem
horário (59.141), 11 fora do período e 8.204 com atraso implausível (fora de
−40 a 720 min, erros de data). Restaram **1.855.980 voos**.

**Alvo.** `atrasado = 1` se a partida real ocorreu 15 minutos ou mais depois da
prevista (critério usual da aviação): **18,5%** dos voos. A classe é
desbalanceada, o que orienta a escolha das métricas (F1 e PR-AUC, não acurácia).

![](figures/histograma_atraso.png)

*Figura 1 — Distribuição do atraso na partida. A linha tracejada marca o limiar de 15 min.*

**Divisão temporal.** Os conjuntos foram separados por data, nunca
aleatoriamente, para simular a previsão do futuro a partir do passado:

| Conjunto | Período | Voos | Atrasados |
|---|---|---:|---:|
| Treino | jan/2024 – ago/2025 | 1.539.522 | 18,1% |
| Validação | set – out/2025 | 158.389 | 16,2% |
| Teste | nov – dez/2025 | 158.069 | 24,3% |

O teste cai no pico sazonal de fim de ano; isso foi documentado, e não
corrigido, porque o futuro tem sazonalidade.

**Atributos (29).** Todos são conhecidos antes da partida:

- **calendário e horário** da partida prevista: hora, minuto, dia da semana,
  fim de semana, período do dia, dia do mês, mês, trimestre, dia do ano,
  feriado e véspera de feriado;
- **duração prevista** do voo e **congestionamento programado** (voos previstos
  na mesma hora no aeroporto de origem e no de destino);
- **tipo de linha**, em one-hot: doméstica mista (N, passageiros), internacional
  mista (I), doméstica cargueira (C) e internacional cargueira (G), na
  codificação da ANAC (CRUZ, 2013);
- **porte** dos aeroportos de origem e destino (quartis de volume no treino,
  one-hot);
- **empresa, origem e destino** por *target encoding*: a taxa de atraso
  suavizada de cada categoria. Codificar centenas de aeroportos em one-hot
  tornaria o algoritmo genético e o H inviáveis.

**Cuidados contra vazamento de informação**, verificados por código e por
testes automáticos:

1. horários realizados, situação do voo e código de justificativa nunca entram
   como atributo (uma guarda interrompe o pipeline se aparecerem);
2. o *target encoding* do treino é calculado **fora da dobra** (5 dobras), para
   que nenhum voo use o próprio rótulo;
3. porte e encoding são ajustados só no treino e aplicados sem reajuste nos
   outros conjuntos;
4. o **teste só é usado em dois momentos**: a avaliação final dos modelos e o
   experimento de fidelidade. Todos os métodos de importância e interação são
   calculados na validação.

## 3. Modelos preditivos

Foram usados três modelos de complexidade crescente, todos com balanceamento
de classe e semente fixa. Os hiperparâmetros foram escolhidos pelo F1 na
validação, e cada modelo final foi avaliado **uma única vez** no teste.

| Modelo | Configuração fixa | Grade testada | Escolhido |
|---|---|---|---|
| Regressão logística | atributos padronizados | C ∈ {0,1; 1; 10} | C = 1 |
| Random Forest | 100 árvores | profundidade 10 (a do artigo original) | profundidade 10 |
| XGBoost | 300 árvores, taxa 0,1, GPU | profundidade ∈ {4, 6, 8} | profundidade 8 |

**Desempenho no teste**, com IC de 95% (bootstrap de dias, pareado; seção 5):

| Modelo | F1 | PR-AUC | ROC-AUC | Acurácia balanceada |
|---|---|---|---|---:|
| Trivial (sempre "não atrasa") | 0,000 | 0,243 | 0,500 | 0,500 |
| Regressão logística | 0,417 [0,393; 0,438] | 0,354 [0,332; 0,377] | 0,644 [0,631; 0,658] | 0,602 |
| Random Forest | 0,440 [0,408; 0,473] | **0,403** [0,374; 0,434] | **0,682** [0,671; 0,693] | 0,603 |
| XGBoost | **0,451** [0,417; 0,484] | 0,394 [0,365; 0,427] | 0,676 [0,662; 0,691] | **0,622** |

As duas árvores ficam à frente da regressão logística nas três métricas, e a
diferença é real: o IC da diferença pareada exclui o zero (PR-AUC +0,04 a
+0,05). Entre elas, **o RF e o XGBoost empatam** no PR-AUC (RF − XGBoost =
+0,009, IC [−0,003; +0,019]) e no ROC-AUC. No F1, o XGBoost fica à frente por
0,010 (IC [0,002; 0,018]), diferença real, mas pequena. O desempenho é moderado, como
esperado: clima e propagação de atrasos entre voos, duas causas importantes de
atraso, ficaram de fora de propósito, porque dependem de informação posterior
à partida ou de bases externas. O objetivo do trabalho não é o melhor preditor, mas modelos
razoáveis cujas explicações possam ser comparadas.

![](figures/matriz_confusao.png)

*Figura 2 — Matrizes de confusão no teste, com corte em 0,5. Cor e % = fração da classe real.*

O F1 parecido esconde modelos que erram de jeitos diferentes. O balanceamento
de classe (`class_weight="balanced"`; no XGBoost, `scale_pos_weight`) faz os
atrasos, 18% do treino, pesarem tanto quanto os voos pontuais, e com o corte em
0,5 os três erram mais por alarme falso do que por atraso não detectado. O
Random Forest detecta 85% dos atrasos, mas marca como atrasados 65% dos voos
pontuais; a regressão logística fica no outro extremo (53% dos atrasos, 33% de
alarmes falsos); o XGBoost fica no meio (75% e 51%). Por isso a acurácia dos
três (47%–64%) fica abaixo da do classificador trivial (76%), e o trabalho usa
F1 e PR-AUC. O corte não foi ajustado: o objetivo é comparar explicações, e o
PR-AUC é a leitura que não depende dele. As matrizes vêm das mesmas previsões
da avaliação final (conferidas contra a tabela acima).

## 4. Métodos comparados

| Método | O que mede | Retreina o modelo? | Onde é calculado |
|---|---|---|---|
| **PyGAwLLfs** | quanto o desempenho muda ao ligar ou desligar atributos (importância) e pares de atributos (interação) | sim, milhares de vezes | subamostra de 50 mil voos (treino e validação) |
| **Permutation** | queda do F1 ao embaralhar um atributo | não | validação |
| **TreeSHAP** | contribuição média de cada atributo à previsão (RF e XGBoost) | não | 10 mil linhas da validação |
| **Intrínseca** | coeficiente padronizado (LogReg), redução de impureza (RF), ganho (XGBoost) | não | o próprio modelo |
| **H de Friedman** | quanto a previsão conjunta de um par não é a soma dos efeitos isolados (só interação) | não | 500 linhas da validação |

### 4.1 PyGAwLLfs

Um algoritmo genético procura o melhor subconjunto de atributos. Cada
indivíduo é um vetor de 29 bits (bit 1 = atributo usado), e o *fitness* é

> fitness = 0,98 · F1 na validação + 0,02 · (fração de atributos descartados).

A cada geração, além do cruzamento e da mutação, o *linkage learning* sorteia
dois atributos g e h de um indivíduo x e avalia três variantes, invertendo o
bit de g, o de h e os dois (ligando o atributo que estava desligado, ou o
contrário). Dessas avaliações saem:

- **importância** de g: a maior variação de fitness já observada ao inverter g;
- **interação** de (g, h): o quanto inverter os dois juntos difere da soma de
  inverter cada um, ou seja, a **não aditividade do desempenho**.

O resultado final de cada execução é o melhor subconjunto (**x\***), um vetor
de importância e uma matriz de interação. Foram usados os parâmetros do artigo
original (população 50, 100 gerações) e 5 sementes por modelo (15 execuções).

**Correções feitas** (todas por subclasse, sem editar o repositório original,
e cobertas por testes):

| # | Problema no original | Correção |
|---|---|---|
| 1 | o fitness é calculado no conjunto de **teste**: a busca otimiza sobre o conjunto de avaliação | fitness na **validação** |
| 2 | fitness por **acurácia**: com 18% de atrasos, "nunca atrasa" tem ~76% de acurácia | **F1** |
| 3 | divisão **aleatória** dos dados | divisão **temporal** |
| 4 | *bug*: a mutação devolve um vetor novo que é ignorado, então nenhum bit é invertido | mutação no próprio vetor |
| 5 | *bug*: o elitismo grava o melhor indivíduo na população antiga, que é descartada | elitismo na população nova |

Uma consequência da correção 1: o F1 de validação do melhor indivíduo é
otimista (o subconjunto foi escolhido olhando para ele). Por isso o
desempenho do x\* só é reportado no teste (seção 6.4).

### 4.2 H de Friedman

Para um par de atributos (j, k), compara a **dependência parcial conjunta**
(como a previsão varia com j e k juntos) com a soma das individuais. Foi
calculado na escala logit, em que a regressão logística é exatamente aditiva
(H = 0 por construção; por isso só RF e XGBoost entram). A versão principal é
a não normalizada (H_abs), mais estável. Por custo, o H foi calculado só para
a união entre o top-15 de importância por consenso e todos os atributos do
grafo de interação do PyGAwLLfs: 23 atributos (253 pares) no RF e 25 (300
pares) no XGBoost. A união garante que os atributos apontados pelo PyGAwLLfs
não fiquem de fora da comparação.

## 5. Como os resultados foram obtidos

**Rankings.** Cada método produz uma importância por atributo. Para cada par
(método, modelo), as 5 sementes (42 a 46) foram resumidas pela média,
gerando 11 rankings: 4 métodos × 3 modelos, menos o SHAP da regressão
logística, que repetiria o coeficiente. As escalas dos métodos não são
comparáveis (queda de F1, diferença de fitness, unidades de logit), então
**toda comparação é por posição no ranking**, nunca por valor.

**Concordância (P1).** Para cada par de rankings: Spearman, Kendall tau-b
(trata empates), **RBO** com p = 0,9 (dá mais peso ao topo, onde a decisão
acontece: as 10 primeiras posições respondem por ~86% do valor) e a
sobreposição dos top-5, top-10 e top-15. Os 55 pares de rankings foram
separados em três grupos: métodos diferentes no **mesmo modelo** (a
comparação principal), o **mesmo método** em modelos diferentes, e **tudo
diferente**. Isso separa a divergência causada pelo método da causada pelo
modelo.

**Estabilidade (P2).** As mesmas métricas, entre as 10 duplas de sementes de
cada método. A intrínseca é determinística e fica de fora.

**Interações (P5).** Numa execução do PyGAwLLfs, só os pares que o *linkage
learning* sorteou recebem valor na matriz de interação (cerca de metade). A
média das 5 sementes usa só as execuções em que o par foi sondado, e o ranking
só inclui pares sondados em pelo menos 2 delas; contar o par não sondado como
interação zero faria o ranking acompanhar a cobertura das sondagens (seção
6.7). No H, a média é das 5 amostras.

**Fidelidade (P3).** Se um método diz que certos atributos são os mais
importantes, um modelo treinado **só com eles** deve chegar perto do modelo
completo. Para cada modelo, o mesmo algoritmo com os mesmos hiperparâmetros
foi retreinado no treino completo com: o top-5, top-10 e top-15 de cada método;
o x\* de cada execução do PyGAwLLfs; e **10 sorteios aleatórios** de k
atributos, como referência de acaso. Foram 138 treinos, avaliados no teste.
A medida é a **fração do PR-AUC do modelo completo** que se mantém. O PR-AUC
não depende de limiar, por isso é mais estável que o F1.

**Custo (P4).** Tempo de cada execução e, como medida independente da máquina,
o **número de treinos de modelo**. Máquina: notebook com Intel Core i7, 14 GB
de RAM e GPU RTX 4050 (usada só pelo XGBoost).

**Incerteza.** O teste é uma amostra de dias (nov–dez/2025). Para saber se uma
diferença é maior que a variação entre amostras, os 61 dias do teste foram
sorteados com reposição 1.000 vezes (bootstrap **por dias**, porque voos do
mesmo dia não são independentes: a taxa de atraso diária vai de 9% a 62%). Em
cada repetição, a métrica foi recalculada para todos os modelos e
configurações com o **mesmo sorteio** (bootstrap pareado). "A é melhor que B"
exige que o IC de 95% da diferença A − B não contenha zero. Isto não é um novo
uso do teste: as previsões são as mesmas da avaliação final e da fidelidade, de
configurações já fixadas, e nada foi escolhido a partir do bootstrap. Detalhes
e limitações em `docs/incerteza.md`.

**Dataset sintético (P5).** 5.000 exemplos com estrutura conhecida:

> z = 2·x1·x2 + x3 + x4 + 0,5·ruído; y = 1 se z > 0.

Nele, x1 e x2 só agem juntos (**interação pura**), x3 e x4 têm efeito aditivo,
x5 e x6 são **cópias correlacionadas** (ρ = 0,9) de x3 e x4, e x7–x10 são ruído.
Todos os métodos rodaram com o mesmo código usado nos dados reais, e cada
resultado foi comparado com esse gabarito.

**Validação de domínio (P6).** Para os pares do topo do PyGAwLLfs (top-10 de
cada modelo, 23 pares), mais 10 pares do fundo do ranking como controle:

1. cada par foi classificado como **estrutural** (duas codificações da mesma
   variável), **redundante** (uma variável quase determina a outra) ou
   **operacional**;
2. para os operacionais, a taxa de atraso em cada célula do cruzamento das
   duas variáveis foi comparada com a prevista por um modelo **só com os
   efeitos isolados**;
3. o par é **corroborado descritivamente** se, em 2024 **e** em 2025, o desvio
   médio supera o ruído amostral (bootstrap) e 1 ponto percentual, e se o
   padrão de desvios se repete entre os dois anos (correlação ≥ 0,5).

Essa análise usa só treino e validação. Corroborar não é provar causa: um
fator não observado pode produzir o mesmo padrão.

**Reprodutibilidade.** Todo o pipeline roda com um comando (`python main.py`,
~6 h, quase tudo no PyGAwLLfs), com retomada por checkpoint. Há 9 suítes de
testes automáticos (45 testes). Entre outras coisas, elas verificam que nenhum
método abre o conjunto de teste, que os dois bugs existem no original e foram
corrigidos, que o H acerta funções com resposta conhecida e que as tabelas
finais reproduzem as fontes.

## 6. Resultados

### 6.1 O que os métodos apontam como importante

Top-5 de cada ranking:

| Modelo | Método | 1º | 2º | 3º | 4º | 5º |
|---|---|---|---|---|---|---|
| LogReg | Intrínseca | dia do ano | mês | dia do mês | origem | empresa |
| LogReg | Permutation | dia do ano | dia do mês | mês | origem | fim de semana |
| LogReg | PyGAwLLfs | hora | origem | empresa | período do dia | fim de semana |
| RF | Intrínseca | empresa | origem | hora | duração prevista | período do dia |
| RF | Permutation | origem | hora | empresa | dia do ano | congest. origem |
| RF | SHAP | origem | hora | empresa | período do dia | dia do ano |
| RF | PyGAwLLfs | período do dia | trimestre | origem | hora | empresa |
| XGBoost | Intrínseca | empresa | hora | origem | linha C (dom. cargueira) | dia do ano |
| XGBoost | Permutation | hora | empresa | origem | congest. origem | duração prevista |
| XGBoost | SHAP | hora | empresa | origem | dia do ano | dia da semana |
| XGBoost | PyGAwLLfs | dia do ano | origem | mês | empresa | hora |

Aeroporto de origem, empresa e hora aparecem no top-5 de quase todos. Na
regressão logística, intrínseca e Permutation são dominadas por atributos de
calendário colineares (dia do ano ≈ 30·mês + dia do mês): os coeficientes de
`dia_ano` e `mes` são grandes e de sinais opostos e se anulam. O PyGAwLLfs,
que retreina o modelo sem cada atributo, não cai nessa armadilha.

### 6.2 Concordância

![](figures/concordancia_intra_modelo.png)

*Figura 3 — Concordância entre os métodos dentro de cada modelo: Spearman (em cima) e RBO (embaixo).*

| Tipo de comparação | Spearman | Spearman: menor e maior dupla | RBO | Top-5 em comum |
|---|---:|---:|---:|---:|
| Métodos diferentes, mesmo modelo | **0,70** | 0,33–0,98 | 0,66 | 68% |
| Mesmo método, modelos diferentes | 0,60 | 0,11–0,84 | 0,59 | 58% |
| Método e modelo diferentes | 0,57 | 0,13–0,90 | 0,61 | 60% |

- A concordância é **moderada a alta** e maior dentro do mesmo modelo: o
  modelo explicado pesa tanto quanto o método. A dispersão entre as duplas,
  porém, é grande perto da diferença entre as médias.
- **Cai com a complexidade do modelo**: Spearman médio entre os métodos de 0,87
  na LogReg, 0,77 no RF e 0,56 no XGBoost.
- O **PyGAwLLfs é o que menos concorda** com os outros (0,64 nas duplas com
  ele, contra 0,78 nas duplas sem ele). Exceção: no RF, a Permutation tem o
  menor Spearman médio (0,69, perto do PyGAwLLfs, 0,70), mas RBO de 0,72, só
  abaixo do SHAP (0,77). Ela concorda no topo e diverge na cauda, onde dá valores nulos ou levemente
  negativos (sem efeito detectável), e a ordem entre eles é arbitrária.

### 6.3 Estabilidade

![](figures/estabilidade_sementes.png)

*Figura 4 — Estabilidade entre sementes: ponto = média nas 10 duplas de sementes; traço até o pior caso.*

- **Permutation e SHAP** praticamente não mudam com a semente (Spearman médio
  0,97–1,00; pior dupla de sementes, 0,93): sobre um modelo fixo, a semente só
  muda o embaralhamento ou a amostra.
- **PyGAwLLfs**: média de 0,89 na LogReg, 0,68 no RF e 0,66 no XGBoost; na pior
  dupla de sementes, 0,79, 0,53 e 0,37. Aqui a semente muda a busca inteira. O
  topo é mais firme que a cauda: 64%–88% do top-5 se repete entre execuções.
- A **matriz de interação** do PyGAwLLfs quase não se repete entre execuções
  (Spearman 0,08–0,11), enquanto a do H fica em 0,98. Com 29 atributos
  há 406 pares, e cada execução faz 300 sondagens. Cada par é sondado, em
  média, 0,7 vez, e metade fica sem nenhuma medição.

### 6.4 Fidelidade

![](figures/fidelidade.png)

*Figura 5 — Fidelidade no teste: desempenho do modelo retreinado só com as top-k features de cada método. Faixa cinza = sorteios aleatórios; estrelas = x\* do PyGAwLLfs.*

Fração do PR-AUC do modelo completo retida no teste, em %, com IC de 95%
(bootstrap de dias, pareado):

| Modelo | k | Intrínseca | Permutation | SHAP | PyGAwLLfs | Sorteio |
|---|---:|---|---|---|---|---|
| LogReg | 5 | 89 [86; 91] | 87 [85; 89] | — | **99 [97; 101]** | 81 [78; 83] |
| LogReg | 10 | 97 [95; 98] | 97 [96; 98] | — | 97 [96; 99] | 91 [89; 92] |
| RF | 5 | 94 [91; 97] | **98 [96; 100]** | **97 [95; 99]** | 94 [91; 96] | 72 [70; 74] |
| RF | 10 | 97 [95; 99] | 100 [99; 101] | 97 [95; 99] | 99 [98; 100] | 86 [85; 87] |
| XGBoost | 5 | 93 [89; 97] | 97 [93; 101] | 95 [93; 97] | 93 [89; 97] | 72 [70; 74] |
| XGBoost | 10 | 96 [94; 98] | 98 [96; 100] | 100 [99; 101] | 96 [93; 100] | 85 [83; 87] |

- **Todos os métodos ficam muito acima do acaso** no top-5 (+6 a +26 pontos
  percentuais), e os 33 ICs da diferença método − sorteio (todos os modelos e
  k) excluem o zero. Com k = 15, todos retêm 98%–100%: a maior parte da
  informação está em poucos atributos.
- **Diferenças entre métodos no top-5, pelo IC da diferença pareada:** na
  regressão logística, o **PyGAwLLfs** é o mais fiel, 10 a 12 pontos à frente
  dos outros (real). No RF, **Permutation e SHAP** ficam 3 a 4 pontos à frente
  da intrínseca e do PyGAwLLfs (real); entre as duas, 0,9 ponto, no limite (IC
  de +0,1 a +1,7). No XGBoost, **os quatro empatam**: nenhuma diferença exclui o
  zero, porque o modelo com 5 features varia muito de um dia para outro.
- O top-5 da intrínseca na LogReg retinha 96% do F1 na
  validação e caiu para 82,5% no teste: efeitos de calendário aprendidos em
  outras estações não se transferem para novembro e dezembro.
- O **x\* do PyGAwLLfs**, com cerca de um terço dos atributos (9 a 10, em
  média), retém 100% (LogReg), 93% (RF) e 97% (XGBoost) do PR-AUC. No RF, todas
  as execuções ficam abaixo do modelo completo (ICs de 89% a 97%).

### 6.5 Custo

| Método | Por execução | Treinos por execução | Total (3 modelos, 5 sementes) |
|---|---:|---:|---:|
| Intrínseca | < 1 s | 0 | < 1 s |
| Permutation (29 atributos × 10 repetições) | 15–25 s | 0 | 4,8 min |
| TreeSHAP (10 mil linhas) | 2 s (XGBoost) – 2,2 min (RF) | 0 | 11,3 min |
| H de Friedman (até 300 pares) | 28–40 s | 0 | 5,7 min |
| **PyGAwLLfs** | 5,2 min (LogReg) – 30 min (XGBoost) | **3.200–4.100** (média por modelo) | **4,9 h** |

Os quatro primeiros explicam um modelo já treinado. O PyGAwLLfs retreina o
modelo milhares de vezes, mesmo numa subamostra de 50 mil voos. Em troca,
entrega também a seleção de atributos (x\*) e a matriz de interação.

### 6.6 Quadro-síntese dos quatro critérios

| Modelo | Método | Concordância¹ | Estabilidade² | Fidelidade top-5³ | Custo por execução |
|---|---|---:|---:|---:|---:|
| LogReg | Intrínseca | 0,89 | determinística | 89% | < 1 s |
| LogReg | Permutation | 0,87 | 0,99 | 87% | 15 s |
| LogReg | PyGAwLLfs | 0,83 | 0,89 | **99%** | 5,2 min · 3.175 treinos |
| RF | Intrínseca | 0,83 | determinística | 94% | < 1 s |
| RF | Permutation | 0,69 | 0,99 | **98%** | 25 s |
| RF | SHAP | 0,85 | 1,00 | **97%** | 2,2 min |
| RF | PyGAwLLfs | 0,70 | 0,68 | 94% | 23 min · 4.119 treinos |
| XGBoost | Intrínseca | 0,57 | determinística | 93% | < 1 s |
| XGBoost | Permutation | 0,51 | 0,97 | 97% | 17 s |
| XGBoost | SHAP | 0,70 | 1,00 | 95% | 2 s |
| XGBoost | PyGAwLLfs | 0,44 | 0,66 | 93% | 30 min · 3.955 treinos |

¹ Spearman médio com os outros métodos do mesmo modelo.
² Spearman médio entre as 10 duplas de sementes.
³ Fração do PR-AUC do modelo completo retida no teste; sorteio: 81% (LogReg),
72% (RF e XGBoost). Em negrito, os mais fiéis do modelo pelo IC da diferença
pareada (seção 6.4); no XGBoost, os quatro empatam.

Comparando os modelos pelas explicações: quanto mais complexo o modelo, menos
os métodos concordam entre si (0,87 → 0,77 → 0,56), mas maior é o ganho do
top-5 sobre o acaso. Na LogReg, 5 atributos quaisquer já retêm 81%; no RF e no
XGBoost, só 72%, e os métodos chegam a 93%–98%.

### 6.7 Interações: PyGAwLLfs × H de Friedman

![](figures/interacao_gawll_x_h.png)

*Figura 6 — Cada ponto é um par de atributos: posição no ranking de interação do PyGAwLLfs (x) contra a do H (y).*

| Modelo | Spearman | RBO | Top-10 em comum | Estabilidade PyGAwLLfs | Estabilidade H |
|---|---:|---:|---:|---:|---:|
| RF | 0,21 | 0,07 | 1 de 10 | 0,08 | 0,98 |
| XGBoost | 0,13 | 0,16 | 2 de 10 | 0,11 | 0,98 |

Os dois métodos praticamente não concordam, por duas causas:

1. **Confiabilidade.** Com o orçamento do artigo e 29 atributos, a matriz do
   PyGAwLLfs é pouco confiável (seção 6.3). A confiabilidade da média das 5
   sementes é de 0,71 na LogReg, 0,51 no XGBoost e 0,12 no RF; mais 5 sementes
   a levariam a ~0,83, ~0,68 e ~0,22.
2. **Conceitos diferentes.** O PyGAwLLfs mede a não aditividade do
   **desempenho**; o H, a da **função** do modelo. Atributos redundantes geram a
   primeira e não a segunda: tirar um só não custa, porque o outro compensa,
   mas tirar os dois custa. Nos dados reais, 7 das 15 arestas mais fortes do
   PyGAwLLfs (top-5 de cada modelo) são pares redundantes, como hora × período
   do dia e dia da semana × fim de semana.

![](figures/sensibilidade_agregacao.png)

*Figura 7 — Posição do par no ranking do PyGAwLLfs pelo número de sementes em que ele foi sondado, com o par não sondado contando como zero (laranja) e só com as sementes sondadas (azul, a versão usada).*

**Sensibilidade à agregação das sementes.** Se o par não sondado contasse
como interação zero, a posição do par acompanharia quantas vezes ele foi
sondado (Spearman de 0,39 a 0,64 entre as duas coisas), e não a força da
interação; só com as sementes sondadas, essa relação some (≈ 0). Que pares
foram sondados se reconstrói sem rodar o GA de novo: os três modelos de uma
mesma semente sorteiam os mesmos pares. Com os zeros na média, o Spearman com
o H cai para 0,06–0,13, a confiabilidade da média para 0,33 ou menos, e o
top-10 de cada modelo troca de 2 a 6 pares. As conclusões desta seção valem
nas duas versões.

### 6.8 Dataset sintético (gabarito conhecido)

**Importância** (RF e XGBoost):

| Método | Separa relevantes¹ de ruído (AUC) | Rank médio de x1, x2 (interação pura) |
|---|---:|---:|
| PyGAwLLfs | **1,00** | **1,5** |
| Intrínseca | 1,00 | 3,5 |
| SHAP | 1,00 | 3,5 (XGBoost) – 5,5 (RF) |
| Permutation | 0,83 – 0,91 | **1,5** |

¹ Relevantes: as diretas (x1–x4) e as redundantes (x5, x6), que são proxies
legítimos do alvo.

- Só o PyGAwLLfs combina as duas propriedades: separa perfeitamente o
  relevante do ruído **e** põe o par de interação no topo.
- A Permutation é a mais fraca na separação. Num modelo fixo, embaralhar uma
  redundante quase não muda a previsão, porque a original compensa.
- Na LogReg, todos os métodos põem x1 e x2 atrás das aditivas e das
  redundantes (rank ~6 de 10). É limitação do modelo linear, que não consegue
  usar uma interação pura, e não dos métodos.

**Interação** (45 pares):

| Método | Posição de (x1, x2) | Posição média dos pares redundantes (x3, x5), (x4, x6) |
|---|---:|---:|
| PyGAwLLfs (RF, XGBoost) | 1º em todas as sementes | **2,5º** |
| H_abs (RF / XGBoost) | 1º em todas as amostras | 7º / 20º |
| H normalizado (RF / XGBoost) | 1º | 28,5º / 44,5º |

Os dois encontram a interação verdadeira; **só o PyGAwLLfs põe a redundância
logo abaixo dela**. Isso confirma a hipótese da seção 6.7: não é um defeito,
mas outra definição de interação, e é útil para seleção de atributos, porque
indica pares substituíveis.

**Seleção (x\*).** O RF selecionou exatamente {x1, x2, x3, x4} (F1 = 1,0); o
XGBoost acrescentou uma redundante (F1 = 0,89). A LogReg incluiu um ruído (F1 =
0,5), sinal do otimismo de escolher o subconjunto pela validação.

### 6.9 Validação de domínio das interações

![](figures/grafo_interacao.png)

*Figura 8 — Grafo de interação reduzido do PyGAwLLfs (5% das arestas mais fortes), com cada aresta classificada como operacional, redundante ou estrutural.*

| Grupo | Pares | Estrutural | Redundante | Operacional | Corroborados |
|---|---:|---:|---:|---:|---:|
| Candidatos (topo do PyGAwLLfs) | 23 | 0 | 5 | 18 | **5** |
| Controles (fundo do ranking) | 10 | 0 | 1 | 9 | 0 |

- **Corroborados descritivamente:** hora × aeroporto de origem, empresa ×
  aeroporto de origem, hora × empresa, congestionamento no destino ×
  aeroporto de destino e período do dia × duração prevista.
- Os candidatos têm interações descritivas maiores que os controles. Desvio
  médio dos operacionais: 1,48 e 1,27 ponto percentual (2024 e 2025), contra
  0,22 e 0,14 nos controles.
- Mas só 5 dos 18 candidatos operacionais passam no critério completo. Aplicado
  ao top-10 do H, o mesmo teste corroborou **11 de 20 pares**, contra **3 de 20**
  do PyGAwLLfs. Parte dessa vantagem é esperada, porque o H e o teste usam a
  mesma definição de interação.
- Com o par não sondado contando como zero, seriam 24 candidatos e 4
  corroborados, e 6 dos 10 controles seriam pares que o PyGAwLLfs nunca mediu.

![](figures/dominio_hora_x_origem.png)

*Figura 9 — Hora × aeroporto de origem. À esquerda, a taxa de atraso observada; à direita, observado menos o previsto sem interação.*

Partidas de madrugada em Guarulhos (2h: 41% de atraso, contra 25% previstos
sem interação) e no Galeão atrasam muito mais do que "ser esse aeroporto" e
"ser essa hora" explicariam somados. O padrão é compatível com bancos noturnos
de voos internacionais longos. Em Congonhas às 22h ocorre o contrário (−8,6
p.p.), compatível com a restrição de horário de operação do aeroporto.

Em empresa × aeroporto, algumas empresas estrangeiras atrasam muito mais num
aeroporto específico (LAN em Confins: 52% contra 23% previstos; TAP em
Brasília: 46% contra 22%). Em hora × empresa, o efeito da hora **muda de
sinal** conforme a empresa: a TAP atrasa 63% às 6h (contra 20% previstos sem
interação) e 6% às 15h (contra 43%); a SID, 52% às 5h (contra 13%) e 3% às 22h
(contra 35%). Em congestionamento no destino × destino, os horários com só 1–2
chegadas previstas atrasam de 6,5 a 9 pontos acima do previsto no Galeão, em
Confins e em Viracopos. O par período do dia × duração prevista também passou
no critério; ele não tem heatmap.

Todas essas leituras são hipóteses de domínio, não verificadas aqui.

## 7. Limitações

- **Um conjunto de dados e um período de teste** (nov–dez, pico sazonal), que
  penaliza rankings baseados em calendário.
- **Orçamento do PyGAwLLfs:** os parâmetros do artigo deixam a matriz de
  interação esparsa com 29 atributos. A importância é bem mais estável
  (0,66–0,89), e o topo dela se mantém. Não se sabe quantas vezes cada par foi
  sondado dentro de uma execução, só se foi sondado (reconstruído pelo
  sorteio comum aos três modelos).
- **H de Friedman e atributos dependentes:** a dependência parcial avalia
  combinações que não existem nos dados (ex.: hora 3 com período "tarde"),
  inflando o H de alguns pares de calendário no RF.
- **Validação de domínio observacional:** corroboração não é causalidade, e
  ficou restrita aos 15 maiores aeroportos e às 10 maiores empresas.
- **Atributos:** o congestionamento conta só voos realizados, não os
  programados. O impacto medido é de 0,0001 a 0,003 no F1, então ficou como limitação
  documentada.
- **Incerteza:** os ICs medem a variação da amostra de dias do teste, e não a
  do treino; 61 dias é pouco para o bootstrap, e dias vizinhos também se
  parecem. São 87 comparações sem correção para multiplicidade: as diferenças
  no limite (ex.: Permutation × SHAP no RF) podem ser falso positivo.
- **Fora do escopo por decisão:** clima, propagação de atrasos entre voos e
  fidelidade por remoção (ROAR).

## 8. Conclusões

1. **Sobre importância, os métodos convergem.** Origem, empresa e hora lideram
   quase todos os rankings, e o top-5 de qualquer método retém 87%–99% do
   desempenho. A maior fonte de divergência é o **modelo explicado**, não o
   método: compare métodos dentro do mesmo modelo antes de comparar modelos.
2. **Para ranquear importância num modelo já treinado, Permutation e SHAP são
   a melhor relação custo-benefício**: estáveis, fiéis (os mais fiéis no RF;
   empatados com os demais no XGBoost) e baratos. A intrínseca
   é instantânea, mas sofre com colinearidade na regressão logística.
3. **O PyGAwLLfs custa duas ordens de grandeza a mais, mas faz outra coisa.**
   Ele seleciona atributos (x\* com um terço das features retém 93%–100% do
   PR-AUC), é robusto à colinearidade (o mais fiel na LogReg) e foi o que
   melhor recuperou a estrutura do sintético no RF e no XGBoost. Sua
   importância é menos estável entre sementes, mas o topo se mantém.
4. **Sobre interações, os métodos divergem, e a divergência tem explicação.**
   O PyGAwLLfs mede a não aditividade do desempenho e, por isso, também detecta
   **redundância**. O H mede a não aditividade da função do modelo. Os dois
   acham a interação verdadeira quando ela existe.
5. **A matriz de interação do PyGAwLLfs precisa de mais sondagens** do que o
   orçamento do artigo oferece com ~30 atributos. Com 10 atributos (sintético),
   a cobertura é total e o método funciona bem. Com cobertura parcial, o par
   não sondado não pode entrar na média como interação zero: o ranking passaria
   a medir a cobertura.
6. **Do grafo de interação saíram cinco padrões operacionais plausíveis**
   (hora × aeroporto, empresa × aeroporto, hora × empresa, congestionamento no
   destino × destino e período do dia × duração prevista). Para extrair
   conhecimento de domínio, porém, o topo do grafo precisa ser filtrado: quase
   metade das arestas mais fortes (7 de 15) é redundância.
7. **Contribuições metodológicas:** a correção do protocolo do PyGAwLLfs
   (fitness na validação, F1, divisão temporal), a identificação e correção de
   dois bugs do código original, e um protocolo que separa redundância de
   interação usando um gabarito sintético.

**Trabalhos futuros:** aumentar as sondagens de *linkage learning* por geração
e registrar quantas vezes cada par foi sondado; versões do H robustas a
atributos dependentes; incluir clima e propagação de
atrasos; repetir o protocolo em outros períodos de teste.

## 9. Respostas às perguntas de pesquisa

**P1 — Os métodos apontam os mesmos atributos? Depende mais do método ou do
modelo?**
Sim, em grande parte. A concordância é moderada a alta (Spearman 0,70 dentro do
mesmo modelo), e o núcleo é comum: origem, empresa e hora. Depende **mais do
modelo do que do método**: o mesmo método em modelos diferentes concorda menos
(0,60) que métodos diferentes no mesmo modelo (0,70), embora a dispersão entre
as duplas seja grande (0,11 a 0,98). A concordância também cai com a
complexidade do modelo (0,87 → 0,77 → 0,56). O PyGAwLLfs é o que mais diverge
(0,64 contra 0,78 nas duplas sem ele).

**P2 — Os rankings se repetem com outra semente?**
Para Permutation e SHAP, sim (0,97–1,00; pior dupla 0,93), e a intrínseca é
determinística. Para o PyGAwLLfs, só em parte: 0,66–0,89 no ranking inteiro (0,37
na pior dupla do XGBoost), mas 64%–88% do top-5 se repete. A matriz de
interação do PyGAwLLfs quase não se repete (0,08–0,11) com o orçamento do
artigo; a do H, sim (0,98).

**P3 — Os atributos apontados como importantes carregam a informação?**
Sim, para todos os métodos. O top-5 retém 87%–99% do PR-AUC no teste, contra
72%–81% do acaso, e o top-10, 96%–100%; a vantagem sobre o acaso se mantém com
IC de 95% em todos os casos. Com o IC da diferença pareada, os mais fiéis no
top-5 são o PyGAwLLfs na regressão logística (99%, +10 a +12 pontos) e a
Permutation e o SHAP no RF (98% e 97%, +3 a +4 pontos); no XGBoost os quatro
métodos empatam. O x\* do PyGAwLLfs retém 93%–100% com um terço dos atributos.

**P4 — Quanto custa cada método?**
Intrínseca: instantânea. Permutation: 15–25 s. TreeSHAP: 2 s a 2,2 min. H de
Friedman: 28–40 s por amostra. Nenhum deles retreina o modelo. O PyGAwLLfs leva
5–31 min e 2.900–4.500 treinos por execução (4,9 h no protocolo), duas ordens de
grandeza a mais, em troca de também selecionar atributos e estimar interações.

**P5 — PyGAwLLfs e H concordam sobre as interações? Qual recupera a verdade?**
Não concordam nos dados reais (Spearman 0,13–0,21). No sintético, **os dois
recuperam a interação verdadeira** (1º lugar em todas as execuções). A
diferença está na redundância: o PyGAwLLfs a põe logo abaixo da interação
verdadeira (2º–3º), e o H não (7º–44º). O PyGAwLLfs mede não aditividade do
desempenho; o H, não aditividade da função. Nos dados reais, a matriz do
PyGAwLLfs é, além disso, pouco confiável com 29 atributos.

**P6 — As interações do PyGAwLLfs correspondem a padrões reais?**
Parcialmente. Os pares do topo têm interações descritivas maiores que os do
fundo, e cinco foram corroborados nos dois anos: hora × aeroporto de origem,
empresa × aeroporto de origem, hora × empresa, congestionamento no destino ×
destino e período do dia × duração prevista. Nenhum controle foi corroborado.
Mas só 5 de 18 pares operacionais passaram, 5 dos 23 pares do topo são
redundância, e o top-10 do H teve mais pares corroborados (11 de 20, contra 3
de 20).

**Pergunta principal — como os métodos se comparam nos quatro critérios?**

- **Concordância:** convergem sobre o que importa (Spearman 0,70 no mesmo
  modelo). O modelo explicado pesa mais que o método, e o PyGAwLLfs é o mais
  divergente.
- **Estabilidade:** Permutation e SHAP ≈ 1; PyGAwLLfs 0,66–0,89, com o topo
  firme. Nas interações, o H é estável e a matriz do PyGAwLLfs não.
- **Fidelidade:** todos ficam muito acima do acaso. Os mais fiéis são a
  Permutation e o SHAP (RF) e o PyGAwLLfs (LogReg); no XGBoost, empate.
- **Custo:** o PyGAwLLfs custa duas ordens de grandeza a mais que os demais.

Em síntese, os métodos **concordam sobre o que importa e discordam sobre o que
interage**. Para explicar um modelo já treinado, Permutation e SHAP têm a
melhor relação entre estabilidade, fidelidade e custo. O PyGAwLLfs é menos
estável e muito mais caro, mas é o único que seleciona atributos, o mais
robusto à colinearidade e o que melhor recuperou a estrutura do dataset
sintético. Sua noção de interação inclui a redundância, o que o torna
complementar, e não substituto, ao H de Friedman.

## Apêndice — Onde está cada coisa

| O quê | Onde |
|---|---|
| Rodar tudo | `python main.py` (`--listar` mostra as 12 etapas e o tempo de cada uma) |
| Testes | `python main.py --testes` ou `python tests/test_*.py` |
| Tabelas de resultados | `results/tables/` (o quadro-síntese em `sintese_*.csv`) |
| Figuras (PNG e PDF) | `results/figures/` |
| Regras metodológicas | `docs/metodologia.md` |
| Detalhe de cada resultado | `docs/fidelidade.md`, `docs/interacoes-h.md`, `docs/sintetico.md`, `docs/validacao-dominio.md` |
| Dados e atributos | `docs/preparacao-dados.md`, `docs/dicionario-features.md` |
| PyGAwLLfs e as correções | `docs/metodologia.md` §4, `src/importance/gawll.py` |
| Decisões em aberto | `docs/decisoes.md` |
| Intervalos de confiança (bootstrap de dias) | `docs/incerteza.md`, `results/tables/incerteza_*.csv` |

## Referências

CRUZ, C. S. BAV – Boletim de Alteração de Voos. In: SEMANA DE QUALIDADE DA
INFORMAÇÃO DO TRANSPORTE AÉREO, 1., 2013, São Paulo. *Apresentação*. Brasília,
DF: ANAC, 2013. Disponível em:
https://www.gov.br/anac/pt-br/assuntos/regulados/empresas-aereas/semana-de-qualidade-da-informacao-do-transporte-aereo/arquivos/i-semana/apresentacao/painel_boletim_de_alteracao_de_voos_empresas_brasileiras.pdf.
Acesso em: 13 out. 2026.

GUIMARÃES, D. G.; CARVALHO, S. S.; NASCIMENTO, R. H.; TINÓS, R. Estimating
variable importance and interaction in machine learning via genetic
algorithms. In: *BRACIS 2025*. Springer, 2026. p. 478–492. (LNAI 16179). DOI:
10.1007/978-3-032-15987-8_32.

As demais referências (algoritmo genético com *linkage learning*, estatística H,
RBO e a base VRA) estão no `README.md`.
