# Incerteza dos resultados no teste (Etapa 12b) — bootstrap de dias, pareado

Base para os intervalos de confiança do Cap. 4. Script:
`src/compare/incerteza.py`; tabelas: `results/tables/incerteza_modelos.csv`,
`incerteza_modelos_pares.csv`, `incerteza_fidelidade.csv` e
`incerteza_fidelidade_pares.csv`. Escolhas e como mudá-las:
`docs/decisoes.md` §16.

## 1. Pergunta

O teste é uma amostra de voos (nov–dez/2025). Com outra amostra do mesmo
período, o PR-AUC de cada modelo e a fidelidade de cada método sairiam um
pouco diferentes. Quanto? E as diferenças que o texto aponta ("o RF tem o
maior PR-AUC", "a Permutation é a mais fiel") são maiores que essa variação?

## 2. Método

- **Bootstrap por dias.** Sorteiam-se os 61 dias do teste com reposição,
  juntam-se todos os voos dos dias sorteados e recalcula-se a métrica. São
  1.000 repetições (semente 42). O IC de 95% vai do percentil 2,5 ao 97,5.
- **Por que dias, e não voos.** Voos do mesmo dia não são independentes: um
  temporal ou um problema num hub atrasa dezenas de voos ao mesmo tempo. No
  teste, a taxa de atraso diária vai de 9% a 62% (mediana 21%). Sortear voos
  trataria como independentes coisas que não são, e o IC sairia estreito demais.
- **Pareado.** O mesmo sorteio de dias vale para todos os modelos e
  configurações. Para dizer se A é melhor que B, o IC é o da **diferença A − B**,
  calculada em cada repetição. Se ele não contém zero, a diferença não se
  explica pela amostra de dias. Ver se os ICs de A e de B se sobrepõem seria
  conservador demais: boa parte da variação vem dos dias (um dia ruim é ruim
  para os dois modelos) e se cancela na diferença. Exemplo do próprio
  resultado: os ICs do PR-AUC do RF e do XGBoost têm ±0,03 de largura, e o da
  diferença entre eles tem ±0,011.
- **Implementação.** Em vez de copiar os voos repetidos, cada voo recebe peso
  igual ao número de vezes que o seu dia foi sorteado (`sample_weight` das
  métricas do scikit-learn). O valor é o mesmo da reamostragem materializada
  (`tests/test_incerteza.py`), e o bootstrap inteiro leva ~4,5 min.
- **O que entra.**
  - *Modelos (Etapa 6):* F1, PR-AUC e ROC-AUC dos três modelos, com as
    previsões dos `.joblib` salvos. As estimativas pontuais conferem com
    `baselines.csv` (diferença < 10⁻⁹).
  - *Fidelidade (Etapa 12):* PR-AUC das 141 configurações, a partir das
    previsões no teste que o `fidelity.py` salva em
    `experiments/runs/fidelidade/previsoes_teste.npz`. A retenção é o PR-AUC
    médio do grupo (método, k) dividido pelo PR-AUC do modelo completo **na
    mesma repetição**. As estimativas conferem com `fidelidade_resumo.csv`.
- **Isolamento do teste (§4.9).** Isto não reabre o teste para tomar decisões.
  Os modelos e as listas de features já estavam fixados, e as previsões são as
  mesmas das Etapas 6 e 12. O bootstrap só mede quanto cada resultado varia
  com a amostra de dias, e nada foi escolhido a partir dele.

## 3. Modelos

| Modelo | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|
| Regressão logística | 0,417 [0,393; 0,438] | 0,354 [0,332; 0,377] | 0,644 [0,631; 0,658] |
| Random Forest | 0,440 [0,408; 0,473] | 0,403 [0,374; 0,434] | 0,682 [0,671; 0,693] |
| XGBoost | 0,451 [0,417; 0,484] | 0,394 [0,365; 0,427] | 0,676 [0,662; 0,691] |

Diferenças pareadas (IC de 95%):

| Comparação | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|
| RF − XGBoost | −0,010 [−0,018; −0,002] | +0,009 [−0,003; +0,019] | +0,006 [−0,005; +0,015] |
| RF − LogReg | +0,024 [+0,008; +0,041] | +0,049 [+0,038; +0,060] | +0,038 [+0,028; +0,047] |
| XGBoost − LogReg | +0,034 [+0,016; +0,053] | +0,040 [+0,025; +0,056] | +0,032 [+0,018; +0,046] |

- **RF e XGBoost empatam no PR-AUC e no ROC-AUC.** A diferença entre eles não
  se distingue da variação entre amostras de dias. Por isso a frase certa é "os
  dois empatam", e não "o RF tem o maior PR-AUC".
- **No F1, o XGBoost fica à frente do RF** por 0,010. A diferença é real, mas
  pequena, e depende do corte em 0,5 (`docs/decisoes.md` §14).
- **As duas árvores superam a regressão logística** nas três métricas, com
  folga (PR-AUC +0,04 a +0,05).

## 4. Fidelidade

Retenção do PR-AUC do modelo completo, com IC de 95% (em %):

| Modelo | k | Intrínseca | Permutation | SHAP | PyGAwLLfs | Sorteio |
|---|---:|---|---|---|---|---|
| LogReg | 5 | 89 [86; 91] | 87 [85; 89] | — | **99 [97; 101]** | 81 [78; 83] |
| LogReg | 10 | 97 [95; 98] | 97 [96; 98] | — | 97 [96; 99] | 91 [89; 92] |
| RF | 5 | 94 [91; 97] | **98 [96; 100]** | **97 [95; 99]** | 94 [91; 96] | 72 [70; 74] |
| RF | 10 | 97 [95; 99] | 100 [99; 101] | 97 [95; 99] | 99 [98; 100] | 86 [85; 87] |
| XGBoost | 5 | 93 [89; 97] | 97 [93; 101] | 95 [93; 97] | 93 [89; 97] | 72 [70; 74] |
| XGBoost | 10 | 96 [94; 98] | 98 [96; 100] | 100 [99; 101] | 96 [93; 100] | 85 [83; 87] |

Com k = 15, todos os métodos ficam entre 98% e 100% (`incerteza_fidelidade.csv`).

Valores acima de 100% são possíveis: em alguns sorteios de dias o modelo com
k features fica à frente do completo.

Diferenças pareadas no top-5 (pontos percentuais de retenção, IC de 95%):

| Modelo | Comparação | Diferença | Leitura |
|---|---|---|---|
| LogReg | PyGAwLLfs − Intrínseca | +10,2 [+7,6; +12,4] | real |
| LogReg | PyGAwLLfs − Permutation | +12,0 [+10,5; +13,3] | real |
| LogReg | Intrínseca − Permutation | +1,8 [−0,6; +4,1] | empate |
| RF | Permutation − SHAP | +0,9 [+0,1; +1,7] | no limite |
| RF | Permutation − PyGAwLLfs | +4,1 [+2,2; +6,1] | real |
| RF | Permutation − Intrínseca | +4,2 [+1,4; +6,6] | real |
| RF | SHAP − PyGAwLLfs | +3,1 [+1,3; +5,2] | real |
| RF | SHAP − Intrínseca | +3,2 [+0,4; +5,7] | real |
| RF | PyGAwLLfs − Intrínseca | +0,1 [−1,6; +1,5] | empate |
| XGBoost | Permutation − SHAP | +2,1 [−2,5; +6,8] | empate |
| XGBoost | Permutation − Intrínseca | +4,2 [−1,6; +10,1] | empate |
| XGBoost | Permutation − PyGAwLLfs | +4,4 [−1,5; +10,3] | empate |
| XGBoost | SHAP − Intrínseca / PyGAwLLfs | +2,1 / +2,3 (ICs de −1,4 a +6,1) | empate |

Leituras:

1. **Todo método supera o acaso em todos os modelos e em todos os k.** Os 33
   ICs de "método − sorteio" excluem o zero: de +6 a +26 pontos no top-5 e
   pelo menos +2 no top-15.
2. **LogReg: a vantagem do PyGAwLLfs é real** (+10 a +12 pontos sobre a
   intrínseca e a Permutation). Ela é o resultado mais forte da fidelidade.
3. **RF: Permutation e SHAP ficam à frente** da intrínseca e do PyGAwLLfs
   (+3 a +4 pontos), e a diferença é real. Entre Permutation e SHAP, a
   diferença de 0,9 ponto está no limite (IC de +0,1 a +1,7). Com 87
   comparações, é o tipo de resultado que pode ser falso positivo.
4. **XGBoost: os quatro métodos empatam no top-5.** As previsões do XGBoost
   com 5 features variam muito de um dia para outro (ICs de até ±4 pontos), e
   nenhuma diferença entre métodos se sustenta. A frase "a Permutation é a mais
   fiel no XGBoost" não é suportada. No top-10, o SHAP fica à frente do
   PyGAwLLfs (+3,4 [+0,4; +6,8]) e da intrínseca (+3,8 [+1,8; +6,0]).
5. **O x\* do PyGAwLLfs**, por número de features selecionadas: na LogReg,
   100%–101% (IC de 98% a 103%), igual ao modelo completo; no RF, 91%–95% (IC
   de 89% a 97%), abaixo do completo; no XGBoost, 94%–98% (IC de 91% a 101%).

## 5. Estabilidade e concordância: dispersão, sem bootstrap

Estes critérios não usam o teste, então não há amostra de dias a reamostrar.
A variação relevante é entre sementes e entre duplas de rankings, e as
tabelas já a guardam:

- **Estabilidade** (`estabilidade.csv`, Spearman entre as 10 duplas de
  sementes): Permutation e SHAP, média 0,97–1,00 e pior dupla ≥ 0,93. PyGAwLLfs,
  média 0,89 (LogReg), 0,68 (RF) e 0,66 (XGBoost); a pior dupla cai a 0,79,
  0,53 e 0,37.
- **Concordância** (`concordancia_resumo.csv`, Spearman): métodos diferentes no
  mesmo modelo, média 0,70 e duplas de 0,33 a 0,98; mesmo método em modelos
  diferentes, 0,60 (0,11 a 0,84); tudo diferente, 0,57 (0,13 a 0,90). A
  diferença entre as médias é pequena perto da dispersão entre as duplas.

## 6. Limitações

- **O IC é condicional aos modelos treinados.** Ele mede a variação da amostra
  de dias do teste, e não a do treino. No sorteio de features, a faixa cinza
  da figura de fidelidade mostra a variação entre sorteios; o IC acima é o da
  média dos 10.
- **61 blocos** é um número pequeno para o bootstrap. Dias vizinhos também se
  parecem (feriados, semanas de pico), e o bloco de um dia não captura isso: o
  IC real pode ser um pouco mais largo.
- **Comparações múltiplas.** São 87 ICs de 95% sem correção (9 entre modelos e
  78 na fidelidade). Por acaso, espera-se que ~4 excluam o zero sem que haja
  diferença. As conclusões acima se apoiam em diferenças com folga. As que
  ficam no limite estão marcadas como tal.
- **Mesmo período.** O IC responde "e se fossem outros dias de nov–dez/2025?",
  e não "e se fosse outra estação". O teste no pico sazonal continua sendo uma
  limitação (`docs/decisoes.md` §7).
