# 1. Entendimento do problema

Primeira fase do CRISP-DM (Chapman et al., 2000): o que se quer saber, sobre
qual unidade, em que momento e com que critério de sucesso. As fases seguintes
estão em [`02-dados.md`](02-dados.md), [`03-preparacao.md`](03-preparacao.md),
[`04-modelagem.md`](04-modelagem.md) e [`05-avaliacao.md`](05-avaliacao.md).

## Pergunta de pesquisa

Como diferentes métodos de estimação de importância e de interação de atributos
se comparam, em termos de **concordância**, **estabilidade**, **fidelidade** e
**custo computacional**, quando aplicados a modelos que preveem atrasos de voos?

O método central é o **PyGAwLLfs** (Guimarães et al., 2026), um algoritmo
genético com *linkage learning* que seleciona atributos e, como subproduto,
estima a importância de cada atributo e a interação entre pares. Ele é comparado
com a **importância intrínseca** dos modelos, a **Permutation Importance**, o
**TreeSHAP** e, para as interações, a **estatística H de Friedman** (Friedman e
Popescu, 2008).

## Objetivos

1. Treinar modelos de classificação de atraso (regressão logística, Random Forest
   e XGBoost) que aprendam acima do classificador trivial.
2. Estimar a importância dos atributos em cada modelo com os cinco métodos.
3. Comparar os métodos pela concordância entre os rankings, pela estabilidade
   entre sementes e entre recortes temporais (2024 × 2025), pela fidelidade (as
   features apontadas bastam para prever?) e pelo custo.
4. Comparar as interações apontadas pelo PyGAwLLfs com as do H de Friedman, num
   dataset sintético com gabarito e nos dados reais.

O objetivo não é obter o melhor preditor de atrasos: os modelos são o objeto que
os métodos explicam.

## Unidade de análise, momento da previsão e alvo

- **Unidade de análise:** um voo regular realizado, registrado na base VRA (Voo
  Regular Ativo) da ANAC, de janeiro de 2024 a dezembro de 2025.
- **Momento da previsão:** antes do horário previsto de partida. Por isso só
  entram atributos conhecidos nesse momento, derivados da malha programada
  (horários previstos, empresa, aeroportos, tipo de linha) ou de estatísticas
  calculadas no período de treino. Tudo o que só existe depois do voo (horário
  real, situação, justificativa) fica de fora.
- **Alvo:** o voo é **atrasado** (classe positiva) quando parte 15 minutos ou mais
  depois do horário previsto. O limiar de 15 minutos é o da definição de
  pontualidade do Bureau of Transportation Statistics (BTS) do Departamento de
  Transportes dos Estados Unidos, a convenção da literatura de previsão de
  atrasos. A ANAC usa intervalos de atraso de 30 e de 60 minutos, medidos na
  chegada (Resolução nº 218/2012); aqui o atraso é medido na partida, que é o
  evento previsto.

## Critério de sucesso

- **Dos modelos:** acurácia acima do classificador trivial nos conjuntos de
  validação e de teste (0,5, porque os conjuntos têm metade de voos atrasados)
  e diferença pequena entre treino e validação (sem sobreajuste).
- **Do estudo:** comparar os métodos nos quatro critérios, com as mesmas regras
  para todos e com o conjunto de teste isolado de qualquer escolha.

## Dois tipos de método

Os métodos comparados respondem a perguntas diferentes, e essa diferença guia a
leitura dos resultados:

| Tipo | Métodos | O que medem |
|---|---|---|
| Baseados no **desempenho** | Permutation, PyGAwLLfs (e a fidelidade, como critério) | quanto a acurácia muda sem a informação de um atributo (ou de um par) |
| Baseados na **função** do modelo | importância intrínseca, SHAP, H de Friedman | como o modelo treinado usa cada atributo para chegar à previsão |

Dois atributos redundantes, por exemplo, mostram essa diferença: para um método
baseado no desempenho, tirar um deles quase não custa nada, porque o outro
substitui; tirar os dois custa muito. Esse padrão é exatamente o que o PyGAwLLfs
registra como interação.

## Roteiro do trabalho (CRISP-DM)

| Fase | Documento | Código | Etapas do `main.py` |
|---|---|---|---|
| 1. Entendimento do problema | este documento | — | — |
| 2. Entendimento dos dados | [`02-dados.md`](02-dados.md) | `src/dados/` | `download`, `dados`, `eda` |
| 3. Preparação dos dados | [`03-preparacao.md`](03-preparacao.md) | `src/preparacao/` | `preparacao`, `checagens` |
| 4. Modelagem | [`04-modelagem.md`](04-modelagem.md) | `src/modelagem/` | `baselines`, `pygawllfs`, `importancias` |
| 5. Avaliação | [`05-avaliacao.md`](05-avaliacao.md) | `src/avaliacao/` | `desempenho`, `rankings`, `sintetico`, `interacoes`, `fidelidade`, `recortes`, `sintese` |
| 6. Interpretação | legendas em `results/figures/legendas.md` | `src/interpretacao/` | `figuras` |

A ordem das etapas segue a dependência dos dados: a análise exploratória roda
depois da preparação porque descreve o alvo e os atributos que a preparação cria.

## Referências

- BUREAU OF TRANSPORTATION STATISTICS. *Airline On-Time Statistics and Delay
  Causes*. Washington, DC: U.S. Department of Transportation. Disponível em:
  https://www.bts.gov/explore-topics-and-geography/topics/airlines-and-airports/airline-time-tables.
- AGÊNCIA NACIONAL DE AVIAÇÃO CIVIL (ANAC). Resolução nº 218, de 28 de
  fevereiro de 2012. Estabelece procedimentos para divulgação de dados
  estatísticos de percentuais de atrasos e cancelamentos de voos.
- CHAPMAN, P. et al. *CRISP-DM 1.0: step-by-step data mining guide*. SPSS, 2000.
- FRIEDMAN, J. H.; POPESCU, B. E. Predictive learning via rule ensembles.
  *Annals of Applied Statistics*, v. 2, n. 3, p. 916–954, 2008.
  DOI: 10.1214/07-AOAS148.
- GUIMARÃES, D. G.; CARVALHO, S. S.; NASCIMENTO, R. H.; TINÓS, R. Estimating
  variable importance and interaction in machine learning via genetic
  algorithms. In: *Intelligent Systems: BRACIS 2025*. Springer, 2026.
  p. 478–492. (LNAI 16179). DOI: 10.1007/978-3-032-15987-8_32.
