# 5. Avaliação

Quinta fase do CRISP-DM: como os modelos e os métodos são julgados. Código em
`src/avaliacao/` (etapas `desempenho`, `rankings`, `sintetico`, `interacoes`,
`fidelidade`, `recortes` e `sintese`). As figuras que interpretam os resultados
saem de `src/interpretacao/` (etapa `figuras`), com as legendas em
`results/figures/legendas.md`.

## Desempenho dos modelos

**Métricas a partir da matriz de confusão** (`matriz_confusao.py`). Com VP, VN, FP
e FN as contagens de verdadeiros e falsos positivos e negativos (positivo = voo
atrasado), no corte de 0,5:

```
acurácia = (VP + VN) / (VP + VN + FP + FN)
precisão = VP / (VP + FP)
recall   = VP / (VP + FN)
F1       = 2 · precisão · recall / (precisão + recall)
```

O script recalcula as quatro métricas a partir das células e para se alguma não
bater com `baselines.csv`. A **curva ROC** mostra, para cada limiar de
probabilidade, a fração de atrasos detectados contra a fração de pontuais marcados
como atrasados; a AUC é a área sob a curva, e 0,5 é o acaso.

**Diagnóstico de ajuste:** a acurácia no treino, na validação e no teste de cada
modelo, contra o trivial (0,5). Uma diferença grande entre treino e validação
indica sobreajuste; a queda da validação (set–out) para o teste (nov–dez) pode vir
também da mudança de período, que o entendimento dos dados mostra (o teste é o
pico sazonal).

**Acurácia por dia e erro-padrão** (`erros.py`). A acurácia de cada modelo é
calculada em cada um dos 61 dias do teste. O erro-padrão é

```
EP = desvio-padrão das acurácias diárias / √61
```

e a diferença entre dois modelos é a média das diferenças diárias (pareadas por
dia), com o seu próprio EP. **Regra de leitura:** diferença menor que 2·EP é
empate. Na fidelidade, o EP do modelo com todas as features serve de escala de
ruído. Dias vizinhos se parecem (o atraso de um dia carrega para o seguinte), então
esse EP é otimista: a incerteza real é um pouco maior. A acurácia por hora prevista
de partida mostra em que horas os modelos erram mais. Não é um novo uso do teste:
são os mesmos modelos e as mesmas previsões da avaliação final, e nada é escolhido
a partir disso.

## Verificação dos métodos num problema com gabarito

Antes de comparar os métodos nos dados reais, eles são testados num dataset
sintético cuja estrutura é conhecida (`sintetico_gerar.py`, 5.000 linhas, 30% de
validação):

```
z = 2 · x1 · x2 + x3 + x4 + 0,5 · ruído,      y = 1 se z > 0
```

| Atributos | Papel |
|---|---|
| x1, x2 | interação pura: sozinhos não dizem nada; juntos, são o sinal mais forte |
| x3, x4 | informativos, com efeito aditivo |
| x5, x6 | redundantes: cópias de x3 e x4 com correlação 0,9, fora de z |
| x7 a x10 | ruído |

Os mesmos modelos, hiperparâmetros e funções dos dados reais rodam nesse dataset
(`sintetico_aplicar.py`), e `sintetico_recuperar.py` mede a recuperação contra o
gabarito: se os relevantes ficam à frente do ruído, se o par (x1, x2) é o primeiro
em interação e onde caem os pares direto-redundante (x3, x5) e (x4, x6). Redundante
não é erro de importância: é uma *proxy*. A hipótese é que o PyGAwLLfs, por medir a
não aditividade do desempenho, ponha os pares redundantes no topo das interações
(tirar um só quase não custa; tirar os dois custa), e que o H de Friedman, por medir
a função do modelo, não ponha.

## Comparação das importâncias

**Por rank, nunca por valor:** as escalas dos métodos não se comparam (variação de
fitness, queda de acurácia, impureza, log-odds). Cada (método, modelo) vira um
ranking pela média das sementes (`rankings.py`), e dois rankings são comparados pela
**correlação de Spearman** e pela **sobreposição top-k** (das k features mais
importantes de um, quantas estão nas k do outro; k = 5, 10 e 15). Kendall tau-b e
RBO também são calculados e salvos. Nas figuras que mostram magnitude, cada método
é normalizado para [0, 1] (valores negativos da Permutation viram 0).

**Primeiro dentro do mesmo modelo:** comparar métodos aplicados a modelos
diferentes misturaria a divergência do método com a do modelo. A comparação entre
modelos é uma pergunta separada: os métodos concordam mais entre si no mesmo modelo
do que em modelos diferentes?

## Os quatro critérios

| Critério | Como é medido | Saídas |
|---|---|---|
| **Concordância** | Spearman e top-k entre os rankings dos métodos, dentro de cada modelo | `concordancia.csv`, figuras `concordancia_intra_modelo` e `dispersao_gawll_x_rf` |
| **Estabilidade** | Spearman e top-k entre as 10 duplas de sementes de cada (método, modelo); a intrínseca é determinística | `estabilidade.csv`, figura `estabilidade_sementes` |
| **Fidelidade** | acurácia no teste do modelo retreinado só com as top-k features de cada método (k = 5, 10, 15), contra 10 sorteios de k features, o modelo com todas e o acaso (0,5); inclui o subconjunto x* de cada execução do PyGAwLLfs | `fidelidade.csv`, `fidelidade_resumo.csv`, figura `fidelidade` |
| **Custo** | tempo e número de treinos de modelo por execução | `custo_importancia.csv` |

A síntese junta os quatro por método e por modelo (`sintese_*.csv`). A fidelidade
usa o treino completo, a validação e, por último, o teste; as listas de features
vêm todas de etapas anteriores, calculadas sem o teste.

## Estabilidade entre recortes temporais

`recortes.py` repete as quatro importâncias em 2024 e em 2025, com o mesmo
calendário e os hiperparâmetros do experimento principal, e compara os rankings dos
dois anos (Spearman e top-k). Ao lado, o Spearman entre sementes do experimento
principal. No PyGAwLLfs cada recorte tem uma semente só, então a variação entre
recortes mistura período e semente. **Regra de leitura:** se o Spearman entre
recortes é parecido com o entre sementes, o período não acrescenta instabilidade;
se é bem menor, a importância muda de um ano para o outro.

## A subamostra do PyGAwLLfs muda o que é importante?

O PyGAwLLfs explica modelos treinados em cerca de 45.800 voos; os outros métodos
explicam o modelo treinado em 557.518. `checagem_subamostra.py` treina cada modelo
na subamostra, calcula a intrínseca e a Permutation e compara com as do modelo
completo (`checagem_subamostra.csv`). **Regra de leitura:** Spearman alto (≥ 0,8)
quer dizer que a subamostra preserva a estrutura de importância, e as diferenças
entre o PyGAwLLfs e os outros métodos vêm do método; baixo quer dizer que parte da
divergência vem do tamanho dos dados.

## Interações

`interacoes.py` compara, no Random Forest e no XGBoost, o ranking de pares do
PyGAwLLfs (v_int) com o do H de Friedman (H_abs), nos 406 pares.

- **Agregação das sementes:** no PyGAwLLfs, a média de um par usa só as execuções em
  que ele foi sondado (o zero de um par não sondado não é medida), e só entram no
  ranking os pares sondados em pelo menos 2 das 5 execuções.
- **Tipo do par**, lido da definição dos atributos (`graph.tipo_por_construcao`),
  sem nada estimado nos dados:
  - **estrutural:** as duas features codificam a mesma coluna do VRA (duas
    *dummies* de tipo de linha; o porte e o encoding do mesmo aeroporto);
  - **redundante:** uma é função da outra (`hora` × `periodo_dia`, `dia_sem` ×
    `fim_semana`, `mes` × `trimestre`, `mes` × `dia_ano`, `trimestre` ×
    `dia_ano`, `dia_mes` × `dia_ano`);
  - **operacional:** os demais.
- **Saídas:** concordância e estabilidade de cada método (`interacoes.csv`), a
  posição dos pares de cada tipo (`interacoes_redundancia.csv`), cada par com o seu
  valor e rank nos dois métodos (`interacoes_pares.csv`) e as figuras
  `interacao_gawll_x_h` e `grafo_interacao` (o grafo reduzido: as 5% arestas mais
  fortes de cada modelo, como no artigo).

## Exemplos trabalhados

`src/interpretacao/exemplos.py` mostra, com números reais, que os métodos baseados
no desempenho são diferenças de diagonais de matrizes de confusão: a matriz do
Random Forest antes e depois de embaralhar o atributo mais importante (a
Permutation) e as quatro matrizes de uma sondagem do *linkage learning* (x, x⊕g,
x⊕h e x⊕gh), com os fitness, δ e ω (figuras `exemplo_permutation` e
`exemplo_linkage`). Os dois exemplos conferem os seus números com as tabelas do
pipeline.

## Isolamento do teste

Nenhum método de importância ou de interação usa o teste, nem para estimar, nem
para escolher parâmetros. O teste aparece só na avaliação final dos modelos
(incluindo a matriz de confusão, a ROC e a acurácia por dia e por hora, que usam as
mesmas previsões) e na fidelidade, sempre por último. Os testes automáticos
bloqueiam a leitura de `test.parquet` e falham se algum método tentar.

## Limitações

1. **O teste 50/50 não é a operação real:** na base, cerca de 1 em cada 5 voos
   atrasa. O balanceamento torna a acurácia legível e os métodos comparáveis, mas as
   métricas no teste não estimam o desempenho em produção.
2. **O erro-padrão por dias é otimista:** dias vizinhos não são independentes.
3. **A interação de um par não é validada contra a operação aérea:** o trabalho
   compara dois métodos entre si e contra um gabarito sintético; confirmar que uma
   interação existe nos voos exigiria um estudo observacional próprio.
4. **Uma semente por recorte no PyGAwLLfs:** a variação entre recortes mistura
   período e semente (ver a regra de leitura acima).
5. **Matriz de interação do PyGAwLLfs com orçamento limitado:** mesmo com 600
   sondagens, cerca de um quarto dos pares fica sem sondagem em cada execução, e um
   par sondado poucas vezes guarda o máximo de poucos retreinos.
