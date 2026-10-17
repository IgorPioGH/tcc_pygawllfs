# Validação de domínio das interações (Diferencial 4) — resultados

Resultados da Etapa 15, base para a seção correspondente do Cap. 4.
Script: `src/domain/validate_pairs.py`; tabelas: `results/tables/dominio_pares.csv`
e `dominio_celulas.csv`; figuras: `grafo_interacao`, `dominio_hora_x_origem`,
`dominio_hora_x_congestion_origem`, `dominio_empresa_x_origem`,
`dominio_hora_x_tipo_linha` (em `results/figures/`). Escolhas e como mudá-las:
`docs/decisoes.md` §11 e §15.

## 1. Pergunta e método, em resumo

Os pares que o PyGAwLLfs aponta como interativos correspondem a padrões reais
de atraso? Para cada par, compara-se a taxa de atraso observada em cada célula
do cruzamento das duas variáveis com a taxa prevista por um modelo **só com os
efeitos isolados** de cada uma (regressão logística sem termo de interação).

- **Pares:** 26 candidatos (a união do top-10 de vint médio de cada modelo) e
  10 controles (do fundo do ranking, sem repetir o par de domínio de um
  candidato). O vint médio usa só as sementes em que o par foi sondado, e o
  ranking só inclui pares sondados em pelo menos 2 das 5 sementes
  (`docs/decisoes.md` §15); os controles, portanto, são pares medidos e fracos.
- **Dados:** treino + validação (1.697.911 voos, jan/2024–out/2025). O teste
  fica de fora (§4.9).
- **Leitura dos pares:** cada feature volta à variável que codifica (ex.:
  `origem_te` → aeroporto de origem, top-15 por volume, 70,6% dos voos;
  dummies de porte → porte; `dia_ano` → semana do ano).
- **Critério de "corroborado descritivamente"**, nos dois anos (2024 e 2025):
  desvio médio acima do p95 do ruído (bootstrap paramétrico, 200 réplicas) **e**
  de pelo menos 1 ponto percentual; e o padrão de desvios por célula se
  **repete** entre os anos (correlação ≥ 0,5). A exigência de replicação
  separa uma interação operacional estável de choques de dias específicos.
- **Antes do teste**, cada par é classificado por tipo: *estrutural* (as duas
  features são a mesma variável), *redundante* (uma determina a outra,
  coeficiente de incerteza ≥ 0,8) ou *operacional*.

## 2. Resultado geral

| Grupo | Pares | Estrutural | Redundante | Operacional | Corroborados |
|---|---:|---:|---:|---:|---:|
| Candidatos (top-10 dos modelos) | 26 | 0 | 6 | 20 | **4** |
| Controles (fundo do ranking) | 10 | 0 | 0 | 10 | 0 |

- **Redundância concentrada no topo.** No grafo reduzido (20 arestas por
  modelo), 10 das 60 arestas (17%) são pares redundantes. No top-5 de cada
  modelo, são **6 das 15 (40%)**: `dia_ano` × `trimestre` (LogReg e RF),
  `hora` × `periodo_dia`, porte da origem × origem, `mes` × `dia_ano` e
  `dia_sem` × `fim_semana`. É o mesmo fenômeno do Diferencial 1: o PyGAwLLfs
  mede interação no fitness, e duas variáveis redundantes produzem exatamente
  esse sinal (tirar uma não custa, tirar as duas custa).
- **Candidatos × controles.** Entre os pares operacionais, o desvio médio dos
  candidatos foi de 1,32 p.p. (2024) e 1,13 p.p. (2025), com máximo de 3,8; o
  dos controles, 0,19 e 0,17, com máximo de 0,62. Os pares que o PyGAwLLfs
  põe no topo têm interações descritivas maiores que os do fundo.
- **Detectável não é relevante.** Com 1,7 milhão de voos, quase todo par passa
  do limiar de ruído — inclusive controles. O que separa é o tamanho do desvio
  e a replicação: os controles até se repetem entre anos (correlação média
  0,47), mas com desvios de décimos de ponto.
- **Sensibilidade à agregação das sementes** (`docs/decisoes.md` §15). Com os
  pares não sondados contando como interação 0 (versão anterior), eram 25
  candidatos e 2 corroborados (`hora` × origem e `hora` × tipo de linha, que
  continuam), com 8 redundantes nas 15 arestas do top-5. E 5 dos 10 controles
  eram pares que o PyGAwLLfs **nunca sondou**: estavam no fundo do ranking por
  falta de medida, não por interação fraca. Nas duas versões, nenhum controle
  é corroborado.

## 3. Pares corroborados descritivamente

### `hora` × aeroporto de origem (`hora` × `origem_te`)

3º par do Random Forest. Desvio médio de 2,84 p.p. em 2024 e 2,48 em 2025
(ruído: 0,60); replicação r = 0,53.

| Célula | Voos | Taxa observada | Prevista sem interação | Desvio |
|---|---:|---:|---:|---:|
| SBGR (Guarulhos), 2h | 1.634 | 41,2% | 25,3% | +16,0 p.p. |
| SBGL (Galeão), 1h | 1.777 | 34,9% | 22,6% | +12,3 p.p. |
| SBGR, 3h | 3.484 | 30,3% | 19,7% | +10,6 p.p. |
| SBSV (Salvador), 22h | 1.387 | 4,4% | 15,4% | −11,0 p.p. |
| SBSP (Congonhas), 22h | 3.242 | 13,6% | 22,2% | −8,6 p.p. |

Leitura para a discussão (hipóteses, não verificadas aqui): as partidas de
madrugada nos dois principais aeroportos internacionais atrasam muito mais do
que "ser Guarulhos" e "ser 2h da manhã" explicariam somados — compatível com
bancos noturnos de voos internacionais de longo curso, que dependem de
aeronaves chegando de outros voos longos. Em Congonhas, o desvio negativo às
22h é compatível com a restrição de horário de operação do aeroporto, que
pressiona a pontualidade no fim do dia. ⚠️ Possível confundidor: a composição
de empresas e de tipos de linha em cada aeroporto e horário (não controlada aqui).

### `hora` × congestionamento na origem (`hora` × `congestion_origem`)

9º par do Random Forest. Desvio médio de 2,15 p.p. em 2024 e 2,12 em 2025
(ruído: 0,32); replicação r = 0,57. O congestionamento é o nº de partidas
previstas na mesma origem e na mesma hora, aqui em quintis.

| Célula | Voos | Taxa observada | Prevista sem interação | Desvio |
|---|---:|---:|---:|---:|
| 0h, 5–8 partidas | 655 | 26,3% | 16,7% | +9,6 p.p. |
| 8h, 1–2 partidas | 15.396 | 20,8% | 12,9% | +7,9 p.p. |
| 7h, 1–2 partidas | 10.563 | 20,7% | 13,8% | +6,9 p.p. |
| 22h, 18–36 partidas | 20.178 | 32,9% | 25,4% | +7,4 p.p. |
| 22h, 3–4 partidas | 4.179 | 12,3% | 20,1% | −7,8 p.p. |

O efeito do congestionamento **muda com a hora**. Às 22h, a origem muito
movimentada atrasa 7 p.p. acima do previsto e a pouco movimentada, 8 abaixo.
De manhã (7h–8h), são os aeroportos com 1–2 partidas na hora que atrasam acima
do previsto. Leitura para a discussão (hipóteses, não verificadas aqui): no fim
do dia, os grandes aeroportos acumulam os atrasos das rotações anteriores; de
manhã, o primeiro voo de um aeroporto pequeno depende de uma única aeronave
pernoitada. ⚠️ O congestionamento conta só voos realizados (limitação 2,
`docs/decisoes.md` §2), e o porte do aeroporto é um confundidor provável.

### Empresa × aeroporto de origem (`empresa_te` × `origem_te`)

6º par da LogReg (é também o 1º par do H no XGBoost). Desvio médio de 2,53 p.p.
em 2024 e 1,56 em 2025 (ruído: 0,28); replicação r = 0,61.

| Célula | Voos | Taxa observada | Prevista sem interação | Desvio |
|---|---:|---:|---:|---:|
| LAN em SBCF (Confins) | 448 | 52,2% | 23,4% | +28,8 p.p. |
| TAP em SBBR (Brasília) | 591 | 46,0% | 22,0% | +24,0 p.p. |
| CMP em SBGL (Galeão) | 1.586 | 41,1% | 25,7% | +15,4 p.p. |
| PTB em SBRF (Recife) | 551 | 11,4% | 28,1% | −16,7 p.p. |
| CMP em SBEG (Manaus) | 285 | 9,5% | 25,8% | −16,4 p.p. |

Códigos ICAO das empresas. Leitura para a discussão (hipóteses, não
verificadas aqui): os maiores desvios positivos são de empresas estrangeiras
em aeroportos onde elas têm poucos voos internacionais por dia, que dependem
de aeronaves vindas do exterior; a mesma empresa pode ser pontual num
aeroporto e não em outro (CMP: +15 no Galeão, −16 em Manaus). ⚠️ Confundidor
provável: o tipo de linha (internacional) e o horário desses voos.

### `hora` × tipo de linha (`hora` × `tipo_linha_N`)

7º par da LogReg. Desvio médio de 1,84 p.p. em 2024 e 1,58 em 2025 (ruído:
0,20); replicação r = 0,70.

| Célula | Voos | Taxa observada | Prevista sem interação | Desvio |
|---|---:|---:|---:|---:|
| tipo C, 2h | 1.073 | 51,2% | 21,9% | +29,2 p.p. |
| tipo C, 3h | 1.349 | 44,6% | 16,5% | +28,1 p.p. |
| tipo I, 5h–6h | ~14.000 | 32,2% | ~15,5% | +16,8 p.p. |
| tipo G, 8h | 1.051 | 66,8% | 49,8% | +17,0 p.p. |
| tipo G, 12h | 1.219 | 35,3% | 56,3% | −21,0 p.p. |
| tipo C, 22h | 1.351 | 6,8% | 27,6% | −20,8 p.p. |

As linhas do tipo N (a maioria dos voos) ficam perto do previsto: o padrão
vem dos outros tipos. O efeito da hora **muda de sinal conforme o tipo de
linha** — interação no sentido mais claro. Pela definição da ANAC (CRUZ,
2013, slide 17), C é doméstica cargueira, G internacional cargueira, I
internacional mista e N doméstica mista (`docs/dicionario-features.md`). Leitura para a discussão
(hipóteses, não verificadas aqui): os cargueiros domésticos de madrugada (2h–3h)
atrasam mais que o dobro do esperado, e os das 22h quase não atrasam — a
programação e as prioridades de um voo de carga diferem das de um voo de
passageiros. Os internacionais de passageiros das 5h–6h atrasam o dobro do
previsto, o mesmo padrão de madrugada do par `hora` × origem (Guarulhos e
Galeão concentram os voos internacionais).

## 4. Pares não corroborados (também são resultado)

- **Redundantes (6):** não são interações operacionais, por construção. Isso
  não é falha do teste, é a leitura correta: são o efeito "redundância como
  interação" do PyGAwLLfs aparecendo nos dados reais.
- **`dia_ano` × empresa:** desvio grande nos dois anos (2,63 e 2,29 p.p.), mas
  **sem replicação** (r = 0,06). O padrão de uma semana numa empresa num ano
  não se repete no outro: são choques pontuais (problemas operacionais de uma
  empresa em semanas específicas), não uma interação estável entre
  sazonalidade e empresa. O mesmo vale para `mes` × `dia_mes` (3,79 e 3,21;
  r = 0,00), `dia_ano` × destino (2,07 e 1,65; r = 0,11), `hora` × `dia_ano`
  (1,96 e 1,57; r = 0,18) e `trimestre` × empresa (1,54 e 1,66; r = −0,21).
- **Quase lá:** `tipo_linha` × origem (0,75 e 1,18 p.p.; r = 0,60), `dia_ano` ×
  `fim_semana` (1,06 e 0,76; r = 0,32) e `trimestre` × destino (1,09 e 0,91;
  r = 0,20) falham no piso de um dos anos ou na replicação. Sensibilidade: com
  piso de 0,75 p.p. o resultado não muda; com 0,5 p.p., `tipo_linha` × origem
  passa a corroborado, mas também um controle (`hora` × porte da origem, 0,62 e
  0,58; r = 0,74) — o piso de 1 p.p. é o que separa candidatos de controles.
- **Desvios pequenos:** os demais ficam abaixo de 1 p.p. nos dois anos (ex.:
  porte da origem × empresa, 0,12 e 0,09; `mes` × porte do destino, 0,10 e
  0,11). Detectáveis em parte, mas sem relevância prática.

## 5. Limitações

1. **Observacional.** "Corroborado descritivamente" não é prova de interação
   causal: um confundidor omitido (ex.: a mistura de empresas por horário)
   pode produzir o mesmo padrão.
2. **Escala logit.** "Sem interação" é definido como efeitos que se somam no
   logit (o padrão para respostas binárias); numa escala aditiva de
   probabilidades, o tamanho das interações mudaria.
3. **Voos não são independentes.** O bootstrap trata cada voo como
   independente, mas voos do mesmo dia compartilham choques (tempo, panes), o
   que subestima o ruído. A exigência de replicação entre anos é a defesa
   contra isso.
4. **Cobertura.** Só os 15 maiores aeroportos de origem e destino (~70% dos
   voos) e as 10 maiores empresas (~92%).
5. **Dummies.** O teste vê a variável, não a dummy: `tipo_linha_I` × origem e
   `tipo_linha_C` × origem viram a mesma tabela. O que se valida é a
   interação entre as variáveis.

## 6. Ligação com o resto do trabalho

- O top do grafo do PyGAwLLfs mistura redundância (laranja no grafo) e
  interações operacionais (azul); com a média com zeros, aparecia também um par
  estrutural (cinza). Só as operacionais são candidatas a conhecimento de
  domínio — e, dessas, 4 de 20 passaram no critério rigoroso.
- Isso reforça, nos dados reais, a hipótese da redundância testada no
  sintético (`docs/sintetico.md` §3).
  A outra metade (o H de Friedman não acusa os pares redundantes) foi
  verificada na Etapa 13 (`docs/interacoes-h.md`).
- **Etapa 13:** o mesmo teste aplicado ao top-10 do H corroborou 9 de 20 pares
  (contra 2 de 20 do top-10 do PyGAwLLfs, nos pares em que o H foi calculado).
  E o ranking de pares do PyGAwLLfs se mostrou pouco confiável entre execuções
  (cada par é sondado ~0,7 vez por execução): mesmo com a média só das
  sementes sondadas, um par do top tem em média 2–3 sondagens, e parte deles
  está lá por ruído de retreino.
