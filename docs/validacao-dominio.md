# Validação de domínio das interações (Diferencial 4) — resultados

Resultados da Etapa 15, base para a seção correspondente do Cap. 4.
Script: `src/domain/validate_pairs.py`; tabelas: `results/tables/dominio_pares.csv`
e `dominio_celulas.csv`; figuras: `grafo_interacao`, `dominio_hora_x_origem`,
`dominio_hora_x_tipo_linha` (em `results/figures/`). Escolhas e como mudá-las:
`docs/decisoes.md` §11.

## 1. Pergunta e método, em resumo

Os pares que o PyGAwLLfs aponta como interativos correspondem a padrões reais
de atraso? Para cada par, compara-se a taxa de atraso observada em cada célula
do cruzamento das duas variáveis com a taxa prevista por um modelo **só com os
efeitos isolados** de cada uma (regressão logística sem termo de interação).

- **Pares:** 25 candidatos (a união do top-10 de vint médio de cada modelo) e
  10 controles (do fundo do ranking, sem repetir o par de domínio de um candidato).
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
| Candidatos (top-10 dos modelos) | 25 | 1 | 5 | 19 | **2** |
| Controles (fundo do ranking) | 10 | 0 | 0 | 10 | 0 |

- **Redundância concentrada no topo.** No grafo reduzido (20 arestas por
  modelo), 11 das 60 arestas (18%) são pares redundantes e 1 é estrutural. No
  top-5 de cada modelo, são **8 das 15 (53%)**: `dia_ano` × `trimestre`,
  `mes` × `dia_ano`, `dia_mes` × `dia_ano`, `hora` × `periodo_dia`,
  `mes` × `trimestre`. É o mesmo fenômeno do Diferencial 1: o PyGAwLLfs mede
  interação no fitness, e duas variáveis redundantes produzem exatamente esse
  sinal (tirar uma não custa, tirar as duas custa).
- **Candidatos × controles.** Entre os pares operacionais, o desvio médio dos
  candidatos foi de 0,88 p.p. (2024) e 0,71 p.p. (2025), com máximo de 2,8; o
  dos controles, 0,26 e 0,27, com máximo de 0,52. Os pares que o PyGAwLLfs
  põe no topo têm interações descritivas maiores que os do fundo.
- **Detectável não é relevante.** Com 1,7 milhão de voos, quase todo par passa
  do limiar de ruído — inclusive controles. O que separa é o tamanho do desvio
  e a replicação: os controles até se repetem entre anos (correlação média
  0,55), mas com desvios de décimos de ponto.

## 3. Pares corroborados descritivamente

### `hora` × aeroporto de origem (`hora` × `origem_te`)

1º par do Random Forest. Desvio médio de 2,84 p.p. em 2024 e 2,48 em 2025
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

### `hora` × tipo de linha (`hora` × `tipo_linha_N`)

Desvio médio de 1,84 p.p. em 2024 e 1,58 em 2025 (ruído: 0,21); replicação r = 0,70.

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
linha** — interação no sentido mais claro. ⚠️ A interpretação depende do
significado das letras: pelo volume, I e N devem ser linhas internacionais e
nacionais, e C e G possivelmente cargueiras; **confirmar no dicionário do VRA**
antes de usar essa leitura no texto (pendente em `docs/dicionario-features.md`).

## 4. Pares não corroborados (também são resultado)

- **Redundantes (5) e estrutural (1):** não são interações operacionais, por
  construção. Isso não é falha do teste, é a leitura correta: são o efeito
  "redundância como interação" do PyGAwLLfs aparecendo nos dados reais.
- **`dia_ano` × empresa:** desvio grande nos dois anos (2,63 e 2,29 p.p.), mas
  **sem replicação** (r = 0,06). O padrão de uma semana numa empresa num ano
  não se repete no outro: são choques pontuais (problemas operacionais de uma
  empresa em semanas específicas), não uma interação estável entre
  sazonalidade e empresa. O mesmo vale para `hora` × `dia_ano` (1,96 e 1,57;
  r = 0,18).
- **Quase lá:** `hora` × `trimestre` (1,32 e 0,86 p.p.; r = 0,34),
  `tipo_linha` × origem (0,75 e 1,18; r = 0,60) e `trimestre` × tipo de linha
  (1,01 e 0,36; r = 0,40) passam em um ano e falham no outro. Sensibilidade:
  com piso de 0,75 p.p. o resultado não muda; com 0,5 p.p., `tipo_linha` ×
  origem passa a corroborado, e nenhum controle passa em nenhum dos casos.
- **Desvios pequenos:** os demais ficam abaixo de 1 p.p. nos dois anos (ex.:
  `dia_sem` × congestionamento na origem, 0,48 e 0,53; `feriado` × origem, 0,08
  e 0,07). Detectáveis em parte, mas sem relevância prática.

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

- O top do grafo do PyGAwLLfs mistura três coisas: redundância (laranja no
  grafo), estrutura do encoding (cinza) e interações operacionais (azul). Só
  as do terceiro tipo são candidatas a conhecimento de domínio — e, dessas, 2
  de 19 passaram no critério rigoroso.
- Isso reforça a hipótese de `docs/sintetico.md` §3 nos dados reais.
  A outra metade (o H de Friedman não acusa os pares redundantes) é verificada
  na Etapa 13, que pode usar `dominio_pares.csv` para comparar os mesmos pares.
