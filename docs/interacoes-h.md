# Interações: PyGAwLLfs × H de Friedman (Etapa 13) — resultados

Base para a seção de interações do Cap. 4. Scripts:
`src/interaction/friedman_h.py` (H), `src/interaction/graph.py` (grafo reduzido),
`src/compare/interacoes.py` (comparação), `src/compare/sensibilidade_agregacao.py`
(sensibilidade à agregação das sementes). Tabelas: `results/tables/interacao_h.csv`,
`interacoes.csv`, `interacoes_redundancia.csv`, `interacoes_pares.csv`,
`interacoes_dominio.csv`, `sensibilidade_*.csv`. Figuras: `results/figures/interacao_gawll_x_h`
e `sensibilidade_agregacao`. Escolhas e como mudá-las: `docs/decisoes.md` §12, §13 e §15.

**Agregação das sementes no PyGAwLLfs (decisão 15):** a média de cada par usa só
as sementes em que o linkage learning sondou o par, e o ranking só inclui pares
sondados em pelo menos 2 das 5 sementes. A versão anterior punha os zeros dos
pares não sondados na média; a comparação entre as duas está na seção 6.

## 1. Como o H foi calculado

- **Definição** (Friedman e Popescu, 2008): o quanto a dependência parcial
  conjunta de um par (PD_jk) não é explicada pela soma das individuais
  (PD_j + PD_k). Duas versões: a original, **normalizada** (H², de 0 a ~1), e
  a **não normalizada** (H_abs, em unidades de logit, como em Inglis et al.,
  2022), que é a principal para ordenar pares.
- **Escala logit**, a mesma da validação de domínio (Etapa 15).
- **Modelos:** RF e XGBoost. A LogReg é aditiva no logit, então seu H é zero
  por construção (conferido: H_abs < 10⁻⁶ em `tests/test_interacao.py`).
  Consequência: **toda** interação que o PyGAwLLfs aponta na LogReg é do
  fitness, não da função do modelo.
- **Pares (§4.7):** todos os pares da união entre o top-15 por consenso de
  importância e as features do grafo reduzido do PyGAwLLfs (nas duas
  agregações da decisão 15) — 25 features (300 pares) no RF, 27 (351 pares) no
  XGBoost. De 10 a 12 features por modelo só entraram por estarem no grafo do
  PyGAwLLfs: o critério só por importância as teria excluído.
- **Amostra:** 500 linhas da validação por semente, 5 sementes; o valor final
  é a média.
- **Validação da implementação:** em f = x1·x2 + x3, H²(x1, x2) = 0,99 e
  exatamente 0 em todos os pares aditivos.
- **Custo:** 48 s por semente no RF e 38 s no XGBoost — ~7 min no total, contra
  5,4 h do PyGAwLLfs (e contra o limite de 6 h do §9).

## 2. Resultado principal: PyGAwLLfs e H não concordam sobre os pares

| Modelo | PyGAwLLfs × H_abs: Spearman | RBO | top-10 em comum |
|---|---:|---:|---:|
| RF | 0,16 | 0,09 | 2 de 10 |
| XGBoost | 0,15 | 0,03 | 0 de 10 |

Spearman nos pares com média no PyGAwLLfs (294 de 300 no RF, 344 de 351 no
XGBoost); RBO e top-10 nos pares ranqueados (sondados em ≥ 2 sementes).

Praticamente nenhuma concordância — e, pelo §9, isso é resultado, não falha.
As seções 3 e 4 explicam de onde ela vem.

**Os pares do topo de cada método:**

- **H (RF):** `hora` × empresa, `hora` × tipo de linha, `hora` × `dia_ano`
- **H (XGBoost):** empresa × origem, duração prevista × empresa, `hora` × empresa
- **PyGAwLLfs:** pares de calendário redundantes e pares diversos (`docs/validacao-dominio.md`)

## 3. Primeira causa: a matriz do PyGAwLLfs é pouco confiável com este orçamento

| Ranking de pares | Estabilidade entre sementes (Spearman) | Split-half (2 × 3 sementes) |
|---|---:|---:|
| PyGAwLLfs, RF | 0,04 | 0,06 |
| PyGAwLLfs, XGBoost | 0,06 | 0,24 |
| PyGAwLLfs, LogReg | — | 0,38 |
| H_abs, RF | 0,98 | 0,99 |
| H_abs, XGBoost | 0,90 | 0,89 |

Estabilidade nos pares do H; split-half nos 406 pares (no H, nos pares do
§4.7). No PyGAwLLfs, cada comparação usa os pares sondados nas duas sementes
(ou nas duas metades).

O ranking de pares do PyGAwLLfs quase não se repete de uma execução para
outra. O motivo é estrutural: com os parâmetros do artigo (população 50, 100
gerações), o linkage learning faz exatamente 300 sondagens por execução; com
N = 29 há 406 pares, então cada par é sondado em média 0,7 vez por execução
(~3,7 somando as 5), e ~48% dos pares ficam sem sondagem em cada uma — o que o
sorteio uniforme prevê (210 pares sondados por execução, contra 212 ± 6
esperados). Como o `vint` guarda o **máximo** das poucas sondagens, um único
retreino mais ruidoso define o valor do par. Pela fórmula de Spearman-Brown
aplicada ao split-half, a confiabilidade da média das 5 execuções fica em 0,55
(LogReg), 0,39 (XGBoost) e 0,12 (RF); dobrar para 10 sementes a levaria a
~0,71, ~0,56 e ~0,21. Com os zeros na média (versão anterior), esses números
eram ~0,2, ~0,17 e ≈ 0 (seção 6): parte da instabilidade vinha de tratar par
não sondado como "sem interação".

Isso **não** vale para a importância (`vimp`), que teve estabilidade de
0,56–0,80 (Etapa 11): cada feature aparece em muitas sondagens. É a matriz
de **interação** que fica esparsa. No sintético (N = 10, 45 pares), a
cobertura é de 100% e o problema não aparece.

## 4. Segunda causa: os dois métodos medem coisas diferentes

### Redundância

| Modelo | Método | Pares redundantes: rank médio (percentil) | Redundantes no top-10 |
|---|---|---:|---:|
| RF | PyGAwLLfs | 32,9% | 4 |
| RF | H_abs | 39,0% | 1 |
| RF | H normalizado | 53,7% | 0 |
| XGBoost | PyGAwLLfs | 40,2% | 2 |
| XGBoost | H_abs | 49,4% | 0 |
| XGBoost | H normalizado | 59,9% | 0 |

O PyGAwLLfs põe os pares redundantes mais alto que o H, nos dois modelos.
Mas, no RF, alguns pares de calendário deterministicamente ligados
(`hora` × `periodo_dia`, `dia_ano` × `trimestre`) ficam alto também no H. É uma
limitação conhecida da dependência parcial com variáveis dependentes: para
calcular PD_jk, o H força combinações que não existem nos dados (hora = 3 com
período = tarde), e ali o modelo não é aditivo. No XGBoost, o contraste é
nítido: `mes` × `dia_ano` é o 1º do PyGAwLLfs e o 194º do H; `dia_sem` ×
`fim_semana` é o 4º do PyGAwLLfs e tem H = 0, porque o XGBoost não usa
`fim_semana` (66 dos 351 pares do XGBoost têm H = 0 pelo mesmo motivo).

### No sintético, onde há gabarito (Diferencial 1)

| Método | (x1, x2) — a interação verdadeira | Pares de redundância (x3, x5), (x4, x6) |
|---|---|---:|
| PyGAwLLfs, RF e XGBoost | 1º em todas as sementes | **2º–3º** |
| H_abs, RF | 1º em todas as amostras | 7º |
| H_abs, XGBoost | 1º em todas as amostras | 20º |
| H normalizado, RF / XGBoost | 1º | 28º / 44º |

**A hipótese da redundância (`docs/sintetico.md` §3) está confirmada no sintético:**
os dois métodos acham a interação verdadeira, mas só o PyGAwLLfs põe a
redundância no topo. Nos dados reais ela se confirma em parte: vale com
clareza no XGBoost; no RF, a extrapolação da dependência parcial também faz o
H acusar pares deterministicamente redundantes.

## 5. Qual top corresponde a padrões reais? (teste de domínio da Etapa 15)

O top-10 de cada método passou pelo mesmo teste descritivo da Etapa 15
(`interacoes_dominio.csv`):

| Modelo | Método | Corroborados | Redundantes / estruturais | Não corroborados |
|---|---|---:|---:|---:|
| RF | **H_abs** | **5** | 1 | 4 |
| RF | PyGAwLLfs | 2 | 4 | 4 |
| XGBoost | **H_abs** | **4** | 0 | 6 |
| XGBoost | PyGAwLLfs | 0 | 2 | 8 |

Corroborados do H: `hora` × empresa, `hora` × tipo de linha, `periodo_dia` ×
empresa, `hora` × duração prevista, empresa × origem, `hora` × origem. Do
PyGAwLLfs: `hora` × origem e `hora` × congestionamento na origem (RF).

⚠️ **Ressalva justa ao PyGAwLLfs:** o H e o teste descritivo usam a mesma
definição de interação (não aditividade no logit), então parte dessa
vantagem é esperada por construção. O PyGAwLLfs mede outra coisa — o quanto
o **desempenho** muda ao tirar features juntas — e isso inclui redundância,
que é informação útil sobre seleção de atributos, mas não é interação no
sentido do H.

## 6. Sensibilidade: e se o zero do vint contasse como "sem interação"?

Na versão anterior, um par que o linkage learning não sorteou numa execução
entrava na média das 5 sementes com interação 0. Com ~48% dos pares sem
sondagem em cada execução, a posição do par passou a acompanhar **quantas
vezes ele foi sondado**, e não a força da interação (figura
`sensibilidade_agregacao`):

| Medida | Com zeros | Só sondadas |
|---|---:|---:|
| Spearman(média do par, nº de sementes sondadas): LogReg / RF / XGBoost | 0,43 / 0,68 / 0,58 | −0,01 / 0,02 / −0,02 |
| PyGAwLLfs × H_abs, Spearman: RF / XGBoost | 0,09 / 0,08 | 0,16 / 0,15 |
| PyGAwLLfs × H_abs, top-10 em comum: RF / XGBoost | 1 / 0 | 2 / 0 |
| Split-half: LogReg / RF / XGBoost | 0,11 / −0,00 / 0,09 | 0,38 / 0,06 / 0,24 |
| Redundantes: rank médio (percentil), RF / XGBoost | 27% / 35% | 33% / 40% |
| Top-10 do PyGAwLLfs corroborados no domínio (RF + XGBoost) | 1 de 20 | 2 de 20 |

Como se sabe se o par foi sondado: os três modelos de uma semente sorteiam os
mesmos pares (o sorteio não depende do fitness), então "sondado" = vint > 0 em
algum modelo naquela semente. Um zero com o par sondado é medida e entra na
média: acontece ~15 vezes por execução no XGBoost, nos pares de features que ele
não usa (como `periodo_dia`). Detalhes, ressalvas e a evidência do sintético
(com N = 10, todo par é sondado e as duas agregações coincidem):
`docs/decisoes.md` §15.

- **Não muda:** os dois métodos não concordam; o ranking de pares do RF não se
  repete entre sementes; o PyGAwLLfs põe a redundância mais alto que o H; o
  top-10 do H corrobora mais pares no domínio.
- **Muda:** o artefato de cobertura some; a confiabilidade da LogReg e do
  XGBoost sobe (a instabilidade era em parte cobertura, em parte ruído); o
  top-10 de cada modelo troca de 3 a 7 pares.

## 7. Conclusões para o texto

1. **PyGAwLLfs e H não concordam** sobre os pares de interação nos dados
   reais (Spearman ≈ 0,15), com qualquer das duas agregações das sementes.
2. A discordância tem duas causas: **(a)** com o orçamento do artigo, a
   matriz de interação do PyGAwLLfs para N = 29 é pouco confiável (cada par é
   sondado poucas vezes; e, se o par não sondado conta como zero, o ranking
   passa a medir a cobertura); **(b)** os métodos medem conceitos diferentes —
   não aditividade do desempenho × não aditividade da função —, e a
   redundância separa os dois.
3. **No sintético a hipótese se confirma:** os dois acham a interação
   verdadeira; só o PyGAwLLfs promove a redundância.
4. **Custo:** o H fez o trabalho em ~7 min, contra 5,4 h do PyGAwLLfs — mas o
   PyGAwLLfs entrega também a seleção de atributos e a importância.
5. **Trabalho futuro:** mais sondagens de linkage learning por geração (ou
   mais gerações) para estimar a matriz de interação com confiabilidade; um
   contador de sondagens por par no PyGAwLLfs; versões do H robustas a
   variáveis dependentes.

## 8. Limitações

- H na escala logit (escolha; na escala de probabilidade os valores mudam).
- 500 linhas por amostra (o H varia pouco entre amostras: estabilidade ≥ 0,90).
- Extrapolação da dependência parcial com features dependentes (seção 4).
- O H só foi calculado no conjunto do §4.7 (até 351 pares), não nos 406.
- No PyGAwLLfs, não se sabe quantas vezes cada par foi sondado dentro de uma
  execução, só se foi sondado (`docs/decisoes.md` §15).
- Reprodutibilidade: o SHAP do XGBoost na GPU varia na sétima casa decimal
  entre execuções (float32), sem mudar nenhum rank.
