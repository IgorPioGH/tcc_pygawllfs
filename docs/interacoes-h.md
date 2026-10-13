# Interações: PyGAwLLfs × H de Friedman (Etapa 13) — resultados

Base para a seção de interações do Cap. 4. Scripts:
`src/interaction/friedman_h.py` (H), `src/interaction/graph.py` (grafo reduzido),
`src/compare/interacoes.py` (comparação). Tabelas: `results/tables/interacao_h.csv`,
`interacoes.csv`, `interacoes_redundancia.csv`, `interacoes_pares.csv`,
`interacoes_dominio.csv`. Figura: `results/figures/interacao_gawll_x_h`.
Escolhas e como mudá-las: `docs/decisoes.md` §12 e §13.

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
  importância e as features do grafo reduzido do PyGAwLLfs — 24 features
  (276 pares) no RF, 25 (300 pares) no XGBoost. De 9 a 10 features por modelo
  só entraram por estarem no grafo do PyGAwLLfs: o critério só por
  importância as teria excluído.
- **Amostra:** 500 linhas da validação por semente, 5 sementes; o valor final
  é a média.
- **Validação da implementação:** em f = x1·x2 + x3, H²(x1, x2) = 0,99 e
  exatamente 0 em todos os pares aditivos.
- **Custo:** 49 s por semente no RF e 33 s no XGBoost — ~7 min no total, contra
  5,4 h do PyGAwLLfs (e contra o limite de 6 h do §9).

## 2. Resultado principal: PyGAwLLfs e H não concordam sobre os pares

| Modelo | PyGAwLLfs × H_abs: Spearman | RBO | top-10 em comum |
|---|---:|---:|---:|
| RF | 0,12 | 0,07 | 1 de 10 |
| XGBoost | 0,08 | 0,02 | 0 de 10 |

Praticamente nenhuma concordância — e, pelo §9, isso é resultado, não falha.
As seções 3 e 4 explicam de onde ela vem.

**Os pares do topo de cada método:**

- **H (RF):** `hora` × empresa, `hora` × tipo de linha, `hora` × `dia_ano`
- **H (XGBoost):** empresa × origem, duração prevista × empresa, `hora` × empresa
- **PyGAwLLfs:** pares de calendário redundantes e pares diversos (`docs/validacao-dominio.md`)

## 3. Primeira causa: a matriz do PyGAwLLfs é pouco confiável com este orçamento

| Ranking de pares | Estabilidade entre sementes (Spearman) | Split-half (2 × 3 sementes) |
|---|---:|---:|
| PyGAwLLfs, RF | −0,01 | ≈ 0 |
| PyGAwLLfs, XGBoost | −0,00 | 0,09 |
| PyGAwLLfs, LogReg | — | 0,11 |
| H_abs, RF | 0,98 | 0,99 |
| H_abs, XGBoost | 0,93 | 0,94 |

O ranking de pares do PyGAwLLfs praticamente não se repete de uma execução
para outra. O motivo é estrutural: com os parâmetros do artigo (população 50,
100 gerações), o linkage learning faz ~300 sondagens por execução; com N = 29
há 406 pares, então cada par é sondado em média 0,7 vez por execução (~3,7
somando as 5), e metade dos pares fica com zero em cada uma — exatamente o que
uma Poisson de média 0,74 prevê. Como o `vint` guarda o **máximo** das poucas
sondagens, um único retreino mais ruidoso define o valor do par. Pela fórmula de
Spearman-Brown aplicada ao split-half, a confiabilidade da média das 5
execuções fica em torno de 0,2 (LogReg, XGBoost) e perto de 0 (RF); dobrar
para 10 sementes a levaria só a ~0,3.

Isso **não** vale para a importância (`vimp`), que teve estabilidade de
0,56–0,80 (Etapa 11): cada feature aparece em muitas sondagens. É a matriz
de **interação** que fica esparsa. No sintético (N = 10, 45 pares), a
cobertura é de 100% e o problema não aparece.

## 4. Segunda causa: os dois métodos medem coisas diferentes

### Redundância

| Modelo | Método | Pares redundantes: rank médio (percentil) | Redundantes no top-10 |
|---|---|---:|---:|
| RF | PyGAwLLfs | 26,6% | 3 |
| RF | H_abs | 40,7% | 1 |
| RF | H normalizado | 54,6% | 0 |
| XGBoost | PyGAwLLfs | 36,7% | 2 |
| XGBoost | H_abs | 52,9% | 0 |
| XGBoost | H normalizado | 62,1% | 0 |

O PyGAwLLfs põe os pares redundantes mais alto que o H, nos dois modelos.
Mas, no RF, alguns pares de calendário deterministicamente ligados
(`hora` × `periodo_dia`, `dia_ano` × `trimestre`) ficam alto também no H. É uma
limitação conhecida da dependência parcial com variáveis dependentes: para
calcular PD_jk, o H força combinações que não existem nos dados (hora = 3 com
período = tarde), e ali o modelo não é aditivo. No XGBoost, o contraste é
nítido: `mes` × `dia_ano` é o 1º do PyGAwLLfs e o 187º do H.

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
| RF | PyGAwLLfs | 1 | 3 | 6 |
| XGBoost | **H_abs** | **4** | 0 | 6 |
| XGBoost | PyGAwLLfs | 0 | 3 | 7 |

Corroborados do H: `hora` × empresa, `hora` × tipo de linha, `periodo_dia` ×
empresa, `hora` × duração prevista, empresa × origem, `hora` × origem.

⚠️ **Ressalva justa ao PyGAwLLfs:** o H e o teste descritivo usam a mesma
definição de interação (não aditividade no logit), então parte dessa
vantagem é esperada por construção. O PyGAwLLfs mede outra coisa — o quanto
o **desempenho** muda ao tirar features juntas — e isso inclui redundância,
que é informação útil sobre seleção de atributos, mas não é interação no
sentido do H.

## 6. Conclusões para o texto

1. **PyGAwLLfs e H não concordam** sobre os pares de interação nos dados
   reais (Spearman ≈ 0,1).
2. A discordância tem duas causas: **(a)** com o orçamento do artigo, a
   matriz de interação do PyGAwLLfs para N = 29 é pouco confiável (cada par é
   sondado poucas vezes); **(b)** os métodos medem conceitos diferentes —
   não aditividade do desempenho × não aditividade da função —, e a
   redundância separa os dois.
3. **No sintético a hipótese se confirma:** os dois acham a interação
   verdadeira; só o PyGAwLLfs promove a redundância.
4. **Custo:** o H fez o trabalho em ~7 min, contra 5,4 h do PyGAwLLfs — mas o
   PyGAwLLfs entrega também a seleção de atributos e a importância.
5. **Trabalho futuro:** mais sondagens de linkage learning por geração (ou
   mais gerações) para estimar a matriz de interação com confiabilidade;
   versões do H robustas a variáveis dependentes.

## 7. Limitações

- H na escala logit (escolha; na escala de probabilidade os valores mudam).
- 500 linhas por amostra (o H varia pouco entre amostras: estabilidade ≥ 0,93).
- Extrapolação da dependência parcial com features dependentes (seção 4).
- O H só foi calculado no conjunto do §4.7 (até 300 pares), não nos 406.
- Reprodutibilidade: o SHAP do XGBoost na GPU varia na sétima casa decimal
  entre execuções (float32), sem mudar nenhum rank.
