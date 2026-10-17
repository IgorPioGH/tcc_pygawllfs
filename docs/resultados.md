# Resultados consolidados — índice para o Cap. 4

Todos os resultados do TCC, organizados pelos quatro critérios da pergunta de
pesquisa (concordância, estabilidade, fidelidade e custo) e pelos dois
diferenciais. Para cada um: o número principal, a tabela e a figura de onde
ele sai, e o documento que o interpreta. Tabelas em `results/tables/`,
figuras em `results/figures/` (PNG e PDF). Resultados congelados em 12 out
2026 (marco de 8 nov antecipado), salvo mudança decidida com o orientador
(`docs/decisoes.md`).

## 0. Dados e modelos

| O quê | Número principal | Fonte |
|---|---|---|
| Base limpa | 1.855.980 voos, 18,5% atrasados | `docs/preparacao-dados.md`; figura `histograma_atraso` |
| Split temporal | treino 1,54 mi (18,1%) · validação 158 mil (16,2%) · teste 158 mil (24,3%) | `docs/preparacao-dados.md` |
| Atributos | 29 (dicionário e limitações) | `docs/dicionario-features.md` |
| Modelos no teste | F1: LogReg 0,42 · RF 0,44 · XGBoost 0,45 (trivial: 0) | `baselines.csv` |

## 1. Concordância entre métodos (Etapa 11)

- **Dentro do mesmo modelo os métodos concordam mais:** Spearman médio 0,70,
  contra 0,60 (mesmo método, modelos diferentes) e 0,56 (tudo diferente).
- **O PyGAwLLfs é o que menos concorda com os outros:** 0,62 nas duplas com
  ele × 0,79 nas sem ele.
- Fontes: `concordancia.csv`, `concordancia_resumo.csv`, `ranking_consolidado.csv`;
  figuras `concordancia_intra_modelo`, `concordancia_geral`.

## 2. Estabilidade entre sementes (Etapas 11 e 13)

- **Importância:** Permutation e SHAP ≥ 0,99 (Spearman entre sementes);
  PyGAwLLfs 0,56–0,80, com o topo razoavelmente estável (top-5 0,68–0,84).
- **Interação:** H de Friedman 0,90–0,98; PyGAwLLfs 0,04–0,06 — o ranking de
  pares quase não se repete entre execuções (cada par é sondado ~0,7 vez por
  execução com N = 29). A média das 5 sementes é mais confiável (split-half
  0,06 no RF, 0,24 no XGBoost, 0,38 na LogReg), mas ainda baixa.
- Fontes: `estabilidade.csv`, `estabilidade_features.csv`, `interacoes.csv`;
  figura `estabilidade_sementes`. Interpretação: `docs/interacoes-h.md` §3.

## 3. Fidelidade por retenção (Etapa 12)

- **Todos os métodos são muito melhores que o acaso no top-5:** 87%–98% do
  PR-AUC do modelo completo, contra 72%–81% de 5 features sorteadas. Do
  top-10 em diante, todos convergem para ~100%.
- **Mais fiéis no top-5:** Permutation no RF e no XGBoost (97%–98%);
  **PyGAwLLfs na LogReg** (98%, contra 87%–89% dos outros).
- **x* do PyGAwLLfs:** ~1/3 das features, 95%–99% do PR-AUC no teste.
- Fontes: `fidelidade.csv`, `fidelidade_resumo.csv`; figura `fidelidade`.
  Interpretação: `docs/fidelidade.md`.

## 4. Custo computacional

| Método | Execuções × modelos | Tempo total | Treinos de modelo |
|---|---|---:|---:|
| Intrínseca | 1 × 3 | ~2 s | 0 (lê o modelo treinado) |
| Permutation (29 × 10 repetições) | 5 × 3 | 5,4 min | 0 |
| TreeSHAP (10 mil linhas) | 5 × 2 | 11,7 min | 0 |
| H de Friedman (até 351 pares) | 5 × 2 | 7,2 min | 0 |
| **PyGAwLLfs** (100 gerações) | 5 × 3 | **5,4 h** | **3.200–4.400 por execução** |

Os quatro primeiros explicam um modelo já treinado (o da Etapa 6: 4 s na
LogReg, 16 s no RF e 3 s no XGBoost, no treino completo). O PyGAwLLfs
retreina o modelo milhares de vezes na subamostra de 50 mil voos — e, em
troca, entrega também a seleção de atributos (x*). Fonte: `custo_importancia.csv`;
detalhes do GA em `docs/piloto-custo.md`.

## 5. Interações: PyGAwLLfs × H de Friedman (Etapa 13)

- **Não concordam:** Spearman 0,15–0,16; top-10 com 0–2 pares em comum.
- **Causas:** baixa confiabilidade da matriz do PyGAwLLfs com este orçamento
  e conceitos diferentes (não aditividade do desempenho × da função do modelo).
- **Teste de domínio dos tops:** H corroborou 9 de 20 pares; PyGAwLLfs, 2 de 20.
- **Sensibilidade à agregação das sementes:** com os pares não sondados
  contando como zero, o ranking do PyGAwLLfs acompanhava a cobertura (Spearman
  0,43–0,68 com o nº de sementes sondadas; ≈ 0 na agregação atual). As
  conclusões acima valem nas duas (`docs/decisoes.md` §15).
- Fontes: `interacao_h.csv`, `interacoes*.csv`, `sensibilidade_*.csv`; figuras
  `interacao_gawll_x_h`, `sensibilidade_agregacao`. Interpretação:
  `docs/interacoes-h.md`.

## 6. Diferencial 1 — dataset sintético com gabarito (Etapas 10 e 13)

- **Importância (RF, XGBoost):** PyGAwLLfs e intrínseca separam relevantes de
  ruído perfeitamente (AUC 1,0); a Permutation é a mais fraca (0,71–0,91).
- **Interação:** PyGAwLLfs e H põem o par verdadeiro (x1, x2) em 1º; **só o
  PyGAwLLfs põe os pares de redundância em 2º–3º** (H: 7º–44º) — a hipótese
  central confirmada.
- **Seleção:** o RF acha exatamente {x1, x2, x3, x4}.
- Fontes: `recuperacao_sintetico*.csv`, `sintetico_*.csv`. Interpretação:
  `docs/sintetico.md` (a hipótese da redundância, no §3).

## 7. Diferencial 4 — validação de domínio (Etapa 15)

- Dos 26 pares do topo do PyGAwLLfs: 6 redundantes e 20 operacionais → **4
  corroborados descritivamente** (`hora` × origem, `hora` × congestionamento na
  origem, empresa × origem e `hora` × tipo de linha); nenhum dos 10 controles.
  Com a média com zeros, eram 25 pares e 2 corroborados (`docs/decisoes.md` §15).
- Redundância concentrada no topo: 6 das 15 arestas do top-5 dos três modelos.
- Fontes: `dominio_pares.csv`, `dominio_celulas.csv`; figuras
  `grafo_interacao` e `dominio_*` (um heatmap por par corroborado).
  Interpretação: `docs/validacao-dominio.md`.

## 8. Quadro-síntese

`src/compare/sintese.py` junta os quatro critérios lado a lado, sem recalcular
nada: `sintese_metodos.csv` (por método e modelo), `sintese_modelos.csv` (por
modelo: desempenho no teste, concordância média entre os métodos e fidelidade
média do top-5) e `sintese_top_features.csv` (top-5 de cada ranking). O
relatório final, com as respostas às perguntas de pesquisa, está em
`results/relatorio-final.md`.

## 9. Correções e auditorias

- Correções metodológicas e de implementação (1–5): `docs/metodologia.md` §4
  (§4.1, §4.2, §4.4 e §4.12) e `results/relatorio-final.md` §4.1.
- Auditoria do congestionamento (limitação): `auditoria_congestionamento.csv`.
- Decisões provisórias e como revertê-las: `docs/decisoes.md`.
