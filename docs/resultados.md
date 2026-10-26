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
| IC de 95% (bootstrap de dias, pareado) | PR-AUC: LogReg 0,354 [0,332; 0,377] · RF 0,403 [0,374; 0,434] · XGBoost 0,394 [0,365; 0,427]; RF e XGBoost **empatam** no PR-AUC e no ROC-AUC; no F1, XGBoost + 0,010 [0,002; 0,018]; as árvores superam a LogReg nas três métricas | `incerteza_modelos.csv`, `incerteza_modelos_pares.csv`; `docs/incerteza.md` |
| Matrizes de confusão no teste | atrasos detectados: LogReg 53% · RF 85% · XGBoost 75%; pontuais marcados como atrasados: 33% · 65% · 51% | `matriz_confusao.csv`; figura `matriz_confusao` |

## 1. Concordância entre métodos (Etapa 11)

- **Dentro do mesmo modelo os métodos concordam mais:** Spearman médio 0,70,
  contra 0,60 (mesmo método, modelos diferentes) e 0,57 (tudo diferente). A
  dispersão entre as duplas é grande perto dessa diferença: 0,33–0,98, 0,11–0,84
  e 0,13–0,90 (`spearman_min` e `spearman_max`).
- **O PyGAwLLfs é o que menos concorda com os outros:** 0,64 nas duplas com
  ele × 0,78 nas sem ele.
- Fontes: `concordancia.csv`, `concordancia_resumo.csv`, `ranking_consolidado.csv`;
  figuras `concordancia_intra_modelo`, `concordancia_geral`.

## 2. Estabilidade entre sementes (Etapas 11 e 13)

- **Importância:** Permutation e SHAP 0,97–1,00 (Spearman médio entre
  sementes; pior dupla 0,93); PyGAwLLfs 0,66–0,89 (pior dupla 0,37–0,79), com o
  topo razoavelmente estável (top-5 0,64–0,88).
- **Interação:** H de Friedman 0,98; PyGAwLLfs 0,08–0,11 — o ranking de
  pares quase não se repete entre execuções (cada par é sondado ~0,7 vez por
  execução com N = 29). A média das 5 sementes é mais confiável (split-half
  0,07 no RF, 0,34 no XGBoost, 0,55 na LogReg), mas ainda baixa.
- Fontes: `estabilidade.csv`, `estabilidade_features.csv`, `interacoes.csv`;
  figura `estabilidade_sementes`. Interpretação: `docs/interacoes-h.md` §3.

## 3. Fidelidade por retenção (Etapa 12)

- **Todos os métodos são muito melhores que o acaso no top-5:** 87%–99% do
  PR-AUC do modelo completo, contra 72%–81% de 5 features sorteadas. Os 33 ICs
  de 95% da diferença método − sorteio (todos os modelos e k) excluem o zero.
  Do top-10 em diante, todos convergem para ~100%.
- **Mais fiéis no top-5, com IC de 95% da diferença (bootstrap pareado por
  dias):** **PyGAwLLfs na LogReg** (99%, +10 a +12 pontos sobre os outros, real);
  Permutation e SHAP no RF (98% e 97%, +3 a +4 pontos sobre intrínseca e
  PyGAwLLfs, real; entre as duas, +0,9 ponto, no limite). **No XGBoost, os
  quatro empatam** (93%–97%; nenhuma diferença exclui o zero).
- **x* do PyGAwLLfs:** ~1/3 das features, 93%–100% do PR-AUC no teste (média
  por modelo). No RF, abaixo do modelo completo (ICs de 89% a 97%).
- Fontes: `fidelidade.csv`, `fidelidade_resumo.csv`, `incerteza_fidelidade.csv`,
  `incerteza_fidelidade_pares.csv`; figura `fidelidade`. Interpretação:
  `docs/fidelidade.md` e `docs/incerteza.md`.

## 4. Custo computacional

| Método | Execuções × modelos | Tempo total | Treinos de modelo |
|---|---|---:|---:|
| Intrínseca | 1 × 3 | < 1 s | 0 (lê o modelo treinado) |
| Permutation (29 × 10 repetições) | 5 × 3 | 4,8 min | 0 |
| TreeSHAP (10 mil linhas) | 5 × 2 | 11,3 min | 0 |
| H de Friedman (até 300 pares) | 5 × 2 | 5,7 min | 0 |
| **PyGAwLLfs** (100 gerações) | 5 × 3 | **4,9 h** | **3.200–4.100 por execução (média por modelo)** |

Os quatro primeiros explicam um modelo já treinado (o da Etapa 6: 5 s na
LogReg, 15 s no RF e 3 s no XGBoost, no treino completo). O PyGAwLLfs
retreina o modelo milhares de vezes na subamostra de 50 mil voos — e, em
troca, entrega também a seleção de atributos (x*). Fonte: `custo_importancia.csv`;
detalhes do GA em `docs/piloto-custo.md`.

## 5. Interações: PyGAwLLfs × H de Friedman (Etapa 13)

- **Não concordam:** Spearman 0,13–0,21; top-10 com 1–2 pares em comum.
- **Causas:** baixa confiabilidade da matriz do PyGAwLLfs com este orçamento
  e conceitos diferentes (não aditividade do desempenho × da função do modelo).
- **Teste de domínio dos tops:** H corroborou 11 de 20 pares; PyGAwLLfs, 3 de 20.
- **Sensibilidade à agregação das sementes:** com os pares não sondados
  contando como zero, o ranking do PyGAwLLfs acompanhava a cobertura (Spearman
  0,39–0,64 com o nº de sementes sondadas; ≈ 0 na agregação atual). As
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

- Dos 23 pares do topo do PyGAwLLfs: 5 redundantes e 18 operacionais → **5
  corroborados descritivamente** (`hora` × origem, empresa × origem, `hora` ×
  empresa, congestionamento no destino × destino e período do dia × duração
  prevista); nenhum dos 10 controles. Com a média com zeros, eram 24 pares e 4
  corroborados (`docs/decisoes.md` §15).
- Redundância concentrada no topo: 7 das 15 arestas do top-5 dos três modelos.
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
