# Registro de decisões (e como revertê-las)

Decisões tomadas para **finalizar o projeto** com a opção mais segura, mas que
podem mudar depois da conversa com o orientador. Para cada uma: o estado
atual, por que foi a escolha, a alternativa, **como trocar** e quanto custa.

Estado em 11 out 2026. A ordem de reexecução do pipeline está no `docs/metodologia.md` §11.

## Resumo

| # | Decisão | Estado atual | Perguntar ao orientador? | Custo para trocar |
|---|---|---|---|---|
| 1 | Bugs de mutação e elitismo do PyGAwLLfs | **corrigidos** no wrapper | **sim** | ~6 h de máquina |
| 2 | Congestionamento conta só voos realizados | **mantido** (limitação documentada) | sim | ~6,5 h |
| 3 | `duracao_prevista_min` sem janela de plausibilidade | **mantido** (limitação documentada) | não | ~6,5 h |
| 4 | RF fixo em `max_depth=10` | **mantido** (igual ao RF do artigo) | não | ~6 h |
| 5 | Métrica da Permutation: F1 | **mantido** | não | ~5 min |
| 6 | Sem SHAP para a LogReg | **mantido** (§4.6) | não | ~1 min |
| 7 | Teste no pico sazonal (nov–dez) | **mantido** (documentado) | não | — (não trocar) |
| 8 | Agregação das sementes, RBO e top-k na Etapa 11 | média das sementes; RBO estendido, p = 0,9 | não | segundos |
| 9 | Parâmetros do sintético (β = 2, σ = 0,5, ρ = 0,9) | **mantido** | não | ~30 min |
| 10 | Significado das letras do `Código Tipo Linha` | **a conferir** (muda a leitura de um par corroborado na Etapa 15) | não | — (só texto) |
| 11 | Critérios da validação de domínio (Etapa 15) | top-10 por modelo; 1 p.p.; replicação ≥ 0,5; escala logit | talvez | ~1 min |
| 12 | Cálculo do H de Friedman (Etapa 13) | logit; H_abs como principal; RF e XGBoost; 500 linhas × 5 sementes | não | ~7 min |
| 13 | Orçamento do PyGAwLLfs para a matriz de interação | **mantido o do artigo** (matriz pouco confiável com N = 29, documentado) | **sim** | ~5,4 h por +5 sementes, ganho pequeno |

## 1. Bugs de mutação e elitismo — corrigidos

- **Estado:** `MutacaoInPlace` e `GAwLLCorrigido._apply_elitism` em
  `src/importance/gawll.py` (correções 4 e 5, `docs/metodologia.md` §4.12). Todos os
  resultados do PyGAwLLfs (Etapas 8, 10 e 11) usam a versão corrigida.
- **Por quê:** o código original não executa o que o artigo descreve, e os
  testes 1 e 2 de `tests/test_gawll.py` provam isso. Rodar com os bugs
  significaria comparar métodos usando um GA sem mutação.
- **Alternativa:** rodar também a versão original, para mostrar o efeito dos
  bugs ou para manter a comparabilidade com os números do artigo.
- **Como trocar** (sem perder a versão corrigida):
  1. em `GAwLLCorrigido.run`, trocar `MutacaoInPlace()` por `Mutation()`, e
     apagar o método `_apply_elitism` da subclasse (passa a valer o original);
  2. rodar `python src/importance/gawll.py --saida experiments/runs/gawll_original`
     (pasta separada: a retomada não mistura as duas versões) — **5,4 h**;
  3. para comparar, apontar `consolidar_gawll.DIR_EXECUCOES` para a pasta nova.
- **Perguntas ao Prof. Tinós:** (a) confirma que são bugs? (b) prefere que o
  TCC reporte só a versão corrigida, ou as duas?

## 2. Congestionamento só de voos realizados — limitação documentada

- **Estado:** `congestion_origem/destino` contam os voos depois da remoção dos
  cancelados. Dataset congelado mantido.
- **Por quê:** o impacto medido é desprezível — F1 de validação −0,001
  (XGBoost) e +0,002 (RF) com a contagem correta
  (`src/data/auditar_congestionamento.py` → `results/tables/auditoria_congestionamento.csv`).
  Refazer tudo custaria ~6,5 h para mudar a terceira casa decimal.
- **Alternativa:** contar sobre a malha programada (com os cancelados), como pede o §4.3.
- **Como trocar:**
  1. em `src/data/features.py`, calcular as duas contagens sobre o
     `vra_bruto.parquet` dentro do recorte (como em
     `auditar_congestionamento.py::contagens_programadas`) e juntar pelo índice;
  2. rodar de novo, nesta ordem: `features.py` → `baselines.py` → `gawll.py`
     (5,4 h) → os 4 scripts da Etapa 9 → `compare/` (Etapa 11) → os testes.
     O sintético (Etapa 10) não muda.
- **Pergunta ao Prof. Tinós:** aceita como limitação documentada?

## 3. `duracao_prevista_min` sem janela de plausibilidade

- **Estado:** valores 0–1.435 min mantidos; 1.638 voos (0,1%) acima de 15 h.
  Limitação 1 em `docs/dicionario-features.md`.
- **Alternativa:** aplicar uma janela (ex.: 15–960 min) com descarte das
  linhas, como no `atraso_min`, ou um corte (clip).
- **Como trocar:** em `features.py::features_sem_alvo`, filtrar ou cortar a
  coluna; mesma cascata de reexecução do item 2.

## 4. RF fixo em `max_depth=10`

- **Estado:** `GRADE["random_forest"] = [{"max_depth": 10}]` em `src/models/baselines.py`.
- **Por quê:** é a configuração do RF dentro do PyGAwLLfs original; com ela,
  o GA e os demais métodos explicam o mesmo modelo (§4.11). No tuning,
  `max_depth=12` ganhava só 0,006 de F1.
- **Como trocar:** pôr `{"max_depth": 12}` na `GRADE` → `baselines.py` →
  `gawll.py --modelos random_forest` (~2,2 h) → Etapa 9 → Etapa 11.

## 5. Métrica da Permutation: F1

- **Estado:** `METRICA = "f1"` em `src/importance/permutation.py`.
- **Por quê:** é a mesma métrica do fitness do PyGAwLLfs, então os dois medem
  "quanto o F1 piora sem essa informação".
- **Alternativa:** `"average_precision"` (PR-AUC, sem limiar e mais estável).
- **Como trocar:** mudar `METRICA` → `permutation.py` (~5 min) → Etapa 11.

## 6. Sem SHAP para a LogReg

- **Estado:** a LogReg tem intrínseca (coeficiente padronizado), Permutation e PyGAwLLfs.
- **Por quê:** `docs/metodologia.md` §4.6. Num modelo linear com features padronizadas,
  `mean(|SHAP|)` é praticamente proporcional a `|coeficiente|`, então seria
  quase uma cópia da intrínseca.
- **Como trocar:** em `shap_global.py`, usar `shap.LinearExplainer` para a
  LogReg e incluí-la em `MODELOS_ARVORE` (renomear) → ~1 min → Etapa 11.

## 7. Teste no pico sazonal

- **Estado:** teste = nov–dez/2025, taxa de atraso de 24,3% (treino 18,1%, validação 16,2%).
- **Por quê:** o split simula prever o futuro; mudar os cortes para fugir da
  sazonalidade seria ajustar o protocolo ao resultado. Documentado em
  `docs/preparacao-dados.md` §3.
- **Não trocar.**

## 8. Escolhas da Etapa 11 (comparação de rankings)

- **Agregação das sementes:** cada (método, modelo) vira um ranking só, pela
  **média da importância nas 5 sementes** (o pipeline original do artigo faz
  a média do vimp entre execuções). A variação entre sementes é medida à
  parte, no `stability.py`.
- **Ordenação:** pela importância; no coeficiente da LogReg, pelo valor
  absoluto. Empates são desfeitos pelo nome da feature (resultado determinístico).
- **RBO:** versão estendida (RBO_ext, Webber et al., 2010), `p = 0,9`. Para
  listas completas e iguais, vale 1.
- **Top-k:** fração de sobreposição `|A_k ∩ B_k| / k`, para k = 5, 10, 15.
- **Como trocar:** constantes no topo de `src/compare/rankings.py`; rodar de
  novo leva segundos.

## 9. Parâmetros do sintético

- **Estado:** `BETA_INTERACAO = 2`, `SIGMA_RUIDO = 0,5`, `RHO_REDUNDANCIA = 0,9`
  em `src/synthetic/generate.py`, calibrados para a interação valer F1 de
  verdade no RF e no XGBoost.
- **Alternativa (opcional, §5 de docs/metodologia.md):** variar ruído e correlação e
  traçar a curva de degradação da recuperação.
- **Como trocar:** mudar as constantes → `generate.py` →
  `aplicar_metodos.py` (~30 min; apagar antes `experiments/runs/sintetico/`) → `recovery.py`.

## 10. `Código Tipo Linha`

- Conferir no dicionário de dados do VRA (ANAC) o significado de C, G, I e N e
  preencher em `docs/dicionario-features.md`. Não muda nenhum resultado.

## 11. Critérios da validação de domínio (Etapa 15)

Constantes no topo de `src/domain/validate_pairs.py`; rodar de novo leva ~1 min.
Resultados em `docs/validacao-dominio.md`.

- **Pares:** união do top-10 de cada modelo (`TOP_PARES_POR_MODELO`) — 25 pares;
  10 controles do fundo do ranking (`N_CONTROLES`), sem repetir par de domínio.
- **Redundância:** coeficiente de incerteza ≥ 0,8 (`LIMIAR_REDUNDANCIA`). Os
  35 pares têm um vão entre 0,40 e 0,85, então qualquer valor nesse intervalo
  dá a mesma classificação.
- **Relevância prática:** desvio médio ≥ 1 ponto percentual (`LIMIAR_PP`),
  nos dois anos. Com 1,7 milhão de voos, quase tudo passa do limiar de ruído;
  sem esse piso, controles seriam "corroborados".
- **Replicação:** correlação ≥ 0,5 entre os desvios por célula de 2024 e de
  2025 (`LIMIAR_REPLICACAO`; convenção de correlação "forte"). Foi incluída
  depois de a primeira versão (sem ela) corroborar `dia_ano` × empresa, cujo
  padrão não se repete entre anos (r = 0,06) — choque pontual, não interação.
- **Semana do ano no lugar de `dia_ano`:** na análise descritiva, cada célula
  junta 7 dias; com o dia exato, cada célula seria uma data e o teste mediria
  eventos de um dia. A redundância, porém, é medida com o `dia_ano` diário
  (como o modelo o vê).
- **Escala logit** para "sem interação" (padrão para resposta binária).
  Alternativa: modelo aditivo nas probabilidades.
- **Cobertura:** top-15 aeroportos e top-10 empresas (`TOP_CATEGORIAS`).
- **Sensibilidade (verificada):** com `LIMIAR_PP = 0,75` o resultado é o mesmo
  (2 corroborados); com 0,5, entra também `tipo_linha` × origem (0,75 e
  1,18 p.p.; r = 0,60) — e, mesmo assim, nenhum controle passa.

## 12. Cálculo do H de Friedman (Etapa 13)

Constantes no topo de `src/interaction/friedman_h.py`; rodar de novo leva ~7 min.
Resultados em `docs/interacoes-h.md`.

- **Escala logit** (log-odds), a mesma da Etapa 15. Alternativa: probabilidade
  (`predict_proba` direto) — os valores mudam, e a LogReg passaria a ter H > 0
  só por causa da curva sigmoide.
- **H_abs (não normalizado) como principal**, H² normalizado salvo junto. O
  normalizado se mostrou menos estável (Spearman entre sementes 0,87 × 0,93 no
  XGBoost) e passou de 1 num par de dummies da mesma variável (denominador
  quase zero). Para trocar: usar `friedman_h` no lugar de `friedman_h_abs` em
  `compare/interacoes.py`.
- **LogReg fora:** aditiva no logit, H = 0 por construção (testado).
- **500 linhas × 5 sementes** (`AMOSTRA`, `comum.SEMENTES`), da validação.
- **Conjunto de pares** do §4.7, sem corte (o custo ficou muito abaixo do limite).

## 13. Orçamento do PyGAwLLfs para a matriz de interação

- **Estado:** parâmetros do artigo (população 50, 100 gerações, 3 sondagens de
  linkage learning por geração). Com N = 29, cada par é sondado em média 0,7
  vez por execução, e o ranking de pares quase não se repete entre execuções
  (split-half ≈ 0–0,11). Documentado como resultado e limitação
  (`docs/interacoes-h.md` §3).
- **Alternativa A — mais sementes:** +5 sementes custam ~5,4 h e levariam a
  confiabilidade da média de ~0,2 a ~0,3 (Spearman-Brown). Pouco ganho.
- **Alternativa B — mais sondagens por geração:** mudaria o algoritmo do artigo
  (a fração de linkage learning na geração é fixa no `_generation_ll`); seria
  uma variante, não o método original.
- **Recomendação:** manter e discutir; levar a pergunta ao orientador.

## Perguntas para a reunião com o Prof. Tinós

1. Os dois comportamentos do código (mutação e elitismo) são bugs? O TCC deve
   reportar só a versão corrigida ou também a original? (item 1)
2. O congestionamento só de realizados pode ficar como limitação, dado o
   impacto medido de ±0,002 no F1? (item 2)
3. Sobre a leitura de redundância como interação no PyGAwLLfs (resultado do
   sintético, `docs/sintetico.md` §3): ele vê isso como propriedade
   esperada do método? Nos dados reais, 8 das 15 arestas do top-5 dos três
   modelos são pares redundantes (`docs/validacao-dominio.md` §2).
4. Os critérios da validação de domínio (1 p.p. e replicação entre anos) são
   adequados, ou ele prefere um critério mais brando? (item 11)
5. Com N = 29, o linkage learning sonda cada par ~0,7 vez por execução, e o
   ranking de interações não se repete entre execuções. Isso é esperado com o
   orçamento do artigo? Vale testar mais gerações ou mais sondagens, ou fica
   como limitação discutida? (item 13)
