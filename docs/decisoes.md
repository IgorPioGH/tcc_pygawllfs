# Registro de decisões (e como revertê-las)

Decisões tomadas para **finalizar o projeto** com a opção mais segura, mas que
podem mudar depois da conversa com o orientador. Para cada uma: o estado
atual, por que foi a escolha, a alternativa, **como trocar** e quanto custa.

Estado em 17 out 2026. A ordem de reexecução do pipeline está no `docs/metodologia.md` §11.

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
| 10 | Significado das letras do `Código Tipo Linha` | **confirmado** (C/G: cargueira doméstica/internacional; N/I: mista doméstica/internacional) | não | — (só texto) |
| 11 | Critérios da validação de domínio (Etapa 15) | top-10 por modelo; 1 p.p.; replicação ≥ 0,5; escala logit | talvez | ~1 min |
| 12 | Cálculo do H de Friedman (Etapa 13) | logit; H_abs como principal; RF e XGBoost; 500 linhas × 5 sementes | não | ~7 min |
| 13 | Orçamento do PyGAwLLfs para a matriz de interação | **mantido o do artigo** (matriz pouco confiável com N = 29, documentado) | **sim** | ~5,4 h por +5 sementes; ganho moderado na LogReg e no XGBoost (item 15) |
| 14 | Desenho da fidelidade (Etapa 12) | k = 5/10/15; 10 sorteios; treino completo; F1 e PR-AUC | não | ~50 min |
| 15 | Agregação das sementes na matriz de interação | **só as sementes em que o par foi sondado**; ranking com pares sondados em ≥ 2 sementes; a média com zeros fica como sensibilidade | **sim** (contador) | ~3 min (ou ~5,4 h com contador) |
| 16 | Incerteza dos resultados no teste (Etapa 12b) | bootstrap **por dias** (61 blocos), **pareado**, B = 1.000, IC percentil de 95%, sem correção para comparações múltiplas | talvez (correção) | ~5 min |

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

- **Confirmado** pela ANAC (CRUZ, 2013, slide 17): C = doméstica
  cargueira, G = internacional cargueira, I = internacional mista, N =
  doméstica mista. Registrado em `docs/dicionario-features.md`; a leitura do par
  `hora` × tipo de linha está em `docs/validacao-dominio.md` §3. Não muda
  nenhum resultado.

## 11. Critérios da validação de domínio (Etapa 15)

Constantes no topo de `src/domain/validate_pairs.py`; rodar de novo leva ~1 min.
Resultados em `docs/validacao-dominio.md`.

- **Pares:** união do top-10 de cada modelo (`TOP_PARES_POR_MODELO`) — 26 pares;
  10 controles do fundo do ranking (`N_CONTROLES`), entre os pares com rank nos
  três modelos e sem repetir par de domínio. O ranking segue a agregação da
  decisão 15.
- **Redundância:** coeficiente de incerteza ≥ 0,8 (`LIMIAR_REDUNDANCIA`). Os
  36 pares têm um vão entre 0,40 e 0,99, então qualquer valor nesse intervalo
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
  (4 corroborados); com 0,5, entra `tipo_linha` × origem (0,75 e 1,18 p.p.;
  r = 0,60), mas também um controle, `hora` × porte da origem (0,62 e 0,58 p.p.;
  r = 0,74). O piso de 1 p.p. é o que separa candidatos de controles.

## 12. Cálculo do H de Friedman (Etapa 13)

Constantes no topo de `src/interaction/friedman_h.py`; rodar de novo leva ~7 min.
Resultados em `docs/interacoes-h.md`.

- **Escala logit** (log-odds), a mesma da Etapa 15. Alternativa: probabilidade
  (`predict_proba` direto) — os valores mudam, e a LogReg passaria a ter H > 0
  só por causa da curva sigmoide.
- **H_abs (não normalizado) como principal**, H² normalizado salvo junto. O
  normalizado se mostrou menos estável (Spearman entre sementes 0,82 × 0,90 no
  XGBoost) e passou de 1 num par de dummies da mesma variável (denominador
  quase zero). Para trocar: usar `friedman_h` no lugar de `friedman_h_abs` em
  `compare/interacoes.py`.
- **LogReg fora:** aditiva no logit, H = 0 por construção (testado).
- **500 linhas × 5 sementes** (`AMOSTRA`, `comum.SEMENTES`), da validação.
- **Conjunto de pares** do §4.7, sem corte (o custo ficou muito abaixo do limite),
  com o grafo do PyGAwLLfs nas duas agregações da decisão 15: 25 features (300
  pares) no RF e 27 (351) no XGBoost.

## 13. Orçamento do PyGAwLLfs para a matriz de interação

- **Estado:** parâmetros do artigo (população 50, 100 gerações, 3 sondagens de
  linkage learning por geração). Com N = 29, cada par é sondado em média 0,7
  vez por execução, e o ranking de pares quase não se repete entre execuções
  (split-half 0,06–0,38 só com as sementes sondadas; ≈ 0–0,11 com a média com
  zeros, decisão 15). Documentado como resultado e limitação
  (`docs/interacoes-h.md` §3).
- **Alternativa A — mais sementes:** +5 sementes custam ~5,4 h. Com a média com
  zeros, levariam a confiabilidade da média de ~0,2 a ~0,3 (Spearman-Brown).
  Com a agregação da decisão 15, a das 5 sementes já é 0,55 (LogReg), 0,39
  (XGBoost) e 0,12 (RF), e com 10 iria a ~0,71, ~0,56 e ~0,21: ganho moderado
  na LogReg e no XGBoost, quase nenhum no RF.
- **Alternativa B — mais sondagens por geração:** mudaria o algoritmo do artigo
  (a fração de linkage learning na geração é fixa no `_generation_ll`); seria
  uma variante, não o método original.
- **Recomendação:** manter e discutir; levar a pergunta ao orientador.

## 14. Desenho da fidelidade (Etapa 12)

Constantes no topo de `src/compare/fidelity.py`; rodar de novo leva ~50 min.
Resultados em `docs/fidelidade.md`.

- **k = 5, 10, 15** (`TOP_K`), como na proposta; mais o x* de cada execução do
  PyGAwLLfs e todas as 29 features.
- **Referência de acaso:** 10 sorteios de k features por modelo
  (`N_SORTEIOS`, semente 42). Não estava no plano original; foi incluída
  porque, sem ela, "retém 95%" não diz se isso é muito ou pouco.
- **Treino completo** (1,54 milhão de linhas), como pede o §3.1, com os
  hiperparâmetros da Etapa 6 sem retuning.
- **F1 (limiar 0,5) e PR-AUC**; o PR-AUC é a leitura principal da retenção
  porque não depende de limiar (o F1 passou de 100% em alguns casos).
- **Validação e teste** avaliados; o teste é o resultado principal.
- **Rankings usados:** os consolidados da Etapa 11 (média das sementes).

## 15. Agregação das sementes na matriz de interação do PyGAwLLfs

- **Problema:** o `vint` de uma execução começa em 0 e só registra um par quando
  uma sondagem do linkage learning o sorteia (300 sorteios por execução, 406
  pares; ~48% dos pares ficam sem sondagem em cada execução). Na média das 5
  sementes, um par não sorteado entrava como "interação 0". Com isso, a posição
  do par no ranking acompanhava **quantas vezes ele foi sondado**: Spearman entre
  a média do par e o nº de sementes em que ele foi sondado de 0,43 (LogReg),
  0,68 (RF) e 0,58 (XGBoost), nos 406 pares.
- **Estado:** `graph.interacao_media(agregacao="sondadas")`, o padrão: média só
  das sementes em que o par foi sondado; par nunca sondado fica sem média; o
  ranking (top-k, grafo reduzido, candidatos e controles da Etapa 15) só inclui
  pares sondados em pelo menos 2 sementes (`MIN_SEMENTES_RANKING`), porque uma
  sondagem única guarda o máximo de um retreino só. Na comparação com o H,
  Spearman e Kendall usam todos os pares com média; RBO e top-k, só os ranqueados.
- **Como se sabe se o par foi sondado, sem contador:** os três modelos de uma
  mesma semente sondam os mesmos pares. O sorteio do par usa o `random` do
  Python, e o nº de chamadas a ele (torneio e sorteio do par) não depende do
  fitness. Então "sondado na semente s" = vint > 0 em algum dos três modelos
  nessa semente (`graph.marcar_sondagens`). Conferido em
  `sensibilidade_cobertura.csv`: 210 pares sondados por execução, contra
  212 ± 6 esperados para 300 sorteios uniformes; o RF nunca deu efeito nulo
  exato num par que outro modelo registrou.
- **Zeros que são medida:** com o par sondado, o vint pode dar 0 exato quando o
  modelo não usa nenhuma das duas features (retreinar com ou sem ela dá o mesmo
  modelo). Isso acontece 0 vez por execução no RF, 3,8 na LogReg e 14,8 no
  XGBoost (ex.: os pares com `periodo_dia`, que o XGBoost não usa: SHAP = 0).
  Esses zeros **entram** na média. A regra literal "média dos valores > 0" os
  trataria como par não sondado.
- **Evidência do sintético:** com N = 10 (45 pares), cada par é sorteado ~6,7
  vezes por execução, e nenhum dos 675 valores do vint (15 execuções × 45
  pares) é zero. As duas agregações coincidem, e o Diferencial 1 não muda. É o
  regime em que o PyGAwLLfs foi desenhado: com cobertura total, a média com
  zeros não tem o problema.
- **Análise de sensibilidade** (`src/compare/sensibilidade_agregacao.py` →
  `results/tables/sensibilidade_*.csv`, figura `sensibilidade_agregacao`):

| Medida | Com zeros (original) | Só sondadas (principal) |
|---|---:|---:|
| Spearman(média do par, nº de sementes sondadas): LogReg / RF / XGBoost | 0,43 / 0,68 / 0,58 | −0,01 / 0,02 / −0,02 |
| PyGAwLLfs × H_abs, Spearman: RF / XGBoost | 0,09 / 0,08 | 0,16 / 0,15 |
| PyGAwLLfs × H_abs, top-10 em comum: RF / XGBoost | 1 / 0 | 2 / 0 |
| Estabilidade entre sementes (Spearman): RF / XGBoost | −0,01 / −0,00 | 0,04 / 0,06 |
| Split-half (2 × 3 sementes): LogReg / RF / XGBoost | 0,11 / −0,00 / 0,09 | 0,38 / 0,06 / 0,24 |
| Confiabilidade da média de 5 (Spearman-Brown): LogReg / RF / XGBoost | 0,20 / −0,01 / 0,17 | 0,55 / 0,12 / 0,39 |
| Pares redundantes: rank médio em percentil, RF / XGBoost (H_abs: 39 / 49) | 27 / 35 | 33 / 40 |
| Pares redundantes no top-5 dos três modelos | 8 de 15 | 6 de 15 |
| Etapa 15: candidatos / corroborados | 25 / 2 | 26 / 4 |
| Etapa 15: controles corroborados / controles nunca sondados | 0 / 5 | 0 / 0 |
| Top-10 do PyGAwLLfs corroborados no domínio (pares do H, RF + XGBoost) | 1 de 20 | 2 de 20 |

  Os números "com zeros" usam o conjunto do H atual (300/351 pares); no
  conjunto anterior (276/300), o Spearman PyGAwLLfs × H_abs do RF era 0,12.
- **O que não muda:** (1) PyGAwLLfs e H não concordam (Spearman ≤ 0,16; top-10
  com 0–2 pares em comum); (2) no RF, o ranking de pares não se repete entre
  sementes; (3) o PyGAwLLfs põe os pares redundantes mais alto que o H; (4)
  nenhum controle é corroborado, e o top-10 do H tem mais pares corroborados
  (9 de 20) que o do PyGAwLLfs.
- **O que muda:** (1) o artefato de cobertura some (0,43–0,68 → ≈ 0); (2) a
  confiabilidade sobe na LogReg e no XGBoost: a instabilidade era em parte
  cobertura e em parte ruído; no RF, segue perto de 0; (3) o top-10 de cada
  modelo muda (mantêm-se 7 dos 10 pares na LogReg, 5 no RF e 3 no XGBoost;
  `sensibilidade_top10.csv`), e a Etapa 15 passa de 2 para 4
  pares corroborados (entram `hora` × congestionamento na origem e empresa ×
  origem); (4) os controles antigos eram, na maioria, pares que o PyGAwLLfs
  nunca mediu (5 dos 10 nunca sondados; 0,6 semente em média); os novos são
  pares medidos e fracos (2,8 sementes); (5) a redundância no topo diminui (8 →
  6 das 15 arestas do top-5), mas continua acima da do H.
- **Ressalvas:** (a) uma média de 2 ou 3 sementes varia mais que uma de 5, e o
  top-10 novo tem em média 2,4–2,8 sementes, contra 2,9 entre os pares
  ranqueados: o viés de cobertura deu lugar a um viés de variância, bem menor;
  (b) o vint guarda o **máximo** das sondagens de cada execução, e quantas vezes
  o par foi sondado dentro da execução (k) não é conhecido: um par sondado 3
  vezes tende a ter máximo maior; (c) a reconstrução depende de o sorteio ser o
  mesmo nos três modelos, o que vale nestas execuções (conferido), mas não é uma
  garantia do PyGAwLLfs.
- **Efeito no H:** o conjunto do §4.7 passou a incluir o grafo das duas
  agregações (+`porte_destino_hub` no RF; +`fim_semana` e
  +`porte_origem_pequeno` no XGBoost): 300 pares no RF (eram 276) e 351 no
  XGBoost (eram 300). O H dos pares antigos é idêntico (diferença < 10⁻¹³). Os
  pares novos do XGBoost têm muitos H = 0 (o modelo não usa `fim_semana`), e a
  estabilidade do H_abs nele foi de 0,93 para 0,90.
- **Como trocar:** `agregacao="com_zeros"` em `validate_pairs.interacao_media` e
  em `interacoes.comparar_modelo`; `MIN_SEMENTES_RANKING` em
  `src/interaction/graph.py`. Rodar `validate_pairs.py`, `interacoes.py` e
  `sensibilidade_agregacao.py`: ~3 min.
- **Alternativa — contador de sondagens:** registrar, no wrapper
  (`GAwLLCorrigido`, sem mexer no PyGAwLLfs), quantas vezes cada par foi
  sondado em cada execução. Daria k, que a reconstrução não dá (ressalva b), e
  dispensaria a ressalva c. Custa uma nova rodada do PyGAwLLfs (~5,4 h). Se o GA
  for determinístico por semente (a conferir com uma execução), todos os outros
  resultados ficariam iguais. Pergunta 6 da reunião.

## 16. Incerteza dos resultados no teste (Etapa 12b)

Constantes no topo de `src/compare/incerteza.py`; rodar de novo leva ~5 min.
Resultados em `docs/incerteza.md`.

- **Estado:** bootstrap por dias (os 61 dias do teste sorteados com
  reposição), B = 1.000 (`N_REPETICOES`), semente 42, IC pelos percentis 2,5 e
  97,5. Pareado: o mesmo sorteio para todos os modelos e configurações, e as
  comparações usam o IC da diferença. Métricas: F1, PR-AUC e ROC-AUC dos
  modelos; PR-AUC (retenção) na fidelidade.
- **Por quê:** voos do mesmo dia não são independentes (a taxa de atraso
  diária no teste vai de 9% a 62%); sortear voos daria IC estreito demais. O
  pareamento cancela a variação comum aos dois lados (um dia ruim é ruim para
  todos os modelos), que a comparação de ICs sobrepostos não cancela.
- **Pré-requisito:** a Etapa 12 passou a salvar o y_score de cada
  configuração no teste (`experiments/runs/fidelidade/previsoes_teste.npz`).
  Rodá-la de novo para isso reproduziu `fidelidade.csv` e
  `fidelidade_resumo.csv` exatamente (só o tempo de treino muda).
- **Isolamento do teste:** não é um novo uso (§4.9 da metodologia). Usa as
  mesmas previsões das Etapas 6 e 12, de configurações já fixadas, e nada é
  escolhido a partir do bootstrap.
- **Alternativas:** (a) bootstrap por voo: mais estreito, mas ignora a
  dependência dentro do dia; (b) blocos de vários dias (ex.: semanas): captura
  a dependência entre dias vizinhos, mas sobram só ~9 blocos; (c) correção de
  Holm ou Bonferroni para as 87 comparações: mais conservadora; as
  diferenças no limite (ex.: Permutation × SHAP no top-5 do RF, IC de +0,1 a
  +1,7 ponto) deixariam de contar como reais.
- **Como trocar:** `N_REPETICOES`, `SEMENTE` e `PERCENTIS` em
  `src/compare/incerteza.py`; o bloco é o dia em `dias_do_teste`. Rodar
  `python src/compare/incerteza.py` (~5 min).

## Perguntas para a reunião com o Prof. Tinós

1. Os dois comportamentos do código (mutação e elitismo) são bugs? O TCC deve
   reportar só a versão corrigida ou também a original? (item 1)
2. O congestionamento só de realizados pode ficar como limitação, dado o
   impacto medido de ±0,002 no F1? (item 2)
3. Sobre a leitura de redundância como interação no PyGAwLLfs (resultado do
   sintético, `docs/sintetico.md` §3): ele vê isso como propriedade
   esperada do método? Nos dados reais, 6 das 15 arestas do top-5 dos três
   modelos são pares redundantes (`docs/validacao-dominio.md` §2).
4. Os critérios da validação de domínio (1 p.p. e replicação entre anos) são
   adequados, ou ele prefere um critério mais brando? (item 11)
5. Com N = 29, o linkage learning sonda cada par ~0,7 vez por execução, e o
   ranking de interações não se repete entre execuções. Isso é esperado com o
   orçamento do artigo? Vale testar mais gerações ou mais sondagens, ou fica
   como limitação discutida? (item 13)
6. Na média das sementes, o zero do vint era tratado como "sem interação",
   quando quase sempre é "par não sondado"; agora a média usa só as sementes
   sondadas, reconstruídas pelo sorteio comum aos três modelos (item 15). Ele
   aceita a reconstrução, ou prefere uma nova rodada (~5,4 h) com um contador de
   sondagens por par? A pergunta conversa com a 5: se houver nova rodada, dá
   para somar o contador e mais sementes (com a agregação nova, 10 sementes
   levariam a confiabilidade da LogReg e do XGBoost a ~0,7 e ~0,56).
