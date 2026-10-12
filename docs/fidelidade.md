# Fidelidade por retenção (Etapa 12) — resultados

Base para a seção de fidelidade do Cap. 4. Script: `src/compare/fidelity.py`;
tabelas: `results/tables/fidelidade.csv` (cada configuração × conjunto) e
`fidelidade_resumo.csv` (médias e fração do modelo completo); figura:
`results/figures/fidelidade`. Escolhas e como mudá-las: `docs/decisoes.md` §14.

## 1. Pergunta e método

Se um método diz que certas features são as mais importantes, um modelo
treinado **só com elas** deveria chegar perto do modelo com todas — e ficar
bem acima de um modelo com o mesmo número de features **sorteadas**.

- **Configurações por modelo:** top-5, top-10 e top-15 de cada método (pelo
  ranking consolidado da Etapa 11); o subconjunto x* que o PyGAwLLfs
  selecionou em cada uma das 5 execuções; 10 sorteios de k features (o
  acaso); e todas as 29 features (o modelo da Etapa 6).
- **Treino:** mesmo modelo e hiperparâmetros da Etapa 6, no **treino completo**
  (1,54 milhão de voos, §3.1). 138 treinos ao todo, ~50 min.
- **Avaliação:** validação e, por último, **teste** — um dos dois únicos
  lugares do projeto que tocam o teste (§4.9). Nenhuma lista de features foi
  escolhida olhando o teste.
- **Métricas:** F1 (a do fitness) e PR-AUC (sem limiar, mais estável); a
  "fração retida" é a métrica com top-k dividida pela métrica com todas.
- **Coerência verificada:** o modelo com todas as features reproduz
  exatamente o resultado da Etapa 6 (`tests/test_fidelidade.py`).

## 2. Resultado no teste — fração do PR-AUC do modelo completo retida

| Modelo | k | Intrínseca | Permutation | SHAP | PyGAwLLfs | Sorteio (média) |
|---|---:|---:|---:|---:|---:|---:|
| LogReg | 5 | 89% | 87% | — | **98%** | 81% |
| LogReg | 10 | 97% | 97% | — | 98% | 91% |
| RF | 5 | 94% | **97%** | **97%** | 94% | 72% |
| RF | 10 | 99% | 100% | 99% | 98% | 86% |
| XGBoost | 5 | 96% | **98%** | 96% | 94% | 72% |
| XGBoost | 10 | 96% | 99% | 100% | 96% | 85% |

Com k = 15, todos os métodos ficam entre 98% e 100%. Em F1 o quadro é o
mesmo (top-5: 95%–100% no RF e no XGBoost, contra 81%–83% dos sorteios).

**Subconjunto selecionado pelo PyGAwLLfs (x*), média das 5 execuções:**

| Modelo | Nº médio de features | Fração do F1 | Fração do PR-AUC |
|---|---:|---:|---:|
| LogReg | 9,0 | 103% | 99% |
| RF | 10,5 | 96% | 95% |
| XGBoost | 9,8 | 96% | 98% |

## 3. Leituras

1. **Todo método é muito melhor que o acaso**, e a diferença está no topo:
   com 5 features, os métodos retêm 87%–98% do PR-AUC, contra 72%–81% dos
   sorteios (e o pior sorteio cai a ~66% no RF). A partir de 10 features,
   tudo converge — a maior parte da informação está em poucas features
   (`origem_te`, `empresa_te`, `hora` e o calendário).
2. **A Permutation é a mais fiel no top-5 do RF e do XGBoost** pelo PR-AUC
   (97% e 98%); o SHAP empata com ela no RF (97%) e, no XGBoost, fica com a
   intrínseca (96%) — embora seja o melhor desse modelo pelo F1 (99%). Faz
   sentido que esses dois liderem: medem a contribuição de cada feature no
   próprio modelo que está sendo retreinado.
3. **O PyGAwLLfs é o mais fiel na LogReg** (top-5 com 98%, contra 87%–89% dos
   outros). O top-5 dos coeficientes é dominado pelo par colinear
   `dia_ano`/`mes` (limitação registrada na Etapa 6) e por `dia_mes` — e
   esse top-5, que retinha 96% do F1 na **validação**, cai para 82,5% no
   **teste**: os efeitos de calendário aprendidos em outras estações não se
   transferem para nov–dez. O PyGAwLLfs, que retreina o modelo sem cada
   feature, não cai nessa armadilha.
4. **O x* do PyGAwLLfs** entrega ~95%–99% do PR-AUC com ~1/3 das features.
   No RF e no XGBoost ele parecia melhor na validação (101%–102% do F1,
   contra 96% no teste): a queda é o otimismo previsto no §4.1 — o
   subconjunto foi escolhido olhando a validação.
5. **F1 acima de 100%** (ex.: 103% do x* na LogReg) é efeito do limiar fixo
   de 0,5 com `class_weight="balanced"`; no PR-AUC, que não depende de
   limiar, nenhum método passa de ~100%.

## 4. Na validação (robustez)

A ordem geral se mantém na validação (top-5 bem acima do sorteio em todos os
modelos). A diferença principal é a da LogReg descrita acima: a intrínseca vai
de 96% do F1 (validação) a 82,5% (teste). Tabela completa em
`fidelidade_resumo.csv` (`conjunto = val`).

## 5. Limitações

- O teste cai no pico sazonal (nov–dez; taxa de atraso de 24%): rankings que
  privilegiam calendário são penalizados por isso — o que é parte do
  resultado, mas também uma particularidade deste período de teste.
- Um só treino por configuração (os modelos são determinísticos; a variação
  entre execuções do PyGAwLLfs e entre sorteios aparece nas faixas da figura).
- F1 com limiar fixo de 0,5; o PR-AUC é a leitura sem limiar.
- Retenção não é remoção: o "Explicitamente NÃO incluídos" do §5 deixa a
  fidelidade por remoção (ROAR) de fora.
