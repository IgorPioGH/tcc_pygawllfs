# Dataset sintético com gabarito (Diferencial 1) — resultados

Base para a seção do Diferencial 1 no Cap. 4. Scripts: `src/synthetic/generate.py`
(gerador), `src/synthetic/aplicar_metodos.py` (os métodos, com o mesmo código
dos dados reais) e `src/synthetic/recovery.py` (comparação com o gabarito).
Tabelas: `results/tables/recuperacao_sintetico.csv`,
`recuperacao_sintetico_interacao.csv`, `recuperacao_sintetico_selecao.csv`.

## 1. O dataset

`z = 2·x1·x2 + x3 + x4 + 0,5·ruído`, `y = 1` se `z > 0`; 5.000 linhas
(treino 3.500 / validação 1.500, estratificado; sem tempo, então o split
aleatório é legítimo). Gabarito em três categorias:

| Features | Categoria | Papel |
|---|---|---|
| x1, x2 | diretas | **interação pura**: sozinhas não dizem nada sobre y; juntas, o sinal mais forte |
| x3, x4 | diretas | efeito aditivo |
| x5, x6 | redundantes | `0,9·x3 + ruído` e `0,9·x4 + ruído` (correlação 0,9): proxies, não ruído |
| x7–x10 | ruído | independentes de y |

Os parâmetros (β = 2, σ = 0,5, ρ = 0,9) foram calibrados com os modelos da
Etapa 6: a interação vale +0,09 de F1 no RF e +0,19 no XGBoost, e a LogReg,
linear, não consegue usá-la.

## 2. Importância: quem separa relevantes (diretas + redundantes) de ruído?

| Método | Modelo | AUC relevante × ruído | Top-6 relevantes | Top-4 = as diretas | Rank médio x1, x2 | Rank médio x5, x6 |
|---|---|---:|---:|---:|---:|---:|
| PyGAwLLfs | RF | **1,00** | 100% | 100% | **1,5** | 5,5 |
| PyGAwLLfs | XGBoost | **1,00** | 100% | 100% | **1,5** | 5,5 |
| Intrínseca | RF | 1,00 | 100% | 100% | 3,5 | 5,5 |
| Intrínseca | XGBoost | 1,00 | 100% | 100% | 3,5 | 5,5 |
| SHAP | RF | 1,00 | 100% | 50% | 5,5 | 3,5 |
| SHAP | XGBoost | 1,00 | 100% | 100% | 3,5 | 5,5 |
| Permutation | RF | 0,83 | 83% | 100% | 1,5 | 7,5 |
| Permutation | XGBoost | 0,91 | 87% | 100% | 1,5 | 6,6 |

Na LogReg, todos os métodos põem x1 e x2 lá embaixo (rank ~6), porque o
modelo linear não usa a interação pura — é limitação do modelo, não do
método (§4.11).

- **PyGAwLLfs e Permutation põem o par de interação à frente das aditivas**
  (no RF e no XGBoost); só o PyGAwLLfs faz isso **e** separa perfeitamente
  relevantes de ruído (AUC 1,0).
- **A Permutation é a mais fraca:** num modelo fixo, embaralhar uma
  redundante quase não muda nada (a original segura), e ela cai ao nível do
  ruído.
- **O SHAP do RF põe x1/x2 atrás das redundantes:** o RF captura mal a
  interação pura (padrão tipo XOR, sem divisão marginal útil), e o SHAP
  explica fielmente esse modelo.

## 3. Interação: quem acha o par verdadeiro, e o que fazem com a redundância?

| Método | Modelo | Posição de (x1, x2) entre 45 pares | Posição média de (x3, x5), (x4, x6) | Top-3 em {verdadeiro + redundância} |
|---|---|---:|---:|---:|
| PyGAwLLfs | RF | 1º (todas as sementes) | **2,5º** | 3 de 3 |
| PyGAwLLfs | XGBoost | 1º (todas as sementes) | **2,5º** | 3 de 3 |
| PyGAwLLfs | LogReg | 21º | 2,5º | 2 de 3 |
| H não normalizado | RF | 1º (todas as amostras) | 7º | 1 de 3 |
| H não normalizado | XGBoost | 1º (todas as amostras) | 20º | 1 de 3 |
| H normalizado | RF | 1º | 28,5º | 1 de 3 |
| H normalizado | XGBoost | 1º | 44,5º | 1 de 3 |

**A hipótese central se confirma:** os dois métodos acham a interação
verdadeira, mas só o PyGAwLLfs põe os pares direta-redundante no topo. Ele
mede a não aditividade do **desempenho** ao tirar features: tirar só x3 ou só
x5 quase não custa (a outra compensa), tirar as duas custa — e isso aparece
como "interação". O H mede a não aditividade da **função do modelo**, onde
esses pares são aditivos.

## 4. Seleção do PyGAwLLfs (x*) contra as 4 diretas

| Modelo | Precisão | Recall | F1 | Redundantes no x* | Ruído no x* |
|---|---:|---:|---:|---:|---:|
| RF | 1,0 | 1,0 | **1,0** — exatamente {x1, x2, x3, x4} | 0 | 0 |
| XGBoost | 0,8 | 1,0 | 0,89 (+ x6) | 1 | 0 |
| LogReg | 0,5 | 0,5 | 0,5 ({x3, x4, x5, x9}) | 1 | 1 |

A LogReg incluiu uma feature de ruído (x9): sobreajuste do subconjunto à
validação de 1.500 linhas — o mesmo otimismo do §4.1. Com N = 10 só existem
1.024 subconjuntos, e as 5 execuções convergem para o mesmo x*.
