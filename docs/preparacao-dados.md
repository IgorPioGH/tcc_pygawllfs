# Preparação dos dados: da base bruta ao dataset congelado

Registro dos números da preparação (Etapas 2, 4 e 5), para a metodologia
(Cap. 3). A aquisição dos CSVs está em `docs/aquisicao-vra.md`; cada atributo
está descrito em `docs/dicionario-features.md`. Todos os números abaixo são
reproduzíveis rodando o pipeline do `docs/metodologia.md` §11.

## 1. Descartes (`clean.py` e `target.py`)

| Etapa | Critério | Removidos | Restam |
|---|---|---:|---:|
| Base bruta (24 CSVs mensais, 2024–2025) | — | — | 1.992.832 |
| Voos cancelados | `Situação Voo` ≠ `REALIZADO` | 69.496 | 1.923.336 |
| Sem horário | `Partida Prevista` ou `Partida Real` vazia | 59.141 | 1.864.195 |
| Fora do recorte | `Partida Prevista` fora de [01/01/2024, 01/01/2026) — registros de 2026 com data mal preenchida | 11 | 1.864.184 |
| Atraso implausível | `atraso_min` fora de [−40, 720] min (6.222 abaixo de −40, 1.982 acima de 720) | 8.204 | **1.855.980** |

Taxa de cancelamento no período: 3,6%. A janela de plausibilidade é
assimétrica de propósito: a companhia não antecipa a partida de forma
significativa, mas atraso de horas é real. Os valores fora dela são erros de
registro de data (aparecem como −4 dias ou +31 dias).

## 2. Alvo (`target.py`)

- `atraso_min = Partida Real − Partida Prevista`, em minutos.
- `atrasado = 1` se `atraso_min ≥ 15` → **18,5%** dos voos da base limpa.
- Distribuição: `results/figures/histograma_atraso.{png,pdf}` (`src/data/figuras.py`).
  A massa fica perto de zero (mediana −1 min; 53,5% dos voos partem antes do
  horário previsto), com cauda longa à direita; 0,7% dos voos têm atraso
  entre 180 e 720 min.
- Colunas pós-hoc removidas da base depois do cálculo do alvo: `Partida Real`,
  `Chegada Real`, `Situação Voo`, `Código Justificativa` (§4.3). A
  demonstração com os dados de por que cada uma vaza está no notebook
  `01_qualidade_dados.ipynb`, seção "Vazamento".

## 3. Split temporal (`features.py::dividir_temporal`)

Cortes em datas absolutas, intervalos semiabertos `[início, fim)` (§4.4):

| Conjunto | Período | Voos | Taxa de atraso |
|---|---|---:|---:|
| Treino | 01/01/2024 – 31/08/2025 | 1.539.522 | 18,11% |
| Validação | 01/09/2025 – 31/10/2025 | 158.389 | 16,17% |
| Teste | 01/11/2025 – 31/12/2025 | 158.069 | 24,26% |

- Invariante verificado no código (e de novo em 11 out): `max(treino) <
  min(validação)` e `max(validação) < min(teste)`; os índices dos três
  conjuntos são disjuntos.
- **O teste cai no pico sazonal** (nov–dez: em 2024, 25,5% e 26,8% de
  atraso). Isso é documentado, não corrigido: o objetivo do split é simular a
  previsão do futuro a partir do passado, e o futuro tem sazonalidade. Efeito
  observado na Etapa 6: F1 de teste maior que o de validação (LogReg 0,42 ×
  0,34; RF 0,44 × 0,35; XGBoost 0,45 × 0,36), porque a classe positiva é mais
  frequente no teste. Comparações de desempenho são sempre dentro do mesmo
  conjunto, nunca validação × teste.

## 4. Subamostra do PyGAwLLfs (`features.py`, bloco final)

| | Linhas | Proporção | Taxa de atraso |
|---|---:|---:|---:|
| Treino | 45.336 | 90,67% | 18,11% |
| Validação | 4.664 | 9,33% | 16,17% |
| **Total** | **50.000** | | |

- Sorteio estratificado por `origem × atrasado` (`train_test_split`,
  `random_state=42`), só sobre treino + validação — o teste nunca entra.
- **Proporção:** a estratificação por `origem` preserva a proporção original
  entre treino e validação (1.539.522 : 158.389 = 90,67% : 9,33%), e a
  estratificação por `atrasado` preserva a taxa de atraso de cada período.
- **Justificativa do tamanho (docs/metodologia.md §3.1):** o artigo original rodou o
  dataset Boson com 52.035 exemplos e levou 7,2 h por execução. Com a base
  completa (~1,5 milhão de linhas de treino), cada execução do GA levaria dias
  e o protocolo de 15 execuções seria inviável. Com 50 mil linhas, o protocolo
  inteiro rodou em 5,4 h (`docs/piloto-custo.md`).
- **Escopo:** a subamostra é usada **só** dentro do PyGAwLLfs. Modelos finais,
  métricas, Permutation, SHAP e fidelidade usam os conjuntos completos.

## 5. Dataset sintético (Diferencial 1)

Gerado por `src/synthetic/generate.py` (5.000 linhas, semente 42, split
aleatório 70/30 estratificado — dado sem tempo); a especificação e o gabarito
estão na docstring do script e em `docs/sintetico.md`.
