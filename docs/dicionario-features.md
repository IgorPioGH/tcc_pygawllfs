# Dicionário das features utilizadas para a previsão de atraso do voo

Os **29 atributos** do dataset congelado (`data/processed/{train,val,test}.parquet`),
gerados por `src/data/features.py`. Regra que todos cumprem (docs/metodologia.md §4.3):
são conhecidos **antes do horário previsto de partida**, porque derivam só da
malha programada (horários previstos, empresa, aeroportos, tipo de linha) ou de
estatísticas calculadas no treino. Números da preparação em
`docs/preparacao-dados.md`.

## 1. Os 29 atributos

### Derivados do horário previsto de partida (`Partida Prevista`)

| Feature | Tipo | Como é calculada | Observação |
|---|---|---|---|
| `hora` | int 0–23 | hora da partida prevista | |
| `minuto` | int 0–59 | minuto da partida prevista | |
| `dia_sem` | int 0–6 | dia da semana (0 = segunda) | |
| `fim_semana` | 0/1 | `dia_sem ≥ 5` (sábado ou domingo) | derivada exata de `dia_sem` |
| `periodo_dia` | int 0–3 | `hora // 6`: 0 = 0–5h, 1 = 6–11h, 2 = 12–17h, 3 = 18–23h | derivada exata de `hora` |
| `dia_mes` | int 1–31 | dia do mês | |
| `mes` | int 1–12 | mês | |
| `trimestre` | int 1–4 | trimestre | derivada exata de `mes` |
| `dia_ano` | int 1–366 | dia do ano | ≈ função de `mes` e `dia_mes` |
| `feriado` | 0/1 | a data é feriado nacional (`holidays.Brazil`, 2024–2026) | 2,2% dos voos (treino + validação) |
| `vespera_feriado` | 0/1 | o dia seguinte é feriado nacional | 2,1% dos voos |

### Duração e congestionamento programados

| Feature | Tipo | Como é calculada | Observação |
|---|---|---|---|
| `duracao_prevista_min` | float | `Chegada Prevista − Partida Prevista`, em minutos | ver limitação 1 |
| `congestion_origem` | int | nº de voos com partida prevista no mesmo aeroporto de origem, na mesma hora cheia (`floor("h")`), contando o próprio voo | mediana 6, p99 33, máx. 36; ver limitação 4 |
| `congestion_destino` | int | idem, para chegadas previstas no aeroporto de destino | ver limitação 4 |

### Tipo de linha (one-hot de `Código Tipo Linha`)

| Feature | Tipo | Como é calculada |
|---|---|---|
| `tipo_linha_C`, `tipo_linha_G`, `tipo_linha_I`, `tipo_linha_N` | 0/1 | uma coluna por código; o one-hot roda na base inteira antes do split, então os três conjuntos têm as mesmas 4 colunas. Volumes na base limpa: N 1.529.990 · I 274.046 · C 29.460 · G 22.484 |

Significado de cada código (natureza da linha, segundo a ANAC):

| Código | Significado | Descrição |
|---|---|---|
| C | Doméstica cargueira | voos nacionais destinados ao transporte de cargas |
| G | Internacional cargueira | voos internacionais destinados ao transporte de cargas |
| I | Internacional mista | voos internacionais de passageiros, que podem transportar carga |
| N | Doméstica mista | voos nacionais de passageiros, que podem transportar carga |

Fonte dos significados: CRUZ (2013), slide 17 ("Natureza da Linha (tipo de
linha)"), que lista também os códigos R (regional mista), E (especial mista),
L (rede postal) e H (sub-regional), ausentes na base de 2024–2025. A coluna
"Descrição" explica os nomes.

> CRUZ, Cleujanio Silva. BAV – Boletim de Alteração de Voos. In: SEMANA DE
> QUALIDADE DA INFORMAÇÃO DO TRANSPORTE AÉREO, 1., 2013, São Paulo.
> *Apresentação*. Brasília, DF: Agência Nacional de Aviação Civil
> (ANAC/SRE/GEAC), 2013. Disponível em:
> https://www.gov.br/anac/pt-br/assuntos/regulados/empresas-aereas/semana-de-qualidade-da-informacao-do-transporte-aereo/arquivos/i-semana/apresentacao/painel_boletim_de_alteracao_de_voos_empresas_brasileiras.pdf.
> Acesso em: 13 out. 2026.

Cardinalidade baixa (4 valores), por isso one-hot é seguro aqui (§3.2).

### Porte dos aeroportos (one-hot de 4 faixas)

| Feature | Tipo | Como é calculada |
|---|---|---|
| `porte_origem_{pequeno,medio,grande,hub}` | 0/1 | faixa do aeroporto de origem |
| `porte_destino_{pequeno,medio,grande,hub}` | 0/1 | faixa do aeroporto de destino |

- As faixas são os **quartis do volume no treino** (partidas + chegadas por
  aeroporto), com `pd.qcut`: 364 aeroportos, 91 por faixa. Limites: pequeno
  ≤ 8 voos; médio 9–378; grande 379–2.065; hub > 2.065.
- Ajustadas **só no treino** (`ajustar_porte_aeroporto`) e aplicadas sem
  reajuste em validação e teste — calcular os quartis na base inteira usaria o
  volume do futuro. Aeroporto que não aparece no treino → "pequeno".
- Categorias fixas desde 11 out (ver limitação 3).

### Target encoding (empresa e aeroportos)

| Feature | Tipo | Codifica |
|---|---|---|
| `empresa_te` | float 0–1 | `ICAO Empresa Aérea` (129 valores no bruto) |
| `origem_te` | float 0–1 | `ICAO Aeródromo Origem` (467 valores) |
| `destino_te` | float 0–1 | `ICAO Aeródromo Destino` (480 valores) |

- Valor = taxa de atraso **suavizada** da categoria:
  `(soma de atrasados + m · média global) / (nº de voos + m)`, com `m = 20`.
  A suavização puxa categorias raras para a média global.
- **Treino: out-of-fold** (`KFold`, 5 folds, `shuffle=True`, `random_state=42`):
  cada linha recebe o valor calculado nos outros 4 folds, nunca com o próprio
  rótulo (§4.10). `verifica_vazamento` prova isso nas categorias com um voo só:
  elas recebem a média global, igual para atrasados e não atrasados.
- **Validação e teste:** mapa ajustado no treino inteiro, aplicado sem
  reajuste. Categoria nova → média global do treino (0,1811).
- Por que não one-hot: aeroportos e empresas gerariam 500–1.000 colunas; o
  GAwLL opera sobre N bits, e N ≤ 40 é decisão fechada (§3.2).

## 2. Colunas do parquet que não são atributos

| Coluna | Por que está no parquet | Por que não é atributo |
|---|---|---|
| `Partida Prevista`, `Chegada Prevista` | split temporal e auditorias por data | timestamps brutos; já viraram os derivados acima |
| `atrasado` | **alvo** (`atraso_min ≥ 15`) | é o que se prevê |
| `atraso_min` | auditoria do alvo | é o alvo contínuo — proibido como atributo |
| `origem` (só na `subamostra.parquet`) | separa treino e validação dentro do GA | é a marcação do conjunto |

`NAO_ATRIBUTOS` em `src/models/baselines.py` remove as quatro primeiras antes
de montar o X; o `gawll.py` remove também `origem`.

## 3. Colunas do VRA que ficaram de fora

| Coluna / atributo | Decisão | Motivo |
|---|---|---|
| `Partida Real`, `Chegada Real`, `Situação Voo`, `Código Justificativa` | **proibidas** (pós-hoc, §4.3) | só existem depois do voo; demonstração no notebook, seção "Vazamento" (a chegada real tem correlação 0,93 com o atraso na partida) |
| `Número Voo` | removida | identificador de alta cardinalidade (11.569 valores, ~3.600 com ≤ 5 voos); redundante com empresa + rota + horário |
| `Código Autorização (DI)` | removida | quase constante: o valor "0" ocupa 96,2% das linhas |
| `ICAO Empresa Aérea`, `ICAO Aeródromo Origem/Destino` (brutas) | substituídas | viraram target encoding + porte |
| `Código Tipo Linha` (bruta) | substituída | virou one-hot |
| `ano` | **removido por decisão** | validação e teste são 100% 2025; o efeito do ano não é estável (muda de sinal conforme o mês) e contaminaria os rankings de importância. Demonstração no notebook, seção "Ano" |
| `regiao_origem`, `regiao_destino` | **cortadas** | exigiriam uma tabela externa ICAO → região; com N = 29, já dentro da faixa 20–40, não compensavam |
| Atraso do voo anterior da mesma aeronave | não implementado (§4.3) | usa dados realizados e exige tratamento temporal fora do prazo |
| Dados meteorológicos | fora do escopo (§3.3) | trabalho futuro |

## 4. A guarda `assert_sem_vazamento`

`src/data/clean.py::assert_sem_vazamento` lança erro se a tabela recebida
tiver qualquer uma de: `Partida Real`, `Chegada Real`, `Situação Voo`,
`Código Justificativa`, `atraso_min`, `atrasado`.

- **Onde é chamada:** sobre a **matriz de atributos X**, não sobre o parquet —
  o parquet carrega `atrasado` e `atraso_min` legitimamente (alvo e auditoria).
  Chamadas: `baselines.py::carregar` (todo modelo e todo método da Etapa 9
  passam por ela) e `gawll.py::carregar_subamostra`.
- **Por que também `atrasado`:** a versão original da lista tinha
  `Chegada Prevista`, que é permitida (horário programado); foi trocada por
  `atrasado`, o alvo, que é o vazamento mais óbvio de todos.

## 5. Limitações conhecidas

1. **`duracao_prevista_min` sem janela de plausibilidade.** No treino +
   validação: mínimo 0 (1 voo), 10 voos com menos de 15 min, **1.638 voos
   (0,1%) acima de 900 min (15 h)**, máximo 1.435 min. Mediana 105, p99 715.
   Voos internacionais de longa distância explicam parte da cauda; valores
   perto de 24 h sugerem erro de registro (causa não investigada). Não tratado
   porque o dataset está congelado; o efeito é pequeno (0,1% das linhas;
   árvores são robustas a valores extremos, e a LogReg vê a coluna padronizada).
2. **Feriados só nacionais.** `holidays.Brazil` sem subdivisões: feriados
   estaduais, municipais e de outros países (voos internacionais) não entram.
3. **`get_dummies` do porte por conjunto — corrigida em 11 out.** O one-hot do
   porte roda separadamente em treino, validação e teste, e derivava as colunas
   dos valores presentes; se uma faixa faltasse num conjunto, os esquemas
   divergiriam. Agora usa categorias fixas (`PORTES` em `features.py`).
   Verificado: rodando a Etapa 5 inteira com o código novo, os três conjuntos
   saem **idênticos** aos congelados (mesmas colunas, mesma ordem, mesmos valores).
4. **Congestionamento conta só voos realizados.** Os cancelados saem no
   `clean.py`, antes da contagem; pelo §4.3 a contagem correta é a da malha
   programada, cancelados incluídos (o cancelamento é informação pós-hoc).
   Auditoria (`src/data/auditar_congestionamento.py` →
   `results/tables/auditoria_congestionamento.csv`): a contagem difere da
   programada em 16,3% das linhas de treino e 11,5% das de validação, e slots
   com cancelamento atrasam mais (validação: 23,3% × 15,3%). Mas, trocando pela
   contagem programada, o F1 de validação muda **−0,001 (XGBoost) e +0,002
   (RF)**: o modelo não aproveita o vazamento. **Decisão:** manter o dataset
   congelado e registrar como limitação (Cap. 3).
5. **Colinearidade entre derivados temporais.** `trimestre` (de `mes`),
   `periodo_dia` (de `hora`) e `fim_semana` (de `dia_sem`) são funções exatas
   de outra feature, e `dia_ano ≈ 30·mes + dia_mes`. A previsão não sofre, mas
   os coeficientes da LogReg ficam distorcidos (`dia_ano` +16,4 e `mes` −16,3 se
   anulam). É o mesmo fenômeno de redundância que o PyGAwLLfs lê como interação
   (ver Diferencial 1, `docs/sintetico.md`).
