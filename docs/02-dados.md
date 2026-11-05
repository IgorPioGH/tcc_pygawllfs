# 2. Entendimento dos dados

Segunda fase do CRISP-DM: de onde vêm os dados, qual a sua qualidade, o que sai
na limpeza e como o atraso se distribui. Tudo o que envolve o alvo usa só voos
com partida prevista até 31/10/2025: novembro e dezembro de 2025 são o período
de teste e ficam fora de qualquer decisão. Os números deste documento são saídas
de `src/dados/eda.py` (etapa `eda`), em `results/tables/eda_*.csv` e
`results/figures/eda_*`.

## Fonte e versão

- **Base:** VRA (Voo Regular Ativo), série histórica de voos da ANAC
  (https://www.gov.br/anac/pt-br/assuntos/dados-e-estatisticas/historico-de-voos).
- **Período:** janeiro de 2024 a dezembro de 2025, 24 arquivos CSV mensais
  (`src/dados/download.py`), com separador `;`, codificação UTF-8 com BOM e uma
  linha extra antes do cabeçalho (`src/dados/ler_dados.py`).
- **Versão:** arquivos com a marca "Atualizado em: 2026-10-07", baixados em
  23/10/2026. A ANAC atualiza a série: outra versão pode mudar registros e a ordem
  das linhas, o que altera um pouco os resultados
  ([`reprodutibilidade.md`](reprodutibilidade.md)). `eda.py` para se o total de
  registros mudar.

## Qualidade da base bruta

A base consolidada tem **1.992.831 registros** e 12 colunas
(`eda_qualidade.csv`):

| Coluna | Nulos | Valores distintos | Observação |
|---|---:|---:|---|
| ICAO Empresa Aérea | 0 | 129 | 16 empresas com menos de 5 voos |
| Número Voo | 0 | 11.569 | identificador; 3.276 números com menos de 5 voos |
| Código Autorização (DI) | 0 | 10 | o valor "0" ocupa 96,2% das linhas |
| Código Tipo Linha | 464 | 4 | N (doméstica mista) em 80,3% |
| ICAO Aeródromo Origem | 0 | 467 | 142 aeroportos com menos de 5 partidas |
| ICAO Aeródromo Destino | 0 | 480 | 157 aeroportos com menos de 5 chegadas |
| Partida Prevista | 59.140 | — | sem horário previsto não há atraso |
| Partida Real | 69.496 | — | vazia exatamente nos cancelados |
| Chegada Prevista | 59.142 | — | |
| Chegada Real | 69.496 | — | vazia exatamente nos cancelados |
| Situação Voo | 0 | 2 | REALIZADO (96,5%) ou CANCELADO |
| Código Justificativa | 1.992.831 | 0 | 100% vazia no período |

## Colunas que vazariam o alvo

Quatro colunas só existem depois do voo e ficam fora dos atributos
(`eda_vazamento.csv`):

| Coluna | Por que vazaria |
|---|---|
| Situação Voo | é preenchida depois do dia do voo; todo cancelado está sem horário real, e todo realizado tem |
| Partida Real | é o próprio atraso |
| Chegada Real | o atraso na chegada tem correlação de 0,93 com o atraso na partida, e 75% dos voos que partem atrasados chegam atrasados |
| Código Justificativa | está vazia no período, mas, se viesse preenchida, seria a justificativa do atraso, ou seja, o alvo explicado |

O ano também fica de fora: validação e teste são 100% de 2025, então o modelo só
aprenderia a diferença entre 2024 e 2025 no treino, e essa diferença nunca seria
testada.

## Descartes

| Passo | Critério | Removidos | Restam |
|---|---|---:|---:|
| base bruta | 24 CSVs mensais, 2024–2025 | — | 1.992.831 |
| cancelados | Situação Voo diferente de REALIZADO | 69.496 | 1.923.335 |
| sem horário | Partida Prevista ou Partida Real vazia | 59.140 | 1.864.195 |
| fora do período | Partida Prevista fora de 2024–2025 | 11 | 1.864.184 |
| atraso implausível | atraso fora de [−40, 720] min | 8.204 | **1.855.980** |

A janela de plausibilidade é assimétrica: uma empresa não antecipa uma partida em
mais de alguns minutos (o passageiro perderia o voo), mas atrasos de horas
existem. Valores como quatro dias de adiantamento ou 31 dias de atraso são erros
de registro da data. Figura: `eda_funil`.

## O alvo

Até outubro de 2025 são 1.697.911 voos, com **17,9% de atrasados** (atraso na
partida de 15 minutos ou mais). A mediana do atraso é −1 minuto: a maior parte dos
voos parte no horário ou um pouco antes, e a cauda é longa (0,7% passa de 3 horas).
Figura: `eda_histograma_atraso`.

No período de teste (novembro e dezembro de 2025), a taxa sobe para 24,3%: é o
pico sazonal do fim do ano.

## Sazonalidade e padrões por atributo

- **Mês:** a taxa de atraso vai de 14,8% (maio) a 26,8% (dezembro), com picos em
  dezembro e janeiro. A série mensal de 2024 e de 2025 tem a mesma forma, mas
  níveis diferentes em alguns meses (`eda_serie_mensal`). Por isso os recortes
  temporais usam o mesmo calendário nos dois anos.
- **Hora:** de 9,5% (6 h) a 22,7% (17 h); os atrasos se acumulam ao longo do dia.
- **Dia da semana:** de 16,0% (domingo) a 20,9% (sexta-feira).
- **Tipo de linha:** 15,6% nas domésticas mistas (N) e 53,2% nas internacionais
  cargueiras (G).
- **Duração prevista:** o quintil mais longo (acima de 190 min) atrasa 26,6%,
  contra 15,1% a 16,5% nos demais.
- **Empresas e aeroportos:** entre as 10 maiores empresas, a taxa vai de 13,9% a
  40,2%; entre os 15 maiores aeroportos de origem, de 10,6% (Santos Dumont) a 23,6%
  (Guarulhos) (`eda_taxa_categorias`). São muitas categorias com taxas muito
  diferentes, o que motiva o *target encoding* da preparação.

Figuras: `eda_taxa_calendario` e `eda_taxa_categorias`; tabela completa em
`eda_taxa_por_atributo.csv`.

**Ressalva:** a taxa por atributo é uma relação marginal. Ela não isola o efeito
de cada variável (a taxa de um aeroporto, por exemplo, mistura o efeito do
aeroporto com o dos horários e das empresas que operam nele). É por isso que o
trabalho usa modelos e métodos de importância, e não só tabelas como estas.

## Correlação entre os atributos

A matriz de Spearman entre os 29 atributos do treino (`eda_correlacao`) mostra,
antes de qualquer modelo, os pares **redundantes por construção**: `hora` ×
`periodo_dia`, `dia_sem` × `fim_semana` e o bloco `mes` × `trimestre` ×
`dia_ano`. As dummies de uma mesma variável (tipo de linha, porte do aeroporto)
aparecem com correlação negativa entre si. Esses pares reaparecem nas
interações do PyGAwLLfs ([`05-avaliacao.md`](05-avaliacao.md)).
