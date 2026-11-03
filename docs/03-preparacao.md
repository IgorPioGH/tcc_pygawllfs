# 3. Preparação dos dados

Terceira fase do CRISP-DM: da base com o alvo até os conjuntos que os modelos
leem. Código em `src/preparacao/` (etapas `preparacao` e `checagens` do
`main.py`):

| Script | O que faz | Produz |
|---|---|---|
| `clean.py` | separa os cancelados e remove os voos sem horário | `data/interim/vra_limpo.parquet` |
| `target.py` | recorte 2024–2025, atraso em minutos, janela de plausibilidade e alvo | `data/interim/vra_target.parquet` |
| `features.py` | atributos, divisão temporal, porte, *target encoding*, balanceamento e subamostra | `data/processed/{train,val,test,subamostra}.parquet` |
| `checagens.py` | confere que o balanceamento, o encoding e a divisão fazem o que dizem | `results/tables/checagem_*.csv` e figuras |

## Divisão temporal

Treino, validação e teste são separados por datas absolutas da partida prevista,
nunca por sorteio: o modelo aprende com o passado e é avaliado no futuro. Cada
corte é um intervalo [início, fim).

| Recorte | Treino | Validação | Teste | Uso |
|---|---|---|---|---|
| principal | jan/2024 – ago/2025 | set–out/2025 | nov–dez/2025 | todo o estudo |
| 2024 | jan–ago/2024 | set–out/2024 | nov–dez/2024 | estabilidade entre recortes |
| 2025 | jan–ago/2025 | set–out/2025 | nov–dez/2025 | estabilidade entre recortes |

Os recortes de 2024 e 2025 têm o mesmo calendário e não compartilham nenhum voo.
O teste de cada recorte nunca é lido. A preparação é uma só
(`features.montar_conjuntos(df, recorte)`); o que muda é a janela de datas.

## Os 29 atributos

Todos são conhecidos antes da partida prevista. Os atributos que não dependem do
alvo (`features_sem_alvo`) são calculados uma vez, na base inteira, antes de
qualquer divisão: o congestionamento, por exemplo, conta os voos de toda a malha.

**Do horário previsto de partida**

| Atributo | Valores | Cálculo |
|---|---|---|
| `hora` | 0–23 | hora da partida prevista |
| `minuto` | 0–59 | minuto da partida prevista |
| `dia_sem` | 0–6 | dia da semana (0 = segunda-feira) |
| `fim_semana` | 0/1 | `dia_sem ≥ 5` (função exata de `dia_sem`) |
| `periodo_dia` | 0–3 | `hora // 6` (função exata de `hora`) |
| `dia_mes` | 1–31 | dia do mês |
| `mes` | 1–12 | mês |
| `trimestre` | 1–4 | trimestre (função exata de `mes`) |
| `dia_ano` | 1–366 | dia do ano (quase função de `mes` e `dia_mes`) |
| `feriado` | 0/1 | a data é feriado nacional (`holidays.Brazil`) |
| `vespera_feriado` | 0/1 | o dia seguinte é feriado nacional |

**Duração e congestionamento programados**

| Atributo | Cálculo |
|---|---|
| `duracao_prevista_min` | chegada prevista − partida prevista, em minutos |
| `congestion_origem` | voos com partida prevista no mesmo aeroporto de origem, na mesma hora cheia (inclui o próprio voo) |
| `congestion_destino` | idem, para as chegadas previstas no aeroporto de destino |

**Tipo de linha** (*one-hot* de `Código Tipo Linha`): `tipo_linha_C` (doméstica
cargueira), `tipo_linha_G` (internacional cargueira), `tipo_linha_I`
(internacional mista) e `tipo_linha_N` (doméstica mista), segundo a natureza da
linha definida pela ANAC (Cruz, 2013). Na base limpa: N 1.529.990, I 274.046,
C 29.460 e G 22.484 voos.

**Porte dos aeroportos** (*one-hot* de 4 faixas): `porte_origem_*` e
`porte_destino_*` ∈ {pequeno, médio, grande, hub}. As faixas são os quartis do
volume de pousos e decolagens no treino (364 aeroportos, 91 por faixa: pequeno
até 8 voos, médio de 9 a 378, grande de 379 a 2.065, hub acima disso). O mapa é
ajustado só no treino e aplicado sem reajuste na validação e no teste; aeroporto
fora do mapa vira "pequeno".

**Target encoding** (`empresa_te`, `origem_te`, `destino_te`): a taxa de atraso
suavizada da empresa e dos aeroportos de origem e de destino,

```
encoding(categoria) = (atrasados + m · média global) / (voos + m),   m = 20
```

A suavização puxa as categorias raras para a média global (0,181 no treino). No
treino, cada voo recebe o valor calculado nas outras 4 de 5 dobras (*out-of-fold*),
nunca com o próprio rótulo; `verifica_vazamento` confere isso a cada execução nas
categorias com um voo só. Validação e teste recebem o mapa ajustado no treino
inteiro; categoria nova recebe a média global. *One-hot* de aeroportos e empresas
geraria mais de mil colunas, e o PyGAwLLfs trabalha com um bit por atributo.

### O encoding generaliza?

O valor do encoding ajustado no treino acompanha a taxa de atraso observada na
validação (sem balancear): correlação de Pearson de 0,54 entre as 73 empresas e de
0,64 entre os 221 aeroportos de origem presentes nos dois períodos
(`checagem_target_encoding.csv`, figura `checagem_target_encoding`). O encoding
carrega informação real fora do período em que foi ajustado.

## Balanceamento 50/50

Cada conjunto (treino, validação e teste) fica com metade de voos atrasados:
todos os atrasados são mantidos e o mesmo número de voos pontuais é sorteado, sem
reposição (`features.balancear`). O balanceamento é feito depois do porte e do
encoding, então as taxas que o encoding carrega são as reais.

O sorteio é **global em cada conjunto**, nunca dentro de um grupo (mês, aeroporto,
empresa): balancear dentro de um grupo deixaria a taxa de atraso de todos os
grupos em 50% e apagaria a informação da variável. Descartar pontuais ao acaso
multiplica as chances (*odds*) de atraso de todos os grupos pelo mesmo fator, o
inverso da fração de pontuais mantida, então a ordem entre os grupos se mantém. É
o argumento da amostragem caso-controle, em que só o intercepto de uma regressão
logística muda (King e Zeng, 2001).

**Conferência** (`checagem_balanceamento.csv`, figura `checagem_balanceamento`):
a taxa de atraso por hora no treino original e no balanceado tem Spearman de
0,997, e a razão das *odds* fica entre 4,44 e 4,58 em todas as 24 horas, perto do
valor teórico (1.260.763 pontuais / 278.759 mantidos ≈ 4,52).

| Conjunto | Período | Voos | Atrasados | Taxa original | Voos após o balanceamento |
|---|---|---:|---:|---:|---:|
| treino | jan/2024 – ago/2025 | 1.539.522 | 278.759 | 18,1% | 557.518 |
| validação | set–out/2025 | 158.389 | 25.613 | 16,2% | 51.226 |
| teste | nov–dez/2025 | 158.069 | 38.355 | 24,3% | 76.710 |

Figura: `checagem_divisao` (linha do tempo da divisão).

Com os conjuntos 50/50, a acurácia passa a ser legível (o classificador trivial
acerta 0,5) e os modelos dispensam pesos de classe.

## Subamostra do PyGAwLLfs

O PyGAwLLfs retreina o modelo milhares de vezes e não cabe no treino completo. Ele
usa uma subamostra de 50.000 voos de treino e validação, estratificada pelo período
e pelo alvo, com semente fixa (`features.criar_subamostra`): cerca de 45.800 voos de
treino e 4.200 de validação. A subamostra é usada só dentro do PyGAwLLfs; os
modelos finais, os outros métodos e a fidelidade usam os conjuntos completos. A
checagem de que treinar na subamostra não muda o que é importante está em
[`05-avaliacao.md`](05-avaliacao.md).

## Uso dos conjuntos

| Etapa | Treino | Validação | Teste | Subamostra |
|---|---|---|---|---|
| ajuste do porte e do encoding | sim | — | — | — |
| treino dos modelos | sim | — | — | — |
| escolha de hiperparâmetros | — | sim | — | — |
| importância intrínseca | (o modelo treinado) | — | — | — |
| Permutation, SHAP, H de Friedman | — | sim | — | — |
| PyGAwLLfs | — | — | — | sim |
| avaliação final dos modelos (matriz de confusão, ROC, erros por dia e hora) | — | — | sim | — |
| fidelidade | treino | validação | sim (por último) | — |

O teste aparece só na avaliação final e na fidelidade, sempre depois de tudo ter
sido escolhido. Nenhum método de importância ou de interação o lê, e os testes
automáticos falham se algum tentar.

## Guarda contra vazamento

`clean.assert_sem_vazamento` levanta erro se a matriz de atributos tiver Partida
Real, Chegada Real, Situação Voo, Código Justificativa, `atraso_min` ou
`atrasado`. Ela é chamada antes de todo treino (`baselines.carregar`,
`gawll.separar_subamostra`, recortes e checagem da subamostra). Ficam fora também
`Número Voo` (identificador com 11.569 valores), `Código Autorização (DI)` (96,2%
de zeros) e as colunas brutas que viraram encoding, porte ou *one-hot*.

## Limitações da preparação

1. **Duração sem janela de plausibilidade:** no treino e na validação, 1.638 voos
   (0,1%) têm duração prevista acima de 15 horas, alguns perto de 24 horas, o que
   sugere erro de registro. O efeito é pequeno: árvores são robustas a valores extremos, e a
   regressão logística vê a coluna padronizada.
2. **Feriados só nacionais:** feriados estaduais, municipais e de outros países
   (voos internacionais) não entram.
3. **Congestionamento só com voos realizados:** os cancelados saem antes da
   contagem; a malha programada completa os incluiria.
4. **Colinearidade entre derivados de data:** `trimestre`, `periodo_dia` e
   `fim_semana` são funções exatas de outro atributo, e `dia_ano` quase é. A
   previsão não sofre, mas os coeficientes da regressão logística se distorcem
   (`dia_ano` e `mes` recebem coeficientes grandes e de sinais opostos).

## Referências

- CRUZ, C. S. BAV – Boletim de Alteração de Voos. In: SEMANA DE QUALIDADE DA
  INFORMAÇÃO DO TRANSPORTE AÉREO, 1., 2013, São Paulo. *Apresentação*. Brasília,
  DF: ANAC, 2013. Disponível em:
  https://www.gov.br/anac/pt-br/assuntos/regulados/empresas-aereas/semana-de-qualidade-da-informacao-do-transporte-aereo/arquivos/i-semana/apresentacao/painel_boletim_de_alteracao_de_voos_empresas_brasileiras.pdf.
  Acesso em: 13 out. 2026.
- KING, G.; ZENG, L. Logistic regression in rare events data. *Political
  Analysis*, v. 9, n. 2, p. 137–163, 2001.
