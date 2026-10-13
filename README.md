# Avaliação comparativa de métodos de importância e interação de atributos aplicados à previsão de atrasos de voos

**Autor:** Igor Pio — Bacharelado em Ciência da Computação, FFCLRP/USP, Ribeirão Preto
**Orientador:** Prof. Dr. Renato Tinós

**Pergunta de pesquisa:** como diferentes métodos de estimação de importância e
interação de atributos se comparam em termos de **concordância**, **estabilidade**,
**fidelidade** e **custo computacional** no problema de previsão de atrasos de voos?

O método central é o **PyGAwLLfs** (Guimarães et al., 2026), um algoritmo genético
com *linkage learning* que, ao selecionar atributos, produz como subproduto um
vetor de importância e uma matriz de interação entre as variáveis. Ele é comparado
com **Permutation Importance**, **TreeSHAP**, as **importâncias intrínsecas** dos
modelos e, para as interações, com a **estatística H de Friedman**.

---

## Em resumo

- **Os métodos concordam sobre o que importa**: aeroporto de origem, empresa e
  hora aparecem no topo de quase todos; a concordância é maior dentro do mesmo
  modelo (Spearman médio 0,70) do que entre modelos diferentes (0,56–0,60).
- **Todos apontam atributos que de fato carregam a informação**: com só as 5
  features mais importantes de cada método, o modelo retém 87%–98% do PR-AUC do
  modelo completo, contra 72%–81% de 5 features sorteadas.
- **Os métodos discordam sobre o que interage**: PyGAwLLfs e H de Friedman quase
  não concordam (Spearman ≈ 0,1). Num dataset sintético com gabarito, os dois
  acham a interação verdadeira, mas **só o PyGAwLLfs trata pares redundantes
  como interação** — ele mede a não aditividade do *desempenho*, e redundância
  produz exatamente esse sinal.
- **O custo difere em duas ordens de grandeza**: 5,4 h para o PyGAwLLfs contra
  segundos a minutos para os demais — em troca, ele também seleciona os atributos.
- Ao adaptar o PyGAwLLfs, foram corrigidas três questões metodológicas (fitness
  calculado no conjunto de teste, acurácia num problema desbalanceado e divisão
  aleatória dos dados) e **dois bugs de implementação** do repositório original.

---

## Dados, métodos e correções

**Dados.** VRA (Voo Regular Ativo) da ANAC, 2024–2025: 1.855.980 voos realizados
após a limpeza. Alvo: atraso na partida ≥ 15 min (18,5% dos voos).
**Split temporal**, nunca aleatório: treino jan/2024–ago/2025 (1,54 mi),
validação set–out/2025 (158 mil), teste nov–dez/2025 (158 mil, no pico sazonal:
24% de atrasos). **29 atributos**, todos conhecidos antes da partida (horário
previsto e derivados, duração, congestionamento programado, tipo de linha, porte
dos aeroportos, *target encoding* out-of-fold de empresa e aeroportos).

<p align="center"><img src="results/figures/histograma_atraso.png" width="620"></p>

**Modelos.** Regressão logística, Random Forest e XGBoost. No teste, F1 de 0,42,
0,44 e 0,45 (o classificador trivial tem 76% de acurácia e F1 = 0). A
dificuldade é esperada: clima e propagação de atrasos ficaram de fora de
propósito, para não usar informação posterior à partida.

**Métodos comparados.**

| Método | O que mede | Onde é calculado |
|---|---|---|
| PyGAwLLfs | variação do fitness (F1 + parcimônia) ao ligar/desligar atributos, retreinando o modelo | subamostra de 50 mil voos, validação |
| Permutation | queda de F1 ao embaralhar cada atributo no modelo treinado | validação |
| TreeSHAP | contribuição média de cada atributo à previsão (RF, XGBoost) | validação |
| Intrínseca | coeficiente padronizado (LogReg), MDI (RF), ganho (XGBoost) | o próprio modelo |
| H de Friedman | não aditividade da função do modelo para cada par (interações) | validação |

O conjunto de teste só é usado na avaliação final dos modelos e na fidelidade.

**Correções feitas no PyGAwLLfs** (por subclasse, sem editar o repositório original):

1. fitness calculado na **validação** (o original usa o conjunto de teste);
2. **F1** no lugar da acurácia;
3. **split temporal** no lugar da divisão aleatória;
4. **mutação**: o operador original devolve um array novo que é ignorado, então nenhum bit era invertido;
5. **elitismo**: o original grava o melhor indivíduo na população antiga, que é descartada.

Os dois bugs são reproduzidos e testados em `tests/test_gawll.py`.

---

## Principais resultados

Tabelas em `results/tables/`, figuras em `results/figures/` (PNG e PDF). Índice
completo, por critério, em [`docs/resultados.md`](docs/resultados.md); relatório
final, com o método, os resultados e as respostas às perguntas de pesquisa, em
[`results/relatorio-final.md`](results/relatorio-final.md).

### 1. Concordância entre os métodos

<p align="center"><img src="results/figures/concordancia_intra_modelo.png" width="820"></p>

Cada célula compara os rankings de dois métodos no mesmo modelo — Spearman em
cima, RBO (que pesa mais o topo do ranking) embaixo. A concordância é
moderada a alta, e maior dentro do mesmo modelo (0,70) do que entre modelos
(0,56–0,60): o modelo explicado pesa tanto quanto o método. O PyGAwLLfs é o que
menos concorda com os outros (0,62 nas duplas com ele, contra 0,79 nas sem ele).

### 2. Estabilidade entre sementes

<p align="center"><img src="results/figures/estabilidade_sementes.png" width="760"></p>

Permutation e SHAP são praticamente idênticos de uma semente para outra
(Spearman ≥ 0,99): sobre um modelo fixo, só muda o embaralhamento ou a amostra.
No PyGAwLLfs a semente muda a busca inteira: a importância varia mais
(0,56–0,80), mas o topo do ranking é razoavelmente estável. Já a **matriz de
interação** do PyGAwLLfs não se repete entre execuções: com os parâmetros do
artigo e 29 atributos, cada par é sondado em média 0,7 vez por execução.

### 3. Fidelidade por retenção

<p align="center"><img src="results/figures/fidelidade.png" width="900"></p>

O modelo é retreinado só com as top-k features de cada método e avaliado no
teste. Todos ficam muito acima do acaso (faixa cinza = 10 sorteios de k
features) e, a partir de 10 features, praticamente igualam o modelo completo.
No top-5, a Permutation é a mais fiel no RF e no XGBoost (97%–98% do PR-AUC) e
o **PyGAwLLfs é o mais fiel na regressão logística** (98%): os coeficientes da
LogReg põem no topo atributos de calendário colineares, que funcionavam na
validação e não se transferem para o período de teste. O subconjunto
selecionado pelo PyGAwLLfs (estrelas), com cerca de um terço das features,
retém 95%–99% do PR-AUC.

### 4. Custo computacional

| Método | Tempo total (3 modelos, 5 sementes) | Treinos de modelo |
|---|---:|---:|
| Intrínseca | ~2 s | 0 (lê o modelo treinado) |
| Permutation | 5,4 min | 0 |
| TreeSHAP | 11,7 min | 0 |
| H de Friedman | 6,8 min | 0 |
| **PyGAwLLfs** | **5,4 h** | **3.200–4.400 por execução** |

Os demais métodos explicam um modelo já treinado; o PyGAwLLfs retreina o modelo
milhares de vezes — e entrega, além da importância, a seleção de atributos e a
matriz de interação.

### 5. Interações: PyGAwLLfs × H de Friedman

<p align="center"><img src="results/figures/interacao_gawll_x_h.png" width="860"></p>

Cada ponto é um par de atributos: a posição no ranking de interação do
PyGAwLLfs (eixo x) contra a do H (eixo y). A nuvem não segue a diagonal: os
dois métodos praticamente não concordam (Spearman 0,08–0,12). Duas causas:
a matriz do PyGAwLLfs é pouco confiável com este orçamento (seção 2), e os
métodos medem coisas diferentes. O dataset sintético com gabarito mostra a
diferença com clareza:

| Método | Par com interação verdadeira (x1, x2) | Pares redundantes (x3, x5) e (x4, x6) |
|---|---|---|
| PyGAwLLfs (RF, XGBoost) | 1º de 45 | **2º–3º** |
| H de Friedman (RF, XGBoost) | 1º de 45 | 7º–44º |

Os dois acham a interação verdadeira; só o PyGAwLLfs põe a redundância no topo.
Nos dados reais, 8 das 15 arestas mais fortes do PyGAwLLfs (top-5 de cada
modelo) são pares redundantes derivados da data ou da hora (ex.: `mes` ×
`dia_ano`, `hora` × `periodo_dia`).

### 6. Validação de domínio das interações

<p align="center"><img src="results/figures/grafo_interacao.png" width="900"></p>

Grafo reduzido do PyGAwLLfs (5% das arestas mais fortes), com cada aresta
classificada: **operacional** (azul), **redundante** (laranja) ou
**estrutural** (cinza, duas codificações da mesma variável). Os pares
operacionais foram testados nos dados: a taxa de atraso no cruzamento das duas
variáveis difere do que os efeitos isolados preveem, **nos dois anos**? Dos 19
operacionais, 2 foram corroborados descritivamente (e nenhum dos 10 pares de
controle):

<p align="center"><img src="results/figures/dominio_hora_x_origem.png" width="860"></p>

Partidas de madrugada em Guarulhos (SBGR) e no Galeão (SBGL) atrasam muito mais
do que "ser esse aeroporto" e "ser essa hora" explicariam somados (à direita,
vermelho = mais atraso que o esperado sem interação). O segundo par corroborado
é hora × tipo de linha. A corroboração é observacional: um fator não observado
poderia produzir o mesmo padrão. Aplicado ao top-10 do H de Friedman, o mesmo
teste corroborou 9 de 20 pares, contra 1 de 20 do PyGAwLLfs.

### Síntese

Os métodos **concordam sobre o que importa** e **discordam sobre o que
interage** — e a discordância se explica por **redundância** (que o PyGAwLLfs lê
como interação e o H não) e por **estabilidade** (a matriz de interação do
PyGAwLLfs precisa de mais sondagens do que o orçamento do artigo oferece com 29
atributos). Para ranquear importância num modelo já treinado, Permutation e
SHAP dão fidelidade igual ou melhor a uma fração do custo; o PyGAwLLfs se
justifica quando o objetivo é **selecionar** atributos e detectar redundância,
e foi o mais robusto à colinearidade na regressão logística.

---

## Estrutura do repositório

```
main.py                    roda o pipeline inteiro, na ordem (ver "Como rodar")
requirements.txt           dependências (pip freeze do ambiente conda)
src/
├── data/                  aquisição, limpeza, alvo, atributos, figuras e auditorias
│   ├── download.py            baixa os CSVs mensais do VRA (ANAC)
│   ├── ler_dados.py           consolida os 24 meses
│   ├── clean.py               separa cancelados; guarda contra vazamento
│   ├── target.py              alvo (atraso >= 15 min) e janela de plausibilidade
│   ├── features.py            29 atributos, split temporal, target encoding OOF, subamostra
│   ├── figuras.py             histograma do atraso
│   └── auditar_congestionamento.py   mede uma limitação dos atributos
├── models/                LogReg, RF e XGBoost (baselines.py) e métricas (evaluate.py)
├── importance/            PyGAwLLfs corrigido (gawll.py) e os métodos de importância
├── interaction/           H de Friedman e grafo de interação reduzido
├── compare/               concordância, estabilidade, interações, fidelidade e síntese
├── synthetic/             dataset sintético com gabarito (Diferencial 1)
└── domain/                validação de domínio das interações (Diferencial 4)
tests/                     testes (scripts com assert), um por etapa
results/
├── tables/                todos os resultados em CSV
└── figures/               figuras em PNG e PDF
docs/                      metodologia, decisões e a interpretação de cada resultado
notebooks/                 auditoria exploratória dos dados brutos (sem lógica de pipeline)
data/, experiments/runs/   dados e checkpoints (não versionados; gerados pelo pipeline)
PyGAwLLfs/                 clone do repositório original (não versionado; ver abaixo)
```

---

## Como rodar

### Requisitos

- Linux (testado no Ubuntu), **Python 3.13** em ambiente **conda**.
- **GPU NVIDIA com CUDA para o XGBoost** (`device="cuda"`). Sem GPU, troque
  `device="cuda"` por `device="cpu"` em `criar_modelo`
  (`src/models/baselines.py`); os tempos do XGBoost aumentam.
- Testado num notebook com Intel Core i7 (16 núcleos lógicos), 14 GB de RAM e
  RTX 4050 (6 GB).
- ~450 MB de disco para os dados e os checkpoints.

### Instalação

```bash
git clone https://github.com/IgorPioGH/tcc_pygawllfs.git && cd tcc_pygawllfs

conda create -n tcc_pygawllfs python=3.13
conda activate tcc_pygawllfs
pip install -r requirements.txt

# PyGAwLLfs original, na versão usada no trabalho (é só importado, nunca editado)
git clone https://github.com/dgabrl/PyGAwLLfs.git PyGAwLLfs
git -C PyGAwLLfs checkout 3be28a694164e9b2cb99fc9d1fb9482d6a5af454
```

⚠️ O projeto e o clone têm, cada um, uma pasta `src/` sem `__init__.py`; as
duas se combinam no `sys.path`. Não crie `__init__.py` em nenhuma delas.

### Executar

Sempre da raiz do repositório, com o ambiente ativado:

```bash
python main.py --listar               # etapas, o que fazem e o tempo esperado
python main.py                        # o pipeline inteiro, do download à fidelidade
python main.py --testes               # idem, e roda as suítes de teste no fim
python main.py --de importancias      # retoma a partir de uma etapa
python main.py --so rankings dominio  # só algumas etapas
python main.py --simular              # mostra os comandos, sem rodar
```

O pipeline inteiro leva cerca de 7,5 h, quase tudo no PyGAwLLfs. Para o notebook
não suspender no meio:

```bash
systemd-inhibit --what=idle:sleep python main.py
```

As etapas longas **retomam de onde pararam**: o PyGAwLLfs e o sintético pulam as
execuções que já têm checkpoint em `experiments/runs/`. Se uma etapa falhar, o
`main.py` para e diz como retomar. O log fica em `experiments/runs/pipeline.log`.
Cada script também roda sozinho (`python src/<pasta>/<arquivo>.py`).

### Tempo esperado

Medido na máquina acima (os tempos marcados com ~ são aproximados):

| Etapa (`main.py`) | O que faz | Tempo |
|---|---|---:|
| `download` | 24 CSVs mensais do VRA (pula os já baixados) | ~5 min (rede) |
| `dados` | consolidação, limpeza, alvo, atributos, split, subamostra | ~3 min |
| `baselines` | tuning na validação, avaliação única no teste | ~3 min |
| `pygawllfs` | 15 execuções (LogReg 5 min, RF 26 min, XGBoost 33 min cada) | **5,4 h** |
| `importancias` | intrínseca, Permutation, TreeSHAP | 18 min |
| `rankings` | concordância, estabilidade, figuras | < 1 min |
| `sintetico` | Diferencial 1 (o GA no sintético leva ~30 min na primeira vez) | 30 min |
| `dominio` | Diferencial 4 | 1 min |
| `interacoes` | H de Friedman e comparação | 9 min |
| `fidelidade` | 138 treinos no treino completo, avaliados no teste | 50 min |
| `sintese` | quadro-síntese dos quatro critérios (só lê CSV) | < 1 s |
| `extras` | histograma e auditoria do congestionamento | ~4 min |
| `--testes` | 8 suítes, 40 testes | ~6 min |

### Testes

```bash
for teste in tests/test_*.py; do python "$teste" || break; done
```

Rodam depois do pipeline (leem os resultados gerados). São scripts com
`assert` (sem pytest). Entre outras coisas, verificam que
nenhum método de importância abre o conjunto de teste, que os bugs do
PyGAwLLfs original existem e foram corrigidos, que o H de Friedman acerta casos
com resposta conhecida e que o modelo com todas as features da fidelidade
reproduz exatamente os baselines.

---

## Documentação

| Documento | Conteúdo |
|---|---|
| [`results/relatorio-final.md`](results/relatorio-final.md) | relatório final: o que foi feito, como, resultados e respostas às perguntas de pesquisa |
| [`docs/resultados.md`](docs/resultados.md) | índice de todos os resultados, pelos quatro critérios |
| [`docs/metodologia.md`](docs/metodologia.md) | regras metodológicas (as referências "§x.y" do código apontam para cá) |
| [`docs/preparacao-dados.md`](docs/preparacao-dados.md) | descartes, alvo, split temporal e subamostra |
| [`docs/dicionario-features.md`](docs/dicionario-features.md) | os 29 atributos, exclusões e limitações |
| [`docs/sintetico.md`](docs/sintetico.md) | Diferencial 1: dataset sintético com gabarito |
| [`docs/validacao-dominio.md`](docs/validacao-dominio.md) | Diferencial 4: validação de domínio das interações |
| [`docs/interacoes-h.md`](docs/interacoes-h.md) | PyGAwLLfs × H de Friedman |
| [`docs/fidelidade.md`](docs/fidelidade.md) | fidelidade por retenção |
| [`docs/piloto-custo.md`](docs/piloto-custo.md) | custo computacional do PyGAwLLfs |
| [`docs/decisoes.md`](docs/decisoes.md) | decisões em aberto e como revertê-las |
| [`docs/aquisicao-vra.md`](docs/aquisicao-vra.md) | origem e download dos dados |

## Limitações

- Um único conjunto de dados e um único período de teste (nov–dez, pico sazonal).
- O orçamento do artigo (população 50, 100 gerações) deixa a matriz de
  interação do PyGAwLLfs esparsa com 29 atributos.
- O H de Friedman, baseado em dependência parcial, extrapola para combinações
  que não existem nos dados quando os atributos são dependentes.
- A validação de domínio é observacional (corroboração, não prova).
- O congestionamento programado conta só voos realizados; o impacto medido é
  desprezível (`docs/dicionario-features.md`).

## Referências principais

- GUIMARÃES, D. G.; CARVALHO, S. S.; NASCIMENTO, R. H.; TINÓS, R. Estimating
  variable importance and interaction in machine learning via genetic algorithms.
  In: *Intelligent Systems: 35th Brazilian Conference, BRACIS 2025, Part I*.
  Springer, 2026. p. 478–492. (LNAI 16179). DOI: 10.1007/978-3-032-15987-8_32.
- TINÓS, R.; PRZEWOZNICZEK, M. W.; WHITLEY, D.; CHICANO, F. Genetic algorithm with
  linkage learning. *Proceedings of the Genetic and Evolutionary Computation
  Conference (GECCO 2023)*, p. 981–989. DOI: 10.1145/3583131.3590349.
- FRIEDMAN, J. H.; POPESCU, B. E. Predictive learning via rule ensembles.
  *The Annals of Applied Statistics*, v. 2, n. 3, p. 916–954, 2008.
  DOI: 10.1214/07-AOAS148.
- INGLIS, A.; PARNELL, A.; HURLEY, C. B. Visualizing variable importance and
  variable interaction effects in machine learning models. *Journal of
  Computational and Graphical Statistics*, v. 31, n. 3, p. 766–778, 2022.
  DOI: 10.1080/10618600.2021.2007935.
- WEBBER, W.; MOFFAT, A.; ZOBEL, J. A similarity measure for indefinite rankings.
  *ACM Transactions on Information Systems*, v. 28, n. 4, p. 1–38, 2010.
  DOI: 10.1145/1852102.1852106.
- ANAC. Histórico de voos: Voo Regular Ativo (VRA).
- CRUZ, C. S. BAV – Boletim de Alteração de Voos. In: *I Semana de Qualidade da
  Informação do Transporte Aéreo*, São Paulo, 2013. Apresentação. ANAC, 2013.
  [PDF](https://www.gov.br/anac/pt-br/assuntos/regulados/empresas-aereas/semana-de-qualidade-da-informacao-do-transporte-aereo/arquivos/i-semana/apresentacao/painel_boletim_de_alteracao_de_voos_empresas_brasileiras.pdf)
  — codificação do tipo de linha (slide 17).
