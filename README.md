# Avaliação comparativa de métodos de importância e interação de atributos aplicados à previsão de atrasos de voos

**Autor:** Igor Pio — Bacharelado em Ciência da Computação, FFCLRP/USP, Ribeirão Preto
**Orientador:** Prof. Dr. Renato Tinós

**Pergunta de pesquisa:** como diferentes métodos de estimação de importância e
interação de atributos se comparam em termos de **concordância**, **estabilidade**,
**fidelidade** e **custo computacional** no problema de previsão de atrasos de voos?

O método central é o **PyGAwLLfs** (Guimarães et al., 2026), um algoritmo genético
com *linkage learning* que, ao selecionar atributos, produz um vetor de importância
e uma matriz de interação entre as variáveis. Ele é comparado com as
**importâncias intrínsecas** dos modelos, a **Permutation Importance**, o
**TreeSHAP** e, para as interações, a **estatística H de Friedman**, em três modelos:
regressão logística, Random Forest e XGBoost.

## O projeto, fase a fase (CRISP-DM)

O trabalho segue as fases do CRISP-DM (Chapman et al., 2000). Cada fase tem um
documento, uma pasta de código e etapas no `main.py`:

| Fase | Documento | Código | O que acontece |
|---|---|---|---|
| 1. Problema | [`docs/01-problema.md`](docs/01-problema.md) | — | pergunta, objetivos, unidade de análise, alvo e critério de sucesso |
| 2. Dados | [`docs/02-dados.md`](docs/02-dados.md) | `src/dados/` | base VRA da ANAC (2024–2025), qualidade, descartes, sazonalidade e taxas de atraso |
| 3. Preparação | [`docs/03-preparacao.md`](docs/03-preparacao.md) | `src/preparacao/` | 29 atributos conhecidos antes da partida, divisão temporal, balanceamento 50/50 e as checagens de cada passo |
| 4. Modelagem | [`docs/04-modelagem.md`](docs/04-modelagem.md) | `src/modelagem/` | os modelos, os cinco métodos e o PyGAwLLfs corrigido |
| 5. Avaliação | [`docs/05-avaliacao.md`](docs/05-avaliacao.md) | `src/avaliacao/` | desempenho, verificação num dataset com gabarito, os quatro critérios, recortes temporais e interações |
| 6. Interpretação | `results/figures/legendas.md` | `src/interpretacao/` | figuras que explicam cada resultado e exemplos trabalhados |

As decisões de método, com o motivo e a alternativa de cada uma, estão em
[`docs/decisoes.md`](docs/decisoes.md); versões, sementes e testes, em
[`docs/reprodutibilidade.md`](docs/reprodutibilidade.md).

## Em poucas linhas

- **Dados:** 1.855.980 voos regulares realizados de 2024 e 2025 (VRA, ANAC). Alvo:
  atraso na partida de 15 minutos ou mais.
- **Preparação:** treino de janeiro de 2024 a agosto de 2025, validação em setembro
  e outubro, teste em novembro e dezembro de 2025, cada um com metade de voos
  atrasados. O teste só é usado na avaliação final e na fidelidade.
- **Métrica central:** acurácia (o trivial acerta 0,5), usada na escolha dos modelos,
  no fitness do PyGAwLLfs, na Permutation e na fidelidade.
- **PyGAwLLfs:** 200 gerações por execução, 3 modelos × 5 sementes, com o fitness
  medido na validação e os dois erros de implementação do código original (mutação e
  elitismo) corrigidos por subclasse, sem editar o repositório.
- **Comparação:** por rank (Spearman e sobreposição top-k), sempre dentro do mesmo
  modelo antes de comparar entre modelos; estabilidade entre sementes e entre os
  recortes de 2024 e 2025.

## Estrutura do repositório

```
main.py              roda o pipeline inteiro, na ordem, etapa por etapa
requirements.txt     dependências
src/
├── graficos.py      paleta, estilo de impressão e legendas de todas as figuras
├── dados/           download, consolidação e análise exploratória
├── preparacao/      limpeza, alvo, atributos, divisão, balanceamento e checagens
├── modelagem/       modelos, PyGAwLLfs corrigido, intrínseca, Permutation, SHAP e H
├── avaliacao/       desempenho, rankings, sintético, interações, fidelidade, recortes e síntese
└── interpretacao/   figuras de interpretação e exemplos trabalhados
tests/               suítes de teste (scripts com assert)
docs/                um documento por fase, decisões e reprodutibilidade
results/
├── tables/          resultados em CSV
└── figures/         figuras em PNG e PDF, com as legendas em legendas.md
data/, experiments/  dados e checkpoints (não versionados; gerados pelo pipeline)
PyGAwLLfs/           clone do repositório original (não versionado)
```

## Como reproduzir

```bash
git clone https://github.com/IgorPioGH/tcc_pygawllfs.git && cd tcc_pygawllfs
conda create -n tcc_pygawllfs python=3.13 && conda activate tcc_pygawllfs
pip install -r requirements.txt
git clone https://github.com/dgabrl/PyGAwLLfs.git PyGAwLLfs
git -C PyGAwLLfs checkout 3be28a694164e9b2cb99fc9d1fb9482d6a5af454

python main.py --listar    # as etapas, por fase
python main.py --testes    # o pipeline inteiro e as suítes de teste
```

O XGBoost usa a GPU (`device="cuda"`). O pipeline completo leva cerca de 16 horas
num notebook com Core i7 e RTX 4050, quase tudo no PyGAwLLfs; as etapas longas
retomam do último checkpoint. Detalhes em
[`docs/reprodutibilidade.md`](docs/reprodutibilidade.md).

## Referências principais

- CHAPMAN, P. et al. *CRISP-DM 1.0: step-by-step data mining guide*. SPSS, 2000.
- FRIEDMAN, J. H.; POPESCU, B. E. Predictive learning via rule ensembles.
  *Annals of Applied Statistics*, v. 2, n. 3, p. 916–954, 2008.
- GUIMARÃES, D. G.; CARVALHO, S. S.; NASCIMENTO, R. H.; TINÓS, R. Estimating
  variable importance and interaction in machine learning via genetic algorithms.
  In: *Intelligent Systems: BRACIS 2025*. Springer, 2026. p. 478–492.
  DOI: 10.1007/978-3-032-15987-8_32.
- TINÓS, R.; PRZEWOZNICZEK, M. W.; WHITLEY, D.; CHICANO, F. Genetic algorithm with
  linkage learning. In: *GECCO 2023*. p. 981–989. DOI: 10.1145/3583131.3590349.
- ANAC. Histórico de voos: Voo Regular Ativo (VRA). Disponível em:
  https://www.gov.br/anac/pt-br/assuntos/dados-e-estatisticas/historico-de-voos.
