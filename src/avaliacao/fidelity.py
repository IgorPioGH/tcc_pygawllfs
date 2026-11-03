"""Avaliação — fidelidade por retenção: as features que cada método aponta bastam?

Recebe o ranking_consolidado.csv (etapa rankings), o selecao_gawll.csv (etapa importancias),
os conjuntos congelados train/val/test e os hiperparâmetros da etapa baselines.
Para cada modelo (mesma classe e hiperparâmetros da etapa baselines, treinado no
treino COMPLETO), treina e avalia:
    top-k        as k features mais importantes de cada método (k = 5, 10, 15)
    gawll_x*     o subconjunto que o PyGAwLLfs selecionou em cada uma das 5 execuções
    aleatorio    k features sorteadas (N_SORTEIOS por k): a referência de acaso
    todas        as 29 features (o modelo da etapa baselines): a referência de cima
Um método é fiel se o modelo com só as suas top-k chega perto do modelo com
todas — e bem acima de k features sorteadas.

Isolamento do teste: este é um dos dois únicos lugares do projeto que
tocam o teste. Cada configuração é treinada no treino e avaliada na validação
e, por último, no teste. Nada é escolhido a partir do teste: todas as listas
de features vêm de etapas anteriores, calculadas sem ele.

Produz:
    results/tables/fidelidade.csv         uma linha por configuração × conjunto
    results/tables/fidelidade_resumo.csv  média por (modelo, método, k, conjunto),
                                          e a fração do desempenho de "todas"
    results/figures/fidelidade.{png,pdf}  curvas de acurácia no teste
Quem consome: a síntese e o texto (a fidelidade é um dos quatro critérios da pergunta de pesquisa).
"""

import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # só gera arquivo, sem abrir janela
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.graficos import (  # noqa: E402
    COR_GRADE,
    COR_TEXTO,
    COR_TEXTO_SECUNDARIO,
    LARGURA_CHEIA,
    ROTULO_METODO,
    ROTULO_MODELO,
    salvar,
)
from src.avaliacao.rankings import MODELOS, ordenar  # noqa: E402
from src.modelagem import comum  # noqa: E402
from src.modelagem.baselines import (  # noqa: E402
    avaliar_modelo,
    carregar,
    criar_modelo,
    ler_configuracoes_escolhidas,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DIR_FIGURAS = RAIZ / "results" / "figures"
TOP_K = [5, 10, 15]
N_SORTEIOS = 10
SEMENTE = 42
METRICAS = ["acuracia", "f1", "pr_auc", "roc_auc", "acuracia_balanceada"]
CONJUNTOS = ["val", "test"]  # o teste sempre por último

# Cor e marcador por método (cores das figuras da etapa rankings + o 4º slot da paleta).
# Marcador diferente e rótulo na ponta da linha: a cor não carrega a identidade sozinha.
ESTILO = {
    "intrinseca": ("#eda100", "s"),
    "permutation": ("#2a78d6", "o"),
    "shap": ("#eb6834", "^"),
    "gawll": ("#1baf7a", "D"),
}


def listas_de_features(modelo: str, features: list[str], ranking: pd.DataFrame,
                       selecao: pd.DataFrame) -> list[dict]:
    """Todas as configurações de um modelo, {method, k, seed, features}: as 29 features,
    o top-5/10/15 de cada método (ranking consolidado), o x* de cada execução do
    PyGAwLLfs e N_SORTEIOS sorteios aleatórios por k.

    Usada em: main; tests/test_fidelidade.py.
    Importância: define o experimento de fidelidade; os sorteios são a linha de base
    que diz se um ranking é melhor que o acaso.
    """
    configuracoes = [{"method": "todas", "k": len(features), "seed": 0, "features": features}]

    do_modelo = ranking[ranking["model"] == modelo]
    for metodo, grupo in do_modelo.groupby("method"):
        ordem = ordenar(grupo.set_index("feature")["pontuacao"])
        for k in TOP_K:
            configuracoes.append({"method": metodo, "k": k, "seed": 0, "features": ordem[:k]})

    for linha in selecao[selecao["model"] == modelo].itertuples():
        escolhidas = linha.features.split(";")
        configuracoes.append({"method": "gawll_x*", "k": len(escolhidas),
                              "seed": linha.seed, "features": escolhidas})

    rng = np.random.default_rng(SEMENTE)
    for k in TOP_K:
        for sorteio in range(N_SORTEIOS):
            configuracoes.append({"method": "aleatorio", "k": k, "seed": sorteio,
                                  "features": list(rng.choice(features, size=k, replace=False))})
    return configuracoes


def avaliar_configuracoes(modelo_nome: str, configuracoes: list[dict],
                          dados: dict) -> list[dict]:
    """Treina cada configuração no treino completo e avalia na validação e, por último,
    no teste. Conjuntos de features repetidos (ex.: dois métodos com o mesmo top-15)
    reaproveitam o resultado: mesmo treino, mesmos números.

    Usada em: main.
    Importância: é um dos dois únicos usos do teste; usa os hiperparâmetros da
    etapa baselines, então a diferença vem só das features.
    """
    x_treino, y_treino = dados["train"]
    parametros = ler_configuracoes_escolhidas()[modelo_nome]

    cache, linhas = {}, []
    for config in configuracoes:
        chave = frozenset(config["features"])
        if chave not in cache:
            colunas = [c for c in x_treino.columns if c in chave]  # ordem original
            modelo = criar_modelo(modelo_nome, parametros)
            inicio = time.perf_counter()
            modelo.fit(x_treino[colunas], y_treino)
            tempo = time.perf_counter() - inicio
            cache[chave] = {
                conjunto: avaliar_modelo(modelo, dados[conjunto][0][colunas], dados[conjunto][1])
                for conjunto in CONJUNTOS  # val e, por último, test
            }
            cache[chave]["tempo_treino_s"] = tempo
        for conjunto in CONJUNTOS:
            linhas.append({
                "model": modelo_nome, "method": config["method"], "k": config["k"],
                "seed": config["seed"], "conjunto": conjunto,
                **{m: cache[chave][conjunto][m] for m in METRICAS},
                "tempo_treino_s": cache[chave]["tempo_treino_s"],
                "features": ";".join(config["features"]),
            })
    logger.info(f"{modelo_nome}: {len(configuracoes)} configurações, {len(cache)} treinos")
    return linhas


def resumir(fidelidade: pd.DataFrame) -> pd.DataFrame:
    """Média por (modelo, método, k, conjunto), extremos dos sorteios e a fração do
    desempenho de "todas" (colunas *_frac_todas). Não confundir com rankings.resumir.

    Usada em: main; tests/test_fidelidade.py.
    Importância: a fração de "todas" é o número da fidelidade que entra no texto.
    """
    resumo = (fidelidade.groupby(["model", "method", "k", "conjunto"], as_index=False)
              .agg(**{m: (m, "mean") for m in METRICAS},
                   acuracia_min=("acuracia", "min"), acuracia_max=("acuracia", "max"),
                   f1_min=("f1", "min"), f1_max=("f1", "max"),
                   pr_auc_min=("pr_auc", "min"), pr_auc_max=("pr_auc", "max"),
                   n_execucoes=("seed", "nunique")))
    todas = (resumo[resumo["method"] == "todas"]
             .set_index(["model", "conjunto"])[["acuracia", "f1", "pr_auc"]])
    for metrica in ["acuracia", "f1", "pr_auc"]:
        referencia = resumo.set_index(["model", "conjunto"]).index.map(todas[metrica])
        resumo[f"{metrica}_frac_todas"] = resumo[metrica].to_numpy() / np.asarray(referencia)
    return resumo


def figura_fidelidade(resumo: pd.DataFrame, fidelidade: pd.DataFrame, caminho_base: Path) -> None:
    """Curvas no teste: acurácia × nº de features retidas, uma linha por método, com a
    faixa dos sorteios aleatórios, o modelo com todas as features e o acaso (0,5). As
    estrelas (x* do PyGAwLLfs) vêm da tabela completa: uma por execução.

    Usada em: main.
    Importância: é a figura do critério de fidelidade.
    """
    teste = resumo[resumo["conjunto"] == "test"]
    selecoes = fidelidade[(fidelidade["conjunto"] == "test") & (fidelidade["method"] == "gawll_x*")]
    # squeeze=False: eixos é sempre 2D, mesmo com um modelo só
    fig, eixos = plt.subplots(1, len(MODELOS), figsize=(LARGURA_CHEIA, 2.9), layout="constrained",
                              sharex=True, sharey=True, squeeze=False)
    for coluna, modelo in enumerate(MODELOS):
        ax = eixos[0, coluna]
        do_modelo = teste[teste["model"] == modelo]
        todas = do_modelo[do_modelo["method"] == "todas"].iloc[0]

        # Acaso: faixa entre o pior e o melhor dos sorteios, e a média
        acaso = do_modelo[do_modelo["method"] == "aleatorio"].sort_values("k")
        ax.fill_between(acaso["k"], acaso["acuracia_min"], acaso["acuracia_max"],
                        color="#d9d8d3", alpha=0.6, linewidth=0)
        ax.plot(acaso["k"], acaso["acuracia"], color="#8c8b85", linestyle="--", linewidth=1.2,
                label=f"sorteio de k features (média e faixa de {N_SORTEIOS})")

        # Referências: todas as features em cima, o acaso (0,5) embaixo
        ax.axhline(todas["acuracia"], color=COR_TEXTO_SECUNDARIO, linewidth=1, linestyle=":",
                   label=f"todas as {int(todas['k'])} features")
        ax.axhline(0.5, color=COR_TEXTO_SECUNDARIO, linewidth=1, linestyle="-.",
                   label="acaso (0,5)")

        for metodo, (cor, marcador) in ESTILO.items():
            curva = do_modelo[do_modelo["method"] == metodo].sort_values("k")
            if curva.empty:
                continue
            ax.plot(curva["k"], curva["acuracia"], color=cor, marker=marcador, linewidth=1.5,
                    markersize=5, markeredgecolor="white", markeredgewidth=0.8,
                    label=ROTULO_METODO[metodo])

        # x* do PyGAwLLfs: um ponto por execução, no seu nº de features
        selecao = selecoes[selecoes["model"] == modelo]
        ax.scatter(selecao["k"], selecao["acuracia"], marker="*", s=70,
                   color=ESTILO["gawll"][0], edgecolors=COR_TEXTO, linewidths=0.5,
                   zorder=4, label="PyGAwLLfs: subconjunto selecionado (x*)")

        ax.set_xticks(TOP_K)
        ax.set_xlim(3, 18.5)
        ax.grid(color=COR_GRADE, linewidth=0.6)
        ax.set_axisbelow(True)
        for lado in ["top", "right"]:
            ax.spines[lado].set_visible(False)
        ax.tick_params(colors=COR_TEXTO_SECUNDARIO)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
        if coluna == 0:
            ax.set_ylabel("acurácia no teste", color=COR_TEXTO_SECUNDARIO)
        ax.set_title(ROTULO_MODELO[modelo], loc="left", color=COR_TEXTO)
        ax.set_xlabel("features retidas (k)", color=COR_TEXTO_SECUNDARIO)

    alcas, rotulos = eixos[0, -1].get_legend_handles_labels()  # XGBoost tem os 4 métodos
    fig.legend(alcas, rotulos, loc="outside lower center", ncol=3, frameon=False)
    salvar(fig, caminho_base, legenda=(
        "Fidelidade por retenção: acurácia no teste (nov–dez/2025, avaliado uma única vez) do "
        "modelo treinado só com as k features mais importantes de cada método. Faixa cinza: "
        f"pior e melhor de {N_SORTEIOS} sorteios de k features; tracejado: a média dos sorteios. "
        "Pontilhado: modelo com todas as features; traço-ponto: acaso (0,5). Estrelas: os "
        "subconjuntos selecionados pelo PyGAwLLfs nas 5 execuções (seleções iguais se sobrepõem)."))


def main() -> None:
    """Lê os três conjuntos, o ranking consolidado e o x* do GA, avalia as
    configurações dos três modelos e salva fidelidade.csv, fidelidade_resumo.csv e a
    figura.

    Usada em: python src/avaliacao/fidelity.py; main.py, etapa "fidelidade";
    tests/test_fidelidade.py.
    Importância: mede se o top-k de cada método preserva o desempenho, o critério de
    fidelidade da pergunta de pesquisa.
    """
    dados = {nome: carregar(nome) for nome in ["train", "val", "test"]}
    features = list(dados["train"][0].columns)
    ranking = pd.read_csv(comum.DIR_TABELAS / "ranking_consolidado.csv")
    selecao = pd.read_csv(comum.DIR_TABELAS / "selecao_gawll.csv")
    logger.info(f"Treino {dados['train'][0].shape} | validação {dados['val'][0].shape} | "
                f"teste {dados['test'][0].shape}")

    linhas = []
    for modelo in MODELOS:
        configuracoes = listas_de_features(modelo, features, ranking, selecao)
        linhas.extend(avaliar_configuracoes(modelo, configuracoes, dados))

    fidelidade = pd.DataFrame(linhas)
    resumo = resumir(fidelidade)
    comum.salvar_tabela(fidelidade.round(5), "fidelidade.csv")
    comum.salvar_tabela(resumo.round(4), "fidelidade_resumo.csv")
    figura_fidelidade(resumo, fidelidade, DIR_FIGURAS / "fidelidade")

    with pd.option_context("display.width", 220, "display.max_rows", 200):
        teste = resumo[(resumo["conjunto"] == "test") & (resumo["method"] != "gawll_x*")]
        logger.info("Teste — acurácia (média):\n" + teste.pivot_table(
            index=["model", "method"], columns="k", values="acuracia").round(3).to_string())
        x_estrela = resumo[(resumo["conjunto"] == "test") & (resumo["method"] == "gawll_x*")]
        logger.info("Teste — subconjuntos do PyGAwLLfs:\n"
                    + x_estrela[["model", "k", "acuracia", "acuracia_frac_todas"]].round(3)
                    .to_string(index=False))


if __name__ == "__main__":
    main()
