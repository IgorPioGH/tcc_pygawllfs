"""Etapa 12 — fidelidade por retenção: as features que cada método aponta bastam?

Recebe o ranking_consolidado.csv (Etapa 11), o selecao_gawll.csv (Etapa 9),
os conjuntos congelados train/val/test e os hiperparâmetros da Etapa 6.
Para cada modelo (mesma classe e hiperparâmetros da Etapa 6, treinado no
treino COMPLETO, §3.1), treina e avalia:
    top-k        as k features mais importantes de cada método (k = 5, 10, 15)
    gawll_x*     o subconjunto que o PyGAwLLfs selecionou em cada uma das 5 execuções
    aleatorio    k features sorteadas (N_SORTEIOS por k): a referência de acaso
    todas        as 29 features (o modelo da Etapa 6): a referência de cima
Um método é fiel se o modelo com só as suas top-k chega perto do modelo com
todas — e bem acima de k features sorteadas.

Isolamento do teste (§4.9): este é um dos dois únicos lugares do projeto que
tocam o teste. Cada configuração é treinada no treino e avaliada na validação
e, por último, no teste. Nada é escolhido a partir do teste: todas as listas
de features vêm de etapas anteriores, calculadas sem ele.

Produz:
    results/tables/fidelidade.csv         uma linha por configuração × conjunto
    results/tables/fidelidade_resumo.csv  média por (modelo, método, k, conjunto),
                                          e a fração do desempenho de "todas"
    results/figures/fidelidade.{png,pdf}  curvas no teste (F1 e PR-AUC)
Quem consome: o Cap. 4 (fidelidade é um dos quatro critérios da pergunta de pesquisa).
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

from src.compare.figuras import (  # noqa: E402
    COR_GRADE,
    COR_TEXTO,
    COR_TEXTO_SECUNDARIO,
    ROTULO_METODO,
    ROTULO_MODELO,
    salvar,
)
from src.compare.rankings import MODELOS, ordenar  # noqa: E402
from src.importance import comum  # noqa: E402
from src.models.baselines import (  # noqa: E402
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
METRICAS = ["f1", "pr_auc", "roc_auc", "acuracia_balanceada"]
CONJUNTOS = ["val", "test"]  # o teste sempre por último

# Cor e marcador por método (cores das figuras da Etapa 11 + o 4º slot da paleta).
# Marcador diferente e rótulo na ponta da linha: a cor não carrega a identidade sozinha.
ESTILO = {
    "intrinseca": ("#eda100", "s"),
    "permutation": ("#2a78d6", "o"),
    "shap": ("#eb6834", "^"),
    "gawll": ("#1baf7a", "D"),
}


def listas_de_features(modelo: str, features: list[str], ranking: pd.DataFrame,
                       selecao: pd.DataFrame) -> list[dict]:
    """Todas as configurações de um modelo: {method, k, seed, features}."""
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


def avaliar_configuracoes(modelo_nome: str, configuracoes: list[dict], dados: dict) -> list[dict]:
    """Treina cada configuração no treino e avalia na validação e no teste.
    Conjuntos de features repetidos (ex.: dois métodos com o mesmo top-15)
    reaproveitam o resultado: mesmo treino, mesmos números."""
    x_treino, y_treino = dados["train"]
    parametros = ler_configuracoes_escolhidas()[modelo_nome]
    peso_positivo = (y_treino == 0).sum() / (y_treino == 1).sum()

    cache, linhas = {}, []
    for config in configuracoes:
        chave = frozenset(config["features"])
        if chave not in cache:
            colunas = [c for c in x_treino.columns if c in chave]  # ordem original
            modelo = criar_modelo(modelo_nome, parametros, peso_positivo)
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
    """Média por (modelo, método, k, conjunto) e fração do desempenho de 'todas'."""
    resumo = (fidelidade.groupby(["model", "method", "k", "conjunto"], as_index=False)
              .agg(**{m: (m, "mean") for m in METRICAS},
                   f1_min=("f1", "min"), f1_max=("f1", "max"),
                   pr_auc_min=("pr_auc", "min"), pr_auc_max=("pr_auc", "max"),
                   n_execucoes=("seed", "nunique")))
    todas = (resumo[resumo["method"] == "todas"]
             .set_index(["model", "conjunto"])[["f1", "pr_auc"]])
    for metrica in ["f1", "pr_auc"]:
        referencia = resumo.set_index(["model", "conjunto"]).index.map(todas[metrica])
        resumo[f"{metrica}_frac_todas"] = resumo[metrica].to_numpy() / np.asarray(referencia)
    return resumo


def figura_fidelidade(resumo: pd.DataFrame, fidelidade: pd.DataFrame, caminho_base: Path) -> None:
    """Curvas no teste: desempenho × nº de features retidas, uma linha por método.
    As estrelas (x* do PyGAwLLfs) vêm da tabela completa: uma por execução."""
    teste = resumo[resumo["conjunto"] == "test"]
    selecoes = fidelidade[(fidelidade["conjunto"] == "test") & (fidelidade["method"] == "gawll_x*")]
    # squeeze=False: eixos é sempre 2D, mesmo com um modelo só
    fig, eixos = plt.subplots(2, len(MODELOS), figsize=(15, 8.4), layout="constrained",
                              sharex=True, squeeze=False)
    for coluna, modelo in enumerate(MODELOS):
        do_modelo = teste[teste["model"] == modelo]
        todas = do_modelo[do_modelo["method"] == "todas"].iloc[0]
        for linha_eixo, (metrica, rotulo) in enumerate([("f1", "F1"), ("pr_auc", "PR-AUC")]):
            ax = eixos[linha_eixo, coluna]

            # Acaso: faixa entre o pior e o melhor dos sorteios, e a média
            acaso = do_modelo[do_modelo["method"] == "aleatorio"].sort_values("k")
            ax.fill_between(acaso["k"], acaso[f"{metrica}_min"], acaso[f"{metrica}_max"],
                            color="#d9d8d3", alpha=0.6, linewidth=0)
            ax.plot(acaso["k"], acaso[metrica], color="#8c8b85", linestyle="--", linewidth=1.5)
            ax.annotate("sorteio", (acaso["k"].iloc[-1], acaso[metrica].iloc[-1]),
                        xytext=(4, -10), textcoords="offset points", fontsize=7.5,
                        color=COR_TEXTO_SECUNDARIO)

            # Todas as features: a referência de cima
            ax.axhline(todas[metrica], color=COR_TEXTO_SECUNDARIO, linewidth=1, linestyle=":")
            ax.annotate(f"todas as {int(todas['k'])}", (5, todas[metrica]), xytext=(0, 3),
                        textcoords="offset points", fontsize=7.5, color=COR_TEXTO_SECUNDARIO)

            for metodo, (cor, marcador) in ESTILO.items():
                curva = do_modelo[do_modelo["method"] == metodo].sort_values("k")
                if curva.empty:
                    continue
                # Identidade pela legenda + marcador próprio (sem rótulo na ponta:
                # as curvas convergem em k = 15 e os rótulos se sobreporiam)
                ax.plot(curva["k"], curva[metrica], color=cor, marker=marcador, linewidth=2,
                        markersize=7, markeredgecolor="white", markeredgewidth=1,
                        label=ROTULO_METODO[metodo])

            # x* do PyGAwLLfs: um ponto por execução, no seu nº de features
            selecao = selecoes[selecoes["model"] == modelo]
            ax.scatter(selecao["k"], selecao[metrica], marker="*", s=140,
                       color=ESTILO["gawll"][0], edgecolors=COR_TEXTO, linewidths=0.6,
                       zorder=4, label="PyGAwLLfs: subconjunto selecionado (x*)")

            ax.set_xticks(TOP_K + [15])
            ax.set_xlim(4, 18.5)
            ax.grid(color=COR_GRADE, linewidth=0.6)
            ax.set_axisbelow(True)
            for lado in ["top", "right"]:
                ax.spines[lado].set_visible(False)
            ax.tick_params(colors=COR_TEXTO_SECUNDARIO, labelsize=8)
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
            if coluna == 0:
                ax.set_ylabel(f"{rotulo} no teste", color=COR_TEXTO_SECUNDARIO, fontsize=9)
            if linha_eixo == 0:
                ax.set_title(ROTULO_MODELO[modelo], loc="left", color=COR_TEXTO, fontsize=11)
            else:
                ax.set_xlabel("nº de features retidas (k)", color=COR_TEXTO_SECUNDARIO, fontsize=9)

    alcas, rotulos = eixos[0, -1].get_legend_handles_labels()  # XGBoost tem os 4 métodos
    fig.legend(alcas, rotulos, loc="outside upper right", ncol=5, frameon=False, fontsize=8.5)
    fig.suptitle("Fidelidade por retenção: o modelo treinado só com as top-k features de cada método",
                 x=0.01, ha="left", color=COR_TEXTO, fontsize=12)
    fig.supxlabel("Teste (nov–dez/2025), avaliado uma única vez. Faixa cinza: pior e melhor de "
                  f"{N_SORTEIOS} sorteios de k features; tracejado: a média. Pontilhado: modelo com "
                  "todas as features. Estrelas: as 5 seleções do PyGAwLLfs (seleções iguais se sobrepõem).",
                  x=0.01, ha="left", color=COR_TEXTO_SECUNDARIO, fontsize=8.5)
    salvar(fig, caminho_base)


def main() -> None:
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
        logger.info("Teste — F1 e PR-AUC (média):\n" + teste.pivot_table(
            index=["model", "method"], columns="k", values=["f1", "pr_auc"]).round(3).to_string())
        x_estrela = resumo[(resumo["conjunto"] == "test") & (resumo["method"] == "gawll_x*")]
        logger.info("Teste — subconjuntos do PyGAwLLfs:\n"
                    + x_estrela[["model", "k", "f1", "pr_auc", "f1_frac_todas"]].round(3)
                    .to_string(index=False))


if __name__ == "__main__":
    main()
