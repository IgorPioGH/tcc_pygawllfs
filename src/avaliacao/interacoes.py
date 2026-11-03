"""Avaliação — PyGAwLLfs × H de Friedman: os dois concordam sobre quais pares interagem?

Recebe interacao_gawll.csv (vint, 5 execuções) e interacao_h.csv (H, 5
amostras), e classifica cada par pela definição dos atributos
(graph.tipo_por_construcao: estrutural, redundante ou operacional).
Para o RF e o XGBoost (a LogReg tem H = 0 por construção):

1. Média das sementes de cada método, nos 406 pares. No PyGAwLLfs, a média usa
   só as sementes em que o par foi sondado (graph.interacao_media, decisão 15).
2. Concordância entre os rankings de pares: PyGAwLLfs × H_abs (a comparação
   principal), PyGAwLLfs × H normalizado e H_abs × H normalizado. Spearman e
   Kendall tau-b em todos os pares com média; RBO e top-5/10/15 só nos pares
   ranqueados (no PyGAwLLfs, sondados em pelo menos 2 sementes).
3. Estabilidade entre sementes de cada método, nos mesmos pares (no
   PyGAwLLfs, cada dupla de sementes compara os pares sondados nas duas).
4. Hipótese da redundância (docs/05-avaliacao.md): onde os pares
   redundantes caem em cada ranking — rank médio (em percentil) e quantos
   entram no top-10.

Produz:
    results/tables/interacoes.csv        concordância e estabilidade, por modelo
    results/tables/interacoes_redundancia.csv  posição dos pares de cada tipo
    results/tables/interacoes_pares.csv  cada par: valor e rank em cada método, tipo e
                                         nº de sementes em que o PyGAwLLfs o sondou
    results/figures/interacao_gawll_x_h.{png,pdf}
Só lê CSVs.
"""

import logging
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # só gera arquivo, sem abrir janela
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.graficos import (  # noqa: E402
    COR_GRADE,
    COR_TEXTO,
    COR_TEXTO_SECUNDARIO,
    LARGURA_CHEIA,
    ROTULO_MODELO,
    salvar,
)
from src.avaliacao.rankings import comparar  # noqa: E402
from src.avaliacao.stability import concordancia_entre_sementes  # noqa: E402
from src.modelagem import comum  # noqa: E402
from src.avaliacao.graph import (  # noqa: E402
    COR_TIPO,
    interacao_media,
    marcar_nao_sondadas,
    tipo_por_construcao,
)
from src.modelagem.friedman_h import MODELOS_H  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DIR_FIGURAS = RAIZ / "results" / "figures"
METODOS = ["gawll", "friedman_h_abs", "friedman_h"]
ROTULO = {"gawll": "PyGAwLLfs", "friedman_h_abs": "H não normalizado",
          "friedman_h": "H normalizado"}
TOP_K_REDUNDANCIA = 10


def carregar_interacoes() -> pd.DataFrame:
    """Junta interacao_gawll.csv e interacao_h.csv, com a chave do par ("a|b", na mesma
    ordem nas duas) e a coluna "sondada" que cada produtor gravou.

    Usada em: main.
    Importância: a chave comum é o que permite comparar par a par o PyGAwLLfs e o H.
    """
    interacao = pd.concat([pd.read_csv(comum.DIR_TABELAS / "interacao_gawll.csv"),
                           pd.read_csv(comum.DIR_TABELAS / "interacao_h.csv")],
                          ignore_index=True)
    if interacao["sondada"].isna().any():
        raise ValueError("tabela de interação sem 'sondada' em alguma linha")
    interacao["sondada"] = interacao["sondada"].astype(bool)
    interacao["par"] = interacao["feature_a"] + "|" + interacao["feature_b"]
    return interacao


def tipos_dos_pares(pares: pd.DataFrame) -> dict[str, str]:
    """Tipo de cada par (estrutural, redundante ou operacional), via
    graph.tipo_por_construcao.

    Usada em: main.
    Importância: separa redundância de interação operacional, a hipótese central
    sobre o que o PyGAwLLfs detecta.
    """
    return {par: tipo_por_construcao(a, b)
            for a, b, par in zip(pares["feature_a"], pares["feature_b"], pares["par"])}


def comparar_modelo(interacao: pd.DataFrame, modelo: str) -> tuple[list[dict], pd.DataFrame]:
    """Para um modelo: concordância PyGAwLLfs × H nos pares em que o H existe, e
    estabilidade de cada método entre sementes.

    Usada em: main.
    Importância: produz o resultado principal da etapa interacoes (os dois métodos apontam as
    mesmas interações?).
    """
    do_modelo = interacao[interacao["model"] == modelo]
    pares_h = set(do_modelo.loc[do_modelo["method"] == "friedman_h_abs", "par"])
    do_modelo = do_modelo[do_modelo["par"].isin(pares_h)]

    media = interacao_media(do_modelo.drop(columns="par"))
    media["par"] = media["feature_a"] + "|" + media["feature_b"]
    por_metodo = {m: media[media["method"] == m].set_index("par") for m in METODOS}
    # Correlações em todos os pares com média; RBO e top-k só nos que têm rank
    valores = {m: tabela["interaction"] for m, tabela in por_metodo.items()}
    no_ranking = {m: tabela["interaction"].where(tabela["rank"].notna())
                  for m, tabela in por_metodo.items()}

    linhas = []
    for metodo_a, metodo_b in combinations(METODOS, 2):
        correlacoes = comparar(valores[metodo_a], valores[metodo_b])
        topos = comparar(no_ranking[metodo_a], no_ranking[metodo_b])
        linhas.append({
            "model": modelo, "comparacao": f"{metodo_a} x {metodo_b}",
            "n_pares": len(pares_h),
            "n_pares_correlacao": int((valores[metodo_a].notna()
                                       & valores[metodo_b].notna()).sum()),
            "spearman": correlacoes["spearman"], "kendall_tau_b": correlacoes["kendall_tau_b"],
            **{k: v for k, v in topos.items() if k == "rbo" or k.startswith("top")},
        })
    sementes = marcar_nao_sondadas(do_modelo)
    for metodo in METODOS:
        grupo = (sementes[sementes["method"] == metodo]
                 .rename(columns={"par": "feature", "interaction": "pontuacao"}))
        linhas.append({"model": modelo, "comparacao": f"estabilidade {metodo}",
                       "n_pares": len(pares_h),
                       **{k: v for k, v in concordancia_entre_sementes(grupo).items()
                          if not k.endswith("_min") and not k.startswith("n_")}})

    # Uma linha por par: valor e rank (1 = mais forte) em cada método, e em quantas
    # sementes o PyGAwLLfs sondou o par (no H, todo par é medido nas 5)
    por_par = media.pivot_table(index=["par", "feature_a", "feature_b"], columns="method",
                                values=["interaction", "rank", "n_sementes"])
    por_par.columns = [f"{coluna}_{metodo}" for coluna, metodo in por_par.columns]
    por_par = por_par.drop(columns=[f"n_sementes_{m}" for m in METODOS if m != "gawll"])
    return linhas, por_par.reset_index().assign(model=modelo)


def resumo_redundancia(pares: pd.DataFrame) -> pd.DataFrame:
    """Para cada (modelo, método, tipo de par): rank médio em percentil e quantos pares
    do tipo estão no top-10. Só contam os pares com rank naquele método.

    Usada em: main.
    Importância: testa nos dados reais se o PyGAwLLfs põe pares redundantes no topo,
    como no sintético.
    """
    linhas = []
    for modelo, grupo in pares.groupby("model"):
        for metodo in METODOS:
            n = grupo[f"rank_{metodo}"].notna().sum()
            for tipo, do_tipo in grupo.groupby("tipo"):
                ranks = do_tipo[f"rank_{metodo}"].dropna()
                linhas.append({
                    "model": modelo, "method": metodo, "tipo": tipo, "n_pares_tipo": len(ranks),
                    "rank_medio_percentil": float((ranks / n).mean() * 100),
                    f"no_top{TOP_K_REDUNDANCIA}": int((ranks <= TOP_K_REDUNDANCIA).sum()),
                })
    return pd.DataFrame(linhas)


def figura_gawll_x_h(pares: pd.DataFrame, caminho_base: Path) -> None:
    """Rank no PyGAwLLfs (x) × rank no H não normalizado (y), cor = tipo do par.

    Usada em: main.
    Importância: mostra a falta de concordância e onde ficam os pares redundantes.
    """
    fig, eixos = plt.subplots(1, len(MODELOS_H), figsize=(LARGURA_CHEIA, 3.4), layout="constrained")
    for ax, modelo in zip(eixos, MODELOS_H):
        grupo = pares[pares["model"] == modelo]
        n = len(grupo)
        grupo = grupo.dropna(subset=["rank_gawll", "rank_friedman_h_abs"])
        ax.plot([1, n], [1, n], color=COR_GRADE, linewidth=1, zorder=0)
        # Operacionais primeiro (fundo); redundantes e estruturais por cima, em destaque
        for tipo in ["operacional", "estrutural", "redundante"]:
            do_tipo = grupo[grupo["tipo"] == tipo]
            ax.scatter(do_tipo["rank_gawll"], do_tipo["rank_friedman_h_abs"],
                       s=10 if tipo == "operacional" else 30, color=COR_TIPO[tipo],
                       alpha=0.55 if tipo == "operacional" else 0.95,
                       edgecolors="white", linewidths=0.5, label=tipo, zorder=2)
        # Rótulo só nos redundantes do top-30 de algum dos dois (os que importam)
        redundantes = grupo[(grupo["tipo"] == "redundante")
                            & ((grupo["rank_gawll"] <= 30) | (grupo["rank_friedman_h_abs"] <= 30))]
        # Rótulos de pontos vizinhos (ex.: empates no H) descem um abaixo do outro
        colocados = []
        for linha in redundantes.sort_values(["rank_friedman_h_abs", "rank_gawll"]).itertuples():
            x, y = linha.rank_gawll, linha.rank_friedman_h_abs
            vizinhos = sum(abs(x - xc) < 0.25 * n and abs(y - yc) < 0.05 * n
                           for xc, yc in colocados)
            colocados.append((x, y))
            ax.annotate(f"{linha.feature_a} × {linha.feature_b}", (x, y),
                        xytext=(4, -3 - 10 * vizinhos), textcoords="offset points",
                        fontsize=8, color=COR_TEXTO_SECUNDARIO)
        ax.set_xlim(0, n + 5)
        ax.set_ylim(0, n + 5)
        ax.invert_xaxis()
        ax.invert_yaxis()  # rank 1 (mais forte) no canto superior direito
        ax.set_xlabel("rank no PyGAwLLfs", color=COR_TEXTO_SECUNDARIO)
        ax.set_ylabel("rank no H (não normalizado)", color=COR_TEXTO_SECUNDARIO)
        ax.set_title(f"{ROTULO_MODELO[modelo]} ({len(grupo)} de {n} pares)", loc="left",
                     color=COR_TEXTO)
        ax.grid(color=COR_GRADE, linewidth=0.6)
        ax.set_axisbelow(True)
        for lado in ["top", "right"]:
            ax.spines[lado].set_visible(False)
        ax.tick_params(colors=COR_TEXTO_SECUNDARIO)
    alcas, rotulos = eixos[-1].get_legend_handles_labels()
    fig.legend(alcas, rotulos, frameon=False, title="tipo do par", loc="outside lower center",
               ncol=3)
    salvar(fig, caminho_base, legenda=(
        "Pares de atributos: rank da interação segundo o PyGAwLLfs (média das sementes em que o "
        "par foi sondado; só pares sondados em pelo menos 2 das 5 sementes) e segundo o H de "
        "Friedman não normalizado (média das 5 amostras). Canto superior direito: forte nos "
        "dois; a diagonal é a concordância perfeita. Rotulados: pares redundantes no top-30 de "
        "algum dos dois métodos."))


def main() -> None:
    """Carrega as interações, compara por modelo, resume a redundância e
    salva interacoes*.csv e a figura.

    Usada em: python src/avaliacao/interacoes.py; main.py, etapa "interacoes";
    tests/test_interacao.py.
    Importância: responde se o PyGAwLLfs e o H de Friedman apontam as mesmas
    interações.
    """
    interacao = carregar_interacoes()

    linhas, tabelas_pares = [], []
    for modelo in MODELOS_H:
        linhas_modelo, por_par = comparar_modelo(interacao, modelo)
        linhas.extend(linhas_modelo)
        tabelas_pares.append(por_par)
    pares = pd.concat(tabelas_pares, ignore_index=True)
    tipos = tipos_dos_pares(pares.drop_duplicates("par"))
    pares["tipo"] = pares["par"].map(tipos)

    concordancia = pd.DataFrame(linhas)
    redundancia = resumo_redundancia(pares)
    comum.salvar_tabela(concordancia.round(4), "interacoes.csv")
    comum.salvar_tabela(redundancia.round(2), "interacoes_redundancia.csv")
    comum.salvar_tabela(pares.drop(columns="par").round(6), "interacoes_pares.csv")
    figura_gawll_x_h(pares, DIR_FIGURAS / "interacao_gawll_x_h")

    with pd.option_context("display.width", 220, "display.max_columns", 20):
        logger.info("Concordância e estabilidade:\n"
                    + concordancia[["model", "comparacao", "n_pares", "spearman", "rbo",
                                    "top5", "top10"]].round(3).to_string(index=False))
        logger.info("Onde caem os pares de cada tipo:\n" + redundancia.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
