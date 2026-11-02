"""Grafo de interação reduzido do PyGAwLLfs, como no artigo (Guimarães et al., 2026).

Recebe interacao_gawll.csv (vint de cada execução, com a coluna "sondada") e
ranking_consolidado.csv (vimp médio normalizado, para o tamanho dos nós). Para
cada modelo, desenha as 5% arestas mais fortes (graph.arestas_reduzidas), na
média das sementes em que o par foi sondado, com a cor pelo tipo do par
(graph.tipo_por_construcao). Produz results/figures/grafo_interacao.{png,pdf}.
Só lê CSVs.
"""

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # só gera arquivo, sem abrir janela
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.avaliacao import graph  # noqa: E402
from src.avaliacao.graph import COR_TIPO  # noqa: E402
from src.avaliacao.rankings import MODELOS  # noqa: E402
from src.graficos import (  # noqa: E402
    COR_TEXTO,
    COR_TEXTO_SECUNDARIO,
    LARGURA_CHEIA,
    ROTULO_MODELO,
    salvar,
)
from src.modelagem import comum  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DIR_FIGURAS = RAIZ / "results" / "figures"


def afastar_nos(posicao: dict, distancia_minima: float = 0.15, iteracoes: int = 100) -> dict:
    """Empurra para longe os pares de nós mais próximos que distancia_minima (em fração
    do tamanho do desenho), para os rótulos não se sobreporem. O resto do layout
    continua como o kamada_kawai deixou.

    Usada em: figura_grafo.
    Importância: só apresentação.
    """
    nos = list(posicao)
    pontos = np.array([posicao[no] for no in nos], dtype=float)
    minimo = distancia_minima * np.ptp(pontos, axis=0).max()
    for _ in range(iteracoes):
        moveu = False
        for i in range(len(nos)):
            for j in range(i + 1, len(nos)):
                diferenca = pontos[i] - pontos[j]
                distancia = np.linalg.norm(diferenca)
                if distancia < minimo:
                    direcao = diferenca / distancia if distancia > 0 else np.array([1.0, 0.0])
                    passo = (minimo - distancia) / 2
                    pontos[i] += direcao * passo
                    pontos[j] -= direcao * passo
                    moveu = True
        if not moveu:
            break
    return dict(zip(nos, pontos))


def figura_grafo(arestas: pd.DataFrame, tipos: dict, importancia: pd.DataFrame,
                 caminho_base: Path) -> None:
    """Grafo reduzido de cada modelo (as arestas de graph.arestas_reduzidas, 5% das
    mais fortes): largura da aresta pela interação, cor pelo tipo do par, tamanho do
    nó pela importância.

    Usada em: main.
    Importância: é a figura análoga ao grafo do artigo do PyGAwLLfs.
    """
    fig, eixos = plt.subplots(3, 1, figsize=(LARGURA_CHEIA, 9.0), layout="constrained")
    for ax, modelo in zip(eixos, MODELOS):
        fortes = arestas[arestas["model"] == modelo]
        n_arestas = len(fortes)

        grafo = nx.Graph()
        for linha in fortes.itertuples():
            grafo.add_edge(linha.feature_a, linha.feature_b, peso=linha.interaction,
                           tipo=tipos[(linha.feature_a, linha.feature_b)])
        posicao = afastar_nos(nx.kamada_kawai_layout(grafo))

        pesos = np.array([d["peso"] for *_, d in grafo.edges(data=True)])
        larguras = 0.8 + 3.5 * (pesos - pesos.min()) / (np.ptp(pesos) or 1)
        cores = [COR_TIPO[d["tipo"]] for *_, d in grafo.edges(data=True)]
        nx.draw_networkx_edges(grafo, posicao, ax=ax, width=larguras, edge_color=cores, alpha=0.85)

        imp = (importancia[importancia["model"] == modelo]
               .set_index("feature")["importance_norm"])
        tamanhos = [60 + 400 * imp.get(no, 0) for no in grafo.nodes]
        nx.draw_networkx_nodes(grafo, posicao, ax=ax, node_size=tamanhos,
                               node_color="#f0efec", edgecolors=COR_TEXTO_SECUNDARIO,
                               linewidths=0.8)
        # Rótulos longos em duas linhas (quebra no último "_"), com fundo branco:
        # ficam legíveis mesmo com nós vizinhos ou uma aresta passando por trás
        rotulos = {no: (no[:no.rindex("_")] + "\n" + no[no.rindex("_") + 1:]
                        if len(no) > 14 and "_" in no else no) for no in grafo.nodes}
        nx.draw_networkx_labels(grafo, posicao, labels=rotulos, ax=ax, font_size=8,
                                font_color=COR_TEXTO,
                                bbox={"facecolor": "white", "edgecolor": "none",
                                      "alpha": 0.75, "pad": 0.5})
        ax.set_title(f"{ROTULO_MODELO[modelo]} ({n_arestas} arestas)", loc="left",
                     color=COR_TEXTO)
        ax.margins(0.12)  # espaço para os rótulos dos nós da borda
        ax.axis("off")

    legenda = [Line2D([0], [0], color=cor, linewidth=3, label=tipo)
               for tipo, cor in COR_TIPO.items()]
    fig.legend(handles=legenda, loc="outside lower center", ncol=3, frameon=False,
               title="tipo do par")
    salvar(fig, caminho_base, legenda=(
        "Grafo de interação reduzido do PyGAwLLfs: as 5% arestas mais fortes de cada modelo, na "
        "média das sementes em que o par foi sondado (só pares sondados em pelo menos 2 das 5 "
        "sementes). Espessura: força da interação (v_int); tamanho do nó: importância (v_imp, "
        "normalizada). Estrutural: as duas features codificam a mesma coluna do VRA; "
        "redundante: uma é função da outra pela definição dos atributos."))


def main() -> None:
    """Lê as interações do PyGAwLLfs e o ranking consolidado, monta as arestas do grafo
    reduzido com o tipo de cada par e salva a figura.

    Usada em: python src/interpretacao/figura_grafo.py; main.py, etapa "figuras".
    Importância: é a figura análoga ao grafo do artigo do PyGAwLLfs.
    """
    interacao = pd.read_csv(comum.DIR_TABELAS / "interacao_gawll.csv")
    arestas = graph.arestas_reduzidas(graph.interacao_media(interacao))
    tipos = {(a, b): graph.tipo_por_construcao(a, b)
             for a, b in zip(arestas["feature_a"], arestas["feature_b"])}
    importancia = pd.read_csv(comum.DIR_TABELAS / "ranking_consolidado.csv")
    importancia = importancia[importancia["method"] == "gawll"]
    figura_grafo(arestas, tipos, importancia, DIR_FIGURAS / "grafo_interacao")


if __name__ == "__main__":
    main()
