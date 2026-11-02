"""Figuras da comparação de rankings: concordância, dispersão e estabilidade.

Recebe as tabelas de rankings.py e stability.py e produz, em results/figures/:
    concordancia_intra_modelo.{png,pdf}  um painel por modelo: matriz de Spearman
                                         métodos × métodos (como a Fig. 1 do artigo
                                         do PyGAwLLfs)
    dispersao_gawll_x_rf.{png,pdf}       importância normalizada do PyGAwLLfs × a
                                         intrínseca do RF, por atributo (como a Fig. 2
                                         do artigo)
    estabilidade_sementes.{png,pdf}      Spearman entre sementes de cada (método,
                                         modelo): média e pior dupla
Só lê CSVs; não toca nenhum conjunto de dados.

Cores: Spearman vai de -1 a 1, então usa a escala divergente vermelho–cinza–azul,
com cinza no zero. Os números ficam escritos nas células, então a leitura não
depende só da cor.
"""

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # só gera arquivo, sem abrir janela
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.avaliacao.rankings import METODOS, MODELOS  # noqa: E402
from src.graficos import (  # noqa: E402
    COR_GRADE,
    COR_SERIE,
    COR_TEXTO,
    COR_TEXTO_SECUNDARIO,
    ESCALA_SPEARMAN,
    LARGURA_CHEIA,
    ROTULO_METODO,
    ROTULO_MODELO,
    cor_do_texto,
    decimal,
    eixo_com_virgula,
    salvar,
)
from src.modelagem import comum  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DIR_FIGURAS = RAIZ / "results" / "figures"
N_ROTULOS_DISPERSAO = 5  # os maiores de cada eixo ganham rótulo

# Uma cor por método na figura de estabilidade (slots 1–3 da paleta categórica)
COR_METODO = {"permutation": "#2a78d6", "shap": "#eb6834", "gawll": "#1baf7a"}


def matriz_simetrica(concordancia: pd.DataFrame, chaves: list, metrica: str) -> np.ndarray:
    """Tabela longa de pares -> matriz quadrada na ordem de "chaves" (diagonal = 1).

    Usada em: figura_intra_modelo.
    Importância: ponte entre o formato salvo (uma dupla por linha) e o heatmap.
    """
    posicao = {chave: i for i, chave in enumerate(chaves)}
    matriz = np.eye(len(chaves))
    for linha in concordancia.itertuples():
        a = posicao.get((linha.method_a, linha.model_a))
        b = posicao.get((linha.method_b, linha.model_b))
        if a is not None and b is not None:
            matriz[a, b] = matriz[b, a] = getattr(linha, metrica)
    return matriz


def desenhar_matriz(ax, matriz: np.ndarray, rotulos: list[str]) -> None:
    """Heatmap da matriz completa, com o valor escrito em cada célula.

    Usada em: figura_intra_modelo.
    Importância: mesma leitura da Fig. 1 do artigo do PyGAwLLfs.
    """
    n = len(rotulos)
    ax.pcolormesh(matriz, cmap=ESCALA_SPEARMAN, vmin=-1, vmax=1, edgecolors="white",
                  linewidth=1.5)
    for i in range(n):
        for j in range(n):
            fundo = ESCALA_SPEARMAN((matriz[i, j] + 1) / 2)
            ax.text(j + 0.5, i + 0.5, decimal(matriz[i, j]), ha="center", va="center",
                    fontsize=8, color=cor_do_texto(fundo))
    ax.set_xticks(np.arange(n) + 0.5, rotulos, rotation=45, ha="right")
    ax.set_yticks(np.arange(n) + 0.5, rotulos)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.tick_params(length=0, colors=COR_TEXTO_SECUNDARIO)
    for lado in ax.spines.values():
        lado.set_visible(False)


def figura_intra_modelo(concordancia: pd.DataFrame, caminho_base: Path) -> None:
    """Matriz de Spearman entre os métodos de importância, um painel por modelo.

    Usada em: main.
    Importância: é a figura da comparação intra-modelo, feita antes da comparação
    entre modelos (§4.11).
    """
    fig, eixos = plt.subplots(1, len(MODELOS), figsize=(LARGURA_CHEIA, 2.7), layout="constrained")
    anteriores = None
    for ax, modelo in zip(eixos, MODELOS):
        metodos = [m for m in METODOS
                   if ((concordancia["method_a"] == m) & (concordancia["model_a"] == modelo)).any()
                   or ((concordancia["method_b"] == m) & (concordancia["model_b"] == modelo)).any()]
        chaves = [(m, modelo) for m in metodos]
        desenhar_matriz(ax, matriz_simetrica(concordancia, chaves, "spearman"),
                        [ROTULO_METODO[m] for m in metodos])
        if metodos == anteriores:  # mesmos métodos do painel ao lado: sem repetir os rótulos
            ax.tick_params(labelleft=False)
        anteriores = metodos
        ax.set_title(ROTULO_MODELO[modelo], color=COR_TEXTO)
    salvar(fig, caminho_base, legenda=(
        "Concordância entre os métodos de importância dentro de cada modelo: correlação de "
        "Spearman entre os rankings, consolidados pela média das sementes (azul: concordância; "
        "vermelho: discordância)."))


def figura_dispersao(ranking: pd.DataFrame, caminho_base: Path) -> None:
    """Importância normalizada do PyGAwLLfs × importância intrínseca (MDI) do Random
    Forest, um ponto por atributo; os maiores de cada eixo ganham rótulo.

    Usada em: main.
    Importância: mostra, atributo a atributo, onde os dois métodos concordam e onde
    divergem, além do número único do Spearman.
    """
    do_rf = ranking[ranking["model"] == "random_forest"]
    eixos_xy = {metodo: do_rf[do_rf["method"] == metodo].set_index("feature")["importance_norm"]
                for metodo in ["intrinseca", "gawll"]}
    tabela = pd.DataFrame(eixos_xy).dropna()
    rotulados = set(tabela["intrinseca"].nlargest(N_ROTULOS_DISPERSAO).index) \
        | set(tabela["gawll"].nlargest(N_ROTULOS_DISPERSAO).index)

    fig, ax = plt.subplots(figsize=(LARGURA_CHEIA * 0.7, 3.6), layout="constrained")
    ax.plot([0, 1], [0, 1], color=COR_GRADE, linewidth=1, zorder=0)
    ax.scatter(tabela["intrinseca"], tabela["gawll"], s=22, color=COR_SERIE,
               edgecolors="white", linewidths=0.5, zorder=2)
    for feature in sorted(rotulados):
        ax.annotate(feature, (tabela.loc[feature, "intrinseca"], tabela.loc[feature, "gawll"]),
                    xytext=(4, 2), textcoords="offset points", fontsize=8,
                    color=COR_TEXTO_SECUNDARIO)
    ax.set_xlim(-0.03, 1.15)
    ax.set_ylim(-0.03, 1.08)
    eixo_com_virgula(ax.xaxis)
    eixo_com_virgula(ax.yaxis)
    ax.set_xlabel("intrínseca do Random Forest (MDI, normalizada)", color=COR_TEXTO_SECUNDARIO)
    ax.set_ylabel("PyGAwLLfs com Random Forest (v_imp, normalizada)", color=COR_TEXTO_SECUNDARIO)
    ax.grid(color=COR_GRADE, linewidth=0.6)
    ax.set_axisbelow(True)
    for lado in ["top", "right"]:
        ax.spines[lado].set_visible(False)
    ax.tick_params(colors=COR_TEXTO_SECUNDARIO)
    salvar(fig, caminho_base, legenda=(
        "Importância de cada atributo segundo o PyGAwLLfs (v_imp) e segundo a importância "
        "intrínseca do Random Forest (redução média de impureza), as duas normalizadas para "
        f"[0, 1]. Rotulados: os {N_ROTULOS_DISPERSAO} maiores de cada eixo; a diagonal é a "
        "concordância perfeita de magnitude."))


def figura_estabilidade(estabilidade: pd.DataFrame, caminho_base: Path) -> None:
    """Spearman entre sementes por (método, modelo): ponto = média nas 10 duplas de
    sementes; traço até o pior caso (mínimo).

    Usada em: main.
    Importância: é a figura do critério de estabilidade.
    """
    linhas = [(metodo, modelo) for modelo in MODELOS for metodo in METODOS
              if ((estabilidade["method"] == metodo) & (estabilidade["model"] == modelo)).any()]
    y = np.arange(len(linhas))[::-1]

    fig, ax = plt.subplots(figsize=(LARGURA_CHEIA, 2.8), layout="constrained")
    for posicao, (metodo, modelo) in zip(y, linhas):
        linha = estabilidade[(estabilidade["method"] == metodo)
                             & (estabilidade["model"] == modelo)].iloc[0]
        cor = COR_METODO[metodo]
        ax.plot([linha["spearman_min"], linha["spearman"]], [posicao, posicao],
                color=cor, linewidth=2, solid_capstyle="round")
        ax.scatter(linha["spearman"], posicao, s=40, color=cor, zorder=3,
                   edgecolors="white", linewidths=1.5)
        ax.text(linha["spearman_min"] - 0.015, posicao, decimal(linha["spearman_min"]),
                ha="right", va="center", fontsize=8, color=COR_TEXTO_SECUNDARIO)
    ax.set_xlim(min(0.3, estabilidade["spearman_min"].min() - 0.1), 1.02)
    ax.set_xlabel("Spearman entre sementes", color=COR_TEXTO_SECUNDARIO)
    ax.grid(axis="x", color=COR_GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ["top", "right", "left"]:
        ax.spines[lado].set_visible(False)
    ax.tick_params(length=0, colors=COR_TEXTO_SECUNDARIO)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
    ax.set_yticks(y, [f"{ROTULO_METODO[m]} · {ROTULO_MODELO[mo]}" for m, mo in linhas])
    salvar(fig, caminho_base, legenda=(
        "Estabilidade entre sementes: correlação de Spearman média entre os rankings das 10 "
        "duplas de sementes (ponto) e a pior dupla (fim do traço). Permutation e SHAP variam só "
        "o embaralhamento ou a amostra sobre o mesmo modelo; no PyGAwLLfs a semente muda a "
        "busca inteira."))


def main() -> None:
    """Lê concordancia.csv, ranking_consolidado.csv e estabilidade.csv e gera as três
    figuras.

    Usada em: python src/interpretacao/figuras_concordancia.py; main.py, etapa
    "figuras"; tests/test_comparacao.py.
    Importância: separa desenho de cálculo: as figuras só leem CSV.
    """
    concordancia = pd.read_csv(comum.DIR_TABELAS / "concordancia.csv")
    figura_intra_modelo(concordancia, DIR_FIGURAS / "concordancia_intra_modelo")
    figura_dispersao(pd.read_csv(comum.DIR_TABELAS / "ranking_consolidado.csv"),
                     DIR_FIGURAS / "dispersao_gawll_x_rf")
    figura_estabilidade(pd.read_csv(comum.DIR_TABELAS / "estabilidade.csv"),
                        DIR_FIGURAS / "estabilidade_sementes")


if __name__ == "__main__":
    main()
