"""Etapa 11 — figuras da comparação de rankings (Cap. 4).

Recebe as tabelas de rankings.py e stability.py e produz, em results/figures/:
    concordancia_intra_modelo.{png,pdf}  um painel por modelo, métodos × métodos;
                                         linha de cima Spearman, de baixo RBO
    concordancia_geral.{png,pdf}         os 11 rankings (método · modelo) entre si,
                                         Spearman, separados em blocos por modelo
    estabilidade_sementes.{png,pdf}      concordância entre sementes de cada
                                         (método, modelo): média e mínimo
Só lê CSVs; não toca nenhum conjunto de dados.

Cores (paleta validada): Spearman vai de -1 a 1, então usa a escala
divergente vermelho–cinza–azul, com cinza no zero; RBO vai de 0 a 1 e usa a
rampa sequencial azul. Os números ficam escritos nas células, então a
leitura não depende só da cor.
"""

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # só gera arquivo, sem abrir janela
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.compare.rankings import METODOS, MODELOS  # noqa: E402
from src.importance import comum  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DIR_FIGURAS = RAIZ / "results" / "figures"

ROTULO_METODO = {"intrinseca": "Intrínseca", "permutation": "Permutation",
                 "shap": "SHAP", "gawll": "PyGAwLLfs"}
ROTULO_MODELO = {"logreg": "LogReg", "random_forest": "Random Forest", "xgboost": "XGBoost"}

COR_TEXTO = "#0b0b0b"
COR_TEXTO_SECUNDARIO = "#52514e"
COR_GRADE = "#e1e0d9"
# Divergente: vermelho (discordância) – cinza neutro (zero) – azul (concordância)
ESCALA_SPEARMAN = LinearSegmentedColormap.from_list(
    "spearman", ["#e34948", "#f0efec", "#256abf"])
# Sequencial: rampa azul, do passo 100 ao 700
ESCALA_RBO = LinearSegmentedColormap.from_list(
    "rbo", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
# Uma cor por método na figura de estabilidade (slots 1–3 da paleta categórica)
COR_METODO = {"permutation": "#2a78d6", "shap": "#eb6834", "gawll": "#1baf7a"}


def decimal(valor: float) -> str:
    """0.94 -> '0,94' (vírgula decimal, como no texto do TCC)."""
    return f"{valor:.2f}".replace(".", ",")


def cor_do_texto(cor_fundo) -> str:
    """Texto branco em célula escura, tinta escura em célula clara."""
    r, g, b = to_rgb(cor_fundo)
    luminancia = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminancia < 0.5 else COR_TEXTO


def matriz_simetrica(concordancia: pd.DataFrame, chaves: list, metrica: str) -> np.ndarray:
    """Tabela longa de pares -> matriz quadrada na ordem de 'chaves' (diagonal = 1)."""
    posicao = {chave: i for i, chave in enumerate(chaves)}
    matriz = np.eye(len(chaves))
    for linha in concordancia.itertuples():
        a = posicao.get((linha.method_a, linha.model_a))
        b = posicao.get((linha.method_b, linha.model_b))
        if a is not None and b is not None:
            matriz[a, b] = matriz[b, a] = getattr(linha, metrica)
    return matriz


def desenhar_triangulo(ax, matriz, rotulos, escala, vmin, vmax, tamanho_fonte=9):
    """Só o triângulo de baixo, sem a diagonal (o resto repete ou vale 1)."""
    n = len(rotulos)
    mascara = np.triu(np.ones((n, n), dtype=bool))  # diagonal e acima
    valores = np.ma.masked_array(matriz, mascara)[1:, :-1]
    imagem = ax.pcolormesh(valores, cmap=escala, vmin=vmin, vmax=vmax,
                           edgecolors="white", linewidth=2)
    for i in range(n - 1):
        for j in range(n - 1):
            if not valores.mask[i, j]:
                fundo = escala((valores[i, j] - vmin) / (vmax - vmin))
                ax.text(j + 0.5, i + 0.5, decimal(valores[i, j]), ha="center", va="center",
                        fontsize=tamanho_fonte, color=cor_do_texto(fundo))
    ax.set_xticks(np.arange(n - 1) + 0.5, rotulos[:-1], rotation=30, ha="right")
    ax.set_yticks(np.arange(n - 1) + 0.5, rotulos[1:])
    ax.invert_yaxis()
    ax.tick_params(length=0, labelsize=8.5, colors=COR_TEXTO_SECUNDARIO)
    for lado in ax.spines.values():
        lado.set_visible(False)
    return imagem


def formatar_barra(barra, rotulo: str) -> None:
    """Barra de cor discreta, com vírgula decimal."""
    barra.set_label(rotulo, color=COR_TEXTO_SECUNDARIO, fontsize=9)
    barra.outline.set_visible(False)
    barra.ax.tick_params(labelsize=8, colors=COR_TEXTO_SECUNDARIO)
    barra.ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))


def figura_intra_modelo(concordancia: pd.DataFrame, caminho_base: Path) -> None:
    fig, eixos = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    for coluna, modelo in enumerate(MODELOS):
        metodos = [m for m in METODOS
                   if ((concordancia["method_a"] == m) & (concordancia["model_a"] == modelo)).any()
                   or ((concordancia["method_b"] == m) & (concordancia["model_b"] == modelo)).any()]
        chaves = [(m, modelo) for m in metodos]
        rotulos = [ROTULO_METODO[m] for m in metodos]
        for linha, (metrica, escala, vmin) in enumerate(
                [("spearman", ESCALA_SPEARMAN, -1), ("rbo", ESCALA_RBO, 0)]):
            ax = eixos[linha, coluna]
            imagem = desenhar_triangulo(ax, matriz_simetrica(concordancia, chaves, metrica),
                                        rotulos, escala, vmin, 1)
            ax.set_aspect("equal")  # células quadradas, do mesmo tamanho nos três painéis
            if linha == 0:
                ax.set_title(ROTULO_MODELO[modelo], color=COR_TEXTO, fontsize=11)
            if coluna == 2:
                barra = fig.colorbar(imagem, ax=eixos[linha, :], shrink=0.8)
                formatar_barra(barra, "Spearman" if metrica == "spearman" else "RBO (p = 0,9)")
    fig.suptitle("Concordância entre métodos de importância, dentro de cada modelo",
                 x=0.01, ha="left", color=COR_TEXTO, fontsize=12)
    fig.supxlabel("Rankings consolidados pela média das 5 sementes. Linha de cima: Spearman "
                  "(-1 a 1); de baixo: RBO, que pesa mais o topo do ranking.",
                  x=0.01, ha="left", color=COR_TEXTO_SECUNDARIO, fontsize=8.5)
    salvar(fig, caminho_base)


def figura_geral(concordancia: pd.DataFrame, caminho_base: Path) -> None:
    chaves, rotulos = [], []
    for modelo in MODELOS:
        for metodo in METODOS:
            presente = (((concordancia["method_a"] == metodo) & (concordancia["model_a"] == modelo))
                        | ((concordancia["method_b"] == metodo) & (concordancia["model_b"] == modelo)))
            if presente.any():
                chaves.append((metodo, modelo))
                rotulos.append(f"{ROTULO_METODO[metodo]} · {ROTULO_MODELO[modelo]}")

    fig, ax = plt.subplots(figsize=(10, 9), layout="constrained")
    imagem = desenhar_triangulo(ax, matriz_simetrica(concordancia, chaves, "spearman"),
                                rotulos, ESCALA_SPEARMAN, -1, 1, tamanho_fonte=8)
    # Linhas que separam os blocos de cada modelo, só dentro do triângulo.
    # A linha da matriz desenhada i é o ranking i+1, e a coluna j é o ranking j.
    n = len(chaves)
    fronteiras = np.cumsum([sum(1 for c in chaves if c[1] == m) for m in MODELOS])[:-1]
    for f in fronteiras:
        ax.plot([0, f], [f - 1, f - 1], color=COR_TEXTO_SECUNDARIO, linewidth=1.2)
        ax.plot([f, f], [f, n - 1], color=COR_TEXTO_SECUNDARIO, linewidth=1.2)
    ax.set_aspect("equal")
    barra = fig.colorbar(imagem, ax=ax, shrink=0.6)
    formatar_barra(barra, "Spearman")
    ax.set_title("Concordância entre todos os rankings (método · modelo)",
                 loc="left", color=COR_TEXTO, fontsize=12)
    fig.supxlabel("Triângulos junto à diagonal: mesmo modelo, métodos diferentes "
                  "(comparação primária, §4.11). Demais blocos: modelos diferentes.",
                  x=0.01, ha="left", color=COR_TEXTO_SECUNDARIO, fontsize=8.5)
    salvar(fig, caminho_base)


def figura_estabilidade(estabilidade: pd.DataFrame, caminho_base: Path) -> None:
    """Pontos = média nas 10 duplas de sementes; traço até o pior caso (mínimo)."""
    linhas = [(metodo, modelo) for modelo in MODELOS for metodo in METODOS
              if ((estabilidade["method"] == metodo) & (estabilidade["model"] == modelo)).any()]
    y = np.arange(len(linhas))[::-1]

    fig, eixos = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True, layout="constrained")
    for ax, (metrica, titulo) in zip(eixos, [("spearman", "Spearman"), ("rbo", "RBO (p = 0,9)")]):
        for posicao, (metodo, modelo) in zip(y, linhas):
            linha = estabilidade[(estabilidade["method"] == metodo)
                                 & (estabilidade["model"] == modelo)].iloc[0]
            cor = COR_METODO[metodo]
            ax.plot([linha[f"{metrica}_min"], linha[metrica]], [posicao, posicao],
                    color=cor, linewidth=2, solid_capstyle="round")
            ax.scatter(linha[metrica], posicao, s=60, color=cor, zorder=3,
                       edgecolors="white", linewidths=2)
            ax.text(linha[f"{metrica}_min"] - 0.015, posicao, decimal(linha[f"{metrica}_min"]),
                    ha="right", va="center", fontsize=7.5, color=COR_TEXTO_SECUNDARIO)
        ax.set_xlim(0.3, 1.02)
        ax.set_title(titulo, loc="left", color=COR_TEXTO, fontsize=10)
        ax.grid(axis="x", color=COR_GRADE, linewidth=0.8)
        ax.set_axisbelow(True)
        for lado in ["top", "right", "left"]:
            ax.spines[lado].set_visible(False)
        ax.spines["bottom"].set_color(COR_TEXTO_SECUNDARIO)
        ax.tick_params(length=0, labelsize=8.5, colors=COR_TEXTO_SECUNDARIO)
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
    eixos[0].set_yticks(y, [f"{ROTULO_METODO[m]} · {ROTULO_MODELO[mo]}" for m, mo in linhas])

    # Legenda fora dos painéis, numa linha só: cor = método
    for metodo, cor in COR_METODO.items():
        eixos[1].scatter([], [], s=60, color=cor, label=ROTULO_METODO[metodo])
    fig.legend(frameon=False, fontsize=8.5, loc="outside upper right", ncol=3,
               labelcolor=COR_TEXTO)
    fig.suptitle("Estabilidade entre sementes: concordância média (ponto) e pior dupla (fim do traço)",
                 x=0.01, ha="left", color=COR_TEXTO, fontsize=11)
    fig.supxlabel("Permutation e SHAP variam só o embaralhamento/amostra sobre o mesmo modelo; "
                  "no PyGAwLLfs a semente muda a busca inteira.",
                  x=0.01, ha="left", color=COR_TEXTO_SECUNDARIO, fontsize=8.5)
    salvar(fig, caminho_base)


def salvar(fig, caminho_base: Path) -> None:
    caminho_base.parent.mkdir(parents=True, exist_ok=True)
    for extensao in ["png", "pdf"]:
        fig.savefig(caminho_base.with_suffix(f".{extensao}"), dpi=200,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Salvo: {caminho_base.name}.png e .pdf")


def main() -> None:
    concordancia = pd.read_csv(comum.DIR_TABELAS / "concordancia.csv")
    estabilidade = pd.read_csv(comum.DIR_TABELAS / "estabilidade.csv")
    figura_intra_modelo(concordancia, DIR_FIGURAS / "concordancia_intra_modelo")
    figura_geral(concordancia, DIR_FIGURAS / "concordancia_geral")
    figura_estabilidade(estabilidade, DIR_FIGURAS / "estabilidade_sementes")


if __name__ == "__main__":
    main()
