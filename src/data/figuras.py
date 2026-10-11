"""Figuras descritivas dos dados (Cap. 3/4 do TCC).

Recebe data/interim/vra_target.parquet (voos realizados de 2024–2025, já com
atraso_min e a janela de plausibilidade [-40, 720] aplicada pelo target.py)
e desenha o histograma do atraso na partida, com o limiar de 15 min que
define a classe positiva. Produz results/figures/histograma_atraso.{png,pdf}.
Quem consome: o texto do TCC (figura de descrição da base).

O eixo vai de -40 a 180 min, onde está quase toda a massa; a cauda até
720 min é resumida numa anotação, para não achatar o resto do gráfico.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # só gera arquivo, sem abrir janela
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parents[2]
ARQUIVO_TARGET = RAIZ / "data" / "interim" / "vra_target.parquet"
DIR_FIGURAS = RAIZ / "results" / "figures"

LIMIAR_ATRASO = 15
EIXO_MIN, EIXO_MAX, LARGURA_BIN = -40, 180, 5

# Paleta: uma série só, em azul; texto em tinta neutra, nunca na cor da série
COR_SERIE = "#2a78d6"
COR_TEXTO = "#0b0b0b"
COR_TEXTO_SECUNDARIO = "#52514e"
COR_GRADE = "#e4e3df"


def porcentagem(fracao: float) -> str:
    """0.1847 -> '18,5%' (vírgula decimal, como no texto do TCC)."""
    return f"{fracao:.1%}".replace(".", ",")


def histograma_atraso(atraso_min: pd.Series, caminho_base: Path) -> None:
    """Desenha e salva o histograma em .png e .pdf (caminho_base sem extensão)."""
    total = len(atraso_min)
    fracao_atrasados = (atraso_min >= LIMIAR_ATRASO).mean()
    fracao_cauda = (atraso_min > EIXO_MAX).mean()

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    # Valores fora de [EIXO_MIN, EIXO_MAX] ficam de fora dos bins (são a cauda)
    bins = np.arange(EIXO_MIN, EIXO_MAX + LARGURA_BIN, LARGURA_BIN)
    contagens, _ = np.histogram(atraso_min, bins=bins)
    ax.bar(bins[:-1], contagens / 1000, width=LARGURA_BIN, align="edge",
           color=COR_SERIE, edgecolor="white", linewidth=0.6, zorder=2)

    # Limiar que define a classe positiva
    ax.axvline(LIMIAR_ATRASO, color=COR_TEXTO_SECUNDARIO, linestyle="--", linewidth=1.2, zorder=3)
    topo = ax.get_ylim()[1]
    ax.text(LIMIAR_ATRASO + 3, topo * 0.92,
            f"limiar de atraso: {LIMIAR_ATRASO} min\n"
            f"{porcentagem(fracao_atrasados)} dos voos à direita (classe positiva)",
            color=COR_TEXTO, fontsize=9, va="top")
    ax.text(EIXO_MAX, topo * 0.30,
            f"{porcentagem(fracao_cauda)} dos voos com\natraso entre {EIXO_MAX} e 720 min\n(fora do eixo)",
            color=COR_TEXTO_SECUNDARIO, fontsize=8.5, ha="right", va="top")

    ax.set_title("Distribuição do atraso na partida — VRA 2024–2025, voos realizados",
                 color=COR_TEXTO, fontsize=11, loc="left")
    ax.set_xlabel("Atraso na partida (min); negativo = partiu antes do previsto",
                  color=COR_TEXTO_SECUNDARIO, fontsize=9)
    ax.set_ylabel("Voos (milhares)", color=COR_TEXTO_SECUNDARIO, fontsize=9)
    ax.set_xlim(EIXO_MIN, EIXO_MAX)
    ax.set_xticks(np.arange(EIXO_MIN, EIXO_MAX + 1, 20))

    # Grade e eixos discretos: o destaque é dos dados
    ax.grid(axis="y", color=COR_GRADE, linewidth=0.8, zorder=0)
    for lado in ["top", "right", "left"]:
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(COR_TEXTO_SECUNDARIO)
    ax.tick_params(colors=COR_TEXTO_SECUNDARIO, labelsize=8.5, length=0)

    fig.text(0.01, 0.01, f"n = {total:,} voos; barras de {LARGURA_BIN} min.".replace(",", "."),
             color=COR_TEXTO_SECUNDARIO, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    caminho_base.parent.mkdir(parents=True, exist_ok=True)
    for extensao in ["png", "pdf"]:
        fig.savefig(caminho_base.with_suffix(f".{extensao}"), dpi=200, facecolor="white")
    plt.close(fig)
    logger.info(f"{total} voos | {fracao_atrasados:.4f} atrasados | "
                f"{fracao_cauda:.4f} acima de {EIXO_MAX} min")


def main() -> None:
    atraso = pd.read_parquet(ARQUIVO_TARGET, columns=["atraso_min"])["atraso_min"]
    caminho = DIR_FIGURAS / "histograma_atraso"
    histograma_atraso(atraso, caminho)
    logger.info(f"Salvo em {caminho}.png e .pdf")


if __name__ == "__main__":
    main()
