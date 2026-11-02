"""Kit comum das figuras do projeto: paleta, escalas de cor, rótulos, formatação com
vírgula decimal e salvar() em .png e .pdf.

Não importa nenhum outro módulo do projeto, então qualquer etapa pode usá-lo sem
criar import circular. Quem consome: todos os scripts que desenham figuras
(dados, avaliação e interpretação).
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # só gera arquivo, sem abrir janela
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402

logger = logging.getLogger(__name__)

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


def decimal(valor: float) -> str:
    """0.94 -> "0,94" (vírgula decimal, como no texto do TCC).

    Usada em: desenhar_triangulo e figura_estabilidade.
    Importância: só apresentação.
    """
    return f"{valor:.2f}".replace(".", ",")


def cor_do_texto(cor_fundo) -> str:
    """Texto branco em célula escura e tinta escura em célula clara, pela luminância da
    cor de fundo.

    Usada em: desenhar_triangulo.
    Importância: legibilidade dos números escritos nos heatmaps.
    """
    r, g, b = to_rgb(cor_fundo)
    luminancia = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminancia < 0.5 else COR_TEXTO


def porcentagem(fracao: float) -> str:
    """0.1847 -> "18,5%" (vírgula decimal, como no texto do TCC).

    Usada em: histograma_atraso; matriz_confusao.
    Importância: só apresentação.
    """
    return f"{fracao:.1%}".replace(".", ",")


def formatar_barra(barra, rotulo: str) -> None:
    """Barra de cor discreta, com vírgula decimal.

    Usada em: figura_intra_modelo e figura_geral.
    Importância: só apresentação.
    """
    barra.set_label(rotulo, color=COR_TEXTO_SECUNDARIO, fontsize=9)
    barra.outline.set_visible(False)
    barra.ax.tick_params(labelsize=8, colors=COR_TEXTO_SECUNDARIO)
    barra.ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))


# O PDF guarda a data de criação; sem ela, a mesma figura gera o mesmo arquivo,
# e rodar o pipeline de novo não altera o que já está versionado no git
METADADOS_PDF = {"CreationDate": None}


def salvar(fig, caminho_base: Path) -> None:
    """Salva em .png e .pdf (caminho_base sem extensão) e fecha a figura. O PDF sai sem
    data de criação, então a mesma figura gera sempre o mesmo arquivo.

    Usada em: todas as figuras do projeto.
    Importância: rodar o pipeline de novo não altera arquivos versionados se os
    resultados não mudaram.
    """
    caminho_base.parent.mkdir(parents=True, exist_ok=True)
    for extensao in ["png", "pdf"]:
        fig.savefig(caminho_base.with_suffix(f".{extensao}"), dpi=200,
                    facecolor="white", bbox_inches="tight",
                    metadata=METADADOS_PDF if extensao == "pdf" else None)
    plt.close(fig)
    logger.info(f"Salvo: {caminho_base.name}.png e .pdf")
