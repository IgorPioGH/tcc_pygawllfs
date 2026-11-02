"""Kit comum das figuras do projeto: paleta, escalas de cor, rótulos, formatação com
vírgula decimal, estilo de impressão e salvar() em .png e .pdf.

Estilo de impressão: as figuras são feitas na largura do texto de uma página A4 com
margens ABNT (16 cm = LARGURA_CHEIA polegadas), com fonte de 8 a 9 pt, para
aparecerem no documento no mesmo tamanho do texto. O estilo é aplicado ao
importar este módulo. Título e nota não ficam dentro da figura: salvar() grava a
legenda em results/figures/legendas.md, que vira o "Figura N – ..." do texto.

Não importa nenhum outro módulo do projeto, então qualquer etapa pode usá-lo sem
criar import circular. Quem consome: todos os scripts que desenham figuras
(dados, preparação, avaliação e interpretação).
"""

import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # só gera arquivo, sem abrir janela
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402

logger = logging.getLogger(__name__)
# O PDF com fontes TrueType passa pelo fontTools, que registra cada passo em INFO
logging.getLogger("fontTools").setLevel(logging.WARNING)

LARGURA_CHEIA = 6.3   # polegadas: 16 cm, a largura do texto na página
LARGURA_MEIA = 3.1
ARQUIVO_LEGENDAS = "legendas.md"


def aplicar_estilo() -> None:
    """Fontes de 8 a 9 pt, 300 dpi e fontes TrueType no PDF (texto selecionável).

    Usada em: no import deste módulo.
    Importância: com as figuras na largura da página, o texto delas sai do tamanho do
    texto do documento.
    """
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "savefig.dpi": 300, "pdf.fonttype": 42,
    })


aplicar_estilo()

ROTULO_METODO = {"intrinseca": "Intrínseca", "permutation": "Permutation",
                 "shap": "SHAP", "gawll": "PyGAwLLfs"}
ROTULO_MODELO = {"logreg": "LogReg", "random_forest": "Random Forest", "xgboost": "XGBoost"}

COR_TEXTO = "#0b0b0b"
COR_TEXTO_SECUNDARIO = "#52514e"
COR_GRADE = "#e1e0d9"
COR_SERIE = "#2a78d6"  # figuras de uma série só
# Divergente: vermelho (discordância) – cinza neutro (zero) – azul (concordância)
ESCALA_SPEARMAN = LinearSegmentedColormap.from_list(
    "spearman", ["#e34948", "#f0efec", "#256abf"])
# Sequencial: rampa azul, do passo 100 ao 700
ESCALA_RBO = LinearSegmentedColormap.from_list(
    "rbo", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def decimal(valor: float, casas: int = 2) -> str:
    """0.94 -> "0,94" (vírgula decimal, como no texto do TCC).

    Usada em: as figuras e os logs que escrevem números.
    Importância: só apresentação.
    """
    return f"{valor:.{casas}f}".replace(".", ",")


def cor_do_texto(cor_fundo) -> str:
    """Texto branco em célula escura e tinta escura em célula clara, pela luminância da
    cor de fundo.

    Usada em: desenhar_triangulo.
    Importância: legibilidade dos números escritos nos heatmaps.
    """
    r, g, b = to_rgb(cor_fundo)
    luminancia = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminancia < 0.5 else COR_TEXTO


def eixo_com_virgula(eixo) -> None:
    """Marcas do eixo (ax.xaxis ou ax.yaxis) com vírgula decimal.

    Usada em: as figuras com eixos numéricos.
    Importância: só apresentação.
    """
    eixo.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))


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


def registrar_legenda(caminho_base: Path, legenda: str) -> None:
    """Grava a legenda da figura em legendas.md, na pasta da figura: uma seção
    "## nome" por figura, em ordem alfabética, substituindo a entrada anterior.

    Usada em: salvar.
    Importância: o título e a nota saem de dentro da figura e ficam num lugar só,
    prontos para o texto.
    """
    arquivo = caminho_base.parent / ARQUIVO_LEGENDAS
    entradas = {}
    if arquivo.exists():
        partes = re.split(r"^## ", arquivo.read_text(encoding="utf-8"), flags=re.MULTILINE)[1:]
        for parte in partes:
            nome, _, texto = parte.partition("\n")
            entradas[nome.strip()] = texto.strip()
    entradas[caminho_base.name] = " ".join(legenda.split())
    corpo = "\n\n".join(f"## {nome}\n\n{texto}" for nome, texto in sorted(entradas.items()))
    arquivo.write_text("# Legendas das figuras\n\n" + corpo + "\n", encoding="utf-8")


def salvar(fig, caminho_base: Path, legenda: str | None = None) -> None:
    """Salva em .png e .pdf (caminho_base sem extensão), grava a legenda (se houver) em
    legendas.md e fecha a figura. O PDF sai sem data de criação, então a mesma figura
    gera sempre o mesmo arquivo.

    Usada em: todas as figuras do projeto.
    Importância: rodar o pipeline de novo não altera arquivos versionados se os
    resultados não mudaram.
    """
    caminho_base.parent.mkdir(parents=True, exist_ok=True)
    for extensao in ["png", "pdf"]:
        fig.savefig(caminho_base.with_suffix(f".{extensao}"), facecolor="white",
                    bbox_inches="tight", metadata=METADADOS_PDF if extensao == "pdf" else None)
    plt.close(fig)
    if legenda:
        registrar_legenda(caminho_base, legenda)
    logger.info(f"Salvo: {caminho_base.name}.png e .pdf")
