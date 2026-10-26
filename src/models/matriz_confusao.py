"""Etapa 6b — matrizes de confusão dos modelos de referência, no teste.

Recebe os modelos salvos pela Etapa 6 (experiments/runs/baselines/<modelo>.joblib),
o conjunto de teste congelado (data/processed/test.parquet) e o baselines.csv.
Não treina nem escolhe nada: são as mesmas previsões (corte em 0,5) que geraram
o baselines.csv, vistas como contagens de acertos e erros por classe.

Isolamento do teste (§4.9): não é um novo uso do teste. É parte da avaliação
final da Etapa 6, com o mesmo modelo e as mesmas previsões; para garantir isso,
precisão, recall e acurácia recalculadas a partir da matriz têm de bater com o
baselines.csv, senão o script para.

Produz:
    results/tables/matriz_confusao.csv          vn, fp, fn, vp de cada modelo
    results/figures/matriz_confusao.{png,pdf}   as três matrizes lado a lado
Quem consome: a seção de modelos do texto do TCC.

Roda em segundos, sem retreinar: python src/models/matriz_confusao.py (depois
da Etapa 6; no pipeline, logo depois de baselines.py).
"""

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # só gera arquivo, sem abrir janela
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.compare.figuras import (  # noqa: E402
    COR_TEXTO,
    COR_TEXTO_SECUNDARIO,
    ESCALA_RBO,
    ROTULO_MODELO,
    cor_do_texto,
    decimal,
    formatar_barra,
    salvar,
)
from src.data.figuras import porcentagem  # noqa: E402
from src.importance import comum  # noqa: E402
from src.models.baselines import carregar  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DIR_FIGURAS = RAIZ / "results" / "figures"
CLASSES = ["pontual", "atrasado"]  # 0 e 1, na ordem do confusion_matrix
TOLERANCIA = 1e-9


def matriz_do_modelo(nome: str, x: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Matriz 2×2 do modelo salvo na Etapa 6 no teste: linhas = classe real,
    colunas = classe prevista, [[vn, fp], [fn, vp]].

    Usada em: main.
    Importância: usa o mesmo modelo e o mesmo predict (corte em 0,5) de
    baselines.avaliar_modelo.
    """
    modelo = comum.carregar_modelo(nome)
    return confusion_matrix(y, modelo.predict(x), labels=[0, 1])


def conferir_com_baselines(nome: str, matriz: np.ndarray, baselines: pd.DataFrame) -> None:
    """Recalcula precisão, recall e acurácia a partir da matriz e compara com
    baselines.csv; levanta ValueError se alguma divergir.

    Usada em: main.
    Importância: prova que a matriz é a da avaliação final da Etapa 6 (mesmo
    modelo, mesmas previsões), e não uma reavaliação do teste (§4.9).
    """
    (vn, fp), (fn, vp) = matriz
    recalculadas = {
        "precisao": vp / (vp + fp) if vp + fp else 0.0,
        "recall": vp / (vp + fn),
        "acuracia": (vp + vn) / matriz.sum(),
    }
    divergentes = {
        metrica: (valor, baselines.loc[nome, metrica])
        for metrica, valor in recalculadas.items()
        if abs(valor - baselines.loc[nome, metrica]) > TOLERANCIA
    }
    if divergentes:
        raise ValueError(
            f"A matriz de '{nome}' não reproduz baselines.csv: {divergentes}. "
            "O .joblib não é o modelo avaliado na Etapa 6; rode a etapa de novo "
            "(python main.py --so baselines)."
        )


def milhar(valor: int) -> str:
    """31452 -> "31.452" (ponto de milhar, como no texto do TCC).

    Usada em: figura_matrizes.
    Importância: só apresentação.
    """
    return f"{valor:,}".replace(",", ".")


def figura_matrizes(matrizes: dict[str, np.ndarray], f1: dict[str, float],
                    caminho_base: Path) -> None:
    """As três matrizes lado a lado. A cor é a fração da linha (classe real), para
    os modelos ficarem comparáveis apesar do desbalanceamento; cada célula traz a
    contagem e essa fração.

    Usada em: main.
    Importância: é a figura da seção de modelos; mostra onde cada modelo erra,
    o que F1 e PR-AUC sozinhos não mostram.
    """
    fig, eixos = plt.subplots(1, len(matrizes), figsize=(11, 3.8), layout="constrained")
    for ax, (nome, matriz) in zip(eixos, matrizes.items()):
        fracao = matriz / matriz.sum(axis=1, keepdims=True)
        imagem = ax.pcolormesh(fracao, cmap=ESCALA_RBO, vmin=0, vmax=1,
                               edgecolors="white", linewidth=2)
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.5,
                        f"{milhar(matriz[i, j])}\n({porcentagem(fracao[i, j])})",
                        ha="center", va="center", fontsize=9.5,
                        color=cor_do_texto(ESCALA_RBO(fracao[i, j])))
        ax.set_xticks([0.5, 1.5], CLASSES)
        ax.set_yticks([0.5, 1.5], CLASSES)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xlabel("Previsto", color=COR_TEXTO_SECUNDARIO, fontsize=9)
        if ax is eixos[0]:
            ax.set_ylabel("Real", color=COR_TEXTO_SECUNDARIO, fontsize=9)
        ax.tick_params(length=0, labelsize=8.5, colors=COR_TEXTO_SECUNDARIO)
        for lado in ax.spines.values():
            lado.set_visible(False)
        ax.set_title(f"{ROTULO_MODELO[nome]} · F1 = {decimal(f1[nome])}",
                     color=COR_TEXTO, fontsize=10)

    barra = fig.colorbar(imagem, ax=eixos, shrink=0.8)
    formatar_barra(barra, "Fração da classe real")
    total = next(iter(matrizes.values())).sum()
    positivos = next(iter(matrizes.values()))[1].sum()
    fig.suptitle("Matrizes de confusão no teste (nov–dez/2025), corte em 0,5",
                 x=0.01, ha="left", color=COR_TEXTO, fontsize=11)
    fig.supxlabel(f"n = {milhar(total)} voos, {porcentagem(positivos / total)} "
                  "atrasados. Cor e % = fração da linha (classe real).",
                  x=0.01, ha="left", color=COR_TEXTO_SECUNDARIO, fontsize=8.5)
    salvar(fig, caminho_base)


def main() -> None:
    """Etapa 6b: matriz de confusão de cada modelo da Etapa 6 no teste, conferida
    contra baselines.csv; salva matriz_confusao.csv e a figura.

    Usada em: python src/models/matriz_confusao.py; main.py, etapa "baselines".
    Importância: complementa a avaliação final da Etapa 6 sem retreinar nada.
    """
    teste_x, teste_y = carregar("test")
    baselines = pd.read_csv(comum.DIR_TABELAS / "baselines.csv").set_index("modelo")

    matrizes, linhas = {}, []
    for nome in comum.MODELOS:
        matriz = matriz_do_modelo(nome, teste_x, teste_y)
        conferir_com_baselines(nome, matriz, baselines)
        matrizes[nome] = matriz
        (vn, fp), (fn, vp) = matriz
        linhas.append({"modelo": nome, "vn": vn, "fp": fp, "fn": fn, "vp": vp})
        logger.info(f"{nome}: vn={vn} fp={fp} fn={fn} vp={vp} (confere com baselines.csv)")

    comum.salvar_tabela(pd.DataFrame(linhas), "matriz_confusao.csv")
    figura_matrizes(matrizes, baselines["f1"].to_dict(), DIR_FIGURAS / "matriz_confusao")


if __name__ == "__main__":
    main()
