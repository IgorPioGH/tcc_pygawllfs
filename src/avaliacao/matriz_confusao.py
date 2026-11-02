"""Etapa 6b — matrizes de confusão dos modelos de referência, no teste.

Recebe os modelos salvos pela Etapa 6 (experiments/runs/baselines/<modelo>.joblib),
o conjunto de teste congelado (data/processed/test.parquet) e o baselines.csv.
Não treina nem escolhe nada: são as mesmas previsões (corte em 0,5) que geraram
o baselines.csv, vistas como contagens de acertos e erros por classe.

Isolamento do teste (§4.9): não é um novo uso do teste. É parte da avaliação
final da Etapa 6, com o mesmo modelo e as mesmas previsões; para garantir isso,
acurácia, precisão, recall e F1 recalculados a partir das quatro células têm de
bater com o baselines.csv, senão o script para.

Produz:
    results/tables/matriz_confusao.csv          vn, fp, fn, vp de cada modelo
    results/figures/matriz_confusao.{png,pdf}   as três matrizes lado a lado, com as
                                                quatro métricas calculadas das células
Quem consome: a seção de modelos do texto do TCC.

Roda em segundos, sem retreinar: python src/avaliacao/matriz_confusao.py (depois
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

from src.graficos import (  # noqa: E402
    COR_TEXTO,
    COR_TEXTO_SECUNDARIO,
    ESCALA_RBO,
    LARGURA_CHEIA,
    ROTULO_MODELO,
    cor_do_texto,
    decimal,
    porcentagem,
    salvar,
)
from src.modelagem import comum  # noqa: E402
from src.modelagem.baselines import carregar  # noqa: E402

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


def metricas_da_matriz(matriz: np.ndarray) -> dict[str, float]:
    """Acurácia, precisão, recall e F1 calculados das quatro células [[vn, fp], [fn, vp]].

    Usada em: conferir_com_baselines e figura_matrizes.
    Importância: mostra de onde vem cada número; são as fórmulas do texto.
    """
    (vn, fp), (fn, vp) = matriz
    precisao = vp / (vp + fp) if vp + fp else 0.0
    recall = vp / (vp + fn)
    return {
        "acuracia": (vp + vn) / matriz.sum(),
        "precisao": precisao,
        "recall": recall,
        "f1": 2 * precisao * recall / (precisao + recall) if precisao + recall else 0.0,
    }


def conferir_com_baselines(nome: str, matriz: np.ndarray, baselines: pd.DataFrame) -> None:
    """Recalcula acurácia, precisão, recall e F1 a partir da matriz e compara com
    baselines.csv; levanta ValueError se alguma divergir.

    Usada em: main.
    Importância: prova que a matriz é a da avaliação final da Etapa 6 (mesmo
    modelo, mesmas previsões), e não uma reavaliação do teste (§4.9).
    """
    recalculadas = metricas_da_matriz(matriz)
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


def figura_matrizes(matrizes: dict[str, np.ndarray], caminho_base: Path) -> None:
    """As três matrizes lado a lado, com acurácia, precisão, recall e F1 calculados das
    quatro células escritos embaixo. A cor é a fração da linha (classe real); cada
    célula traz a contagem e essa fração.

    Usada em: main.
    Importância: é a figura da seção de desempenho; mostra onde cada modelo erra e como
    as métricas saem da matriz.
    """
    fig, eixos = plt.subplots(1, len(matrizes), figsize=(LARGURA_CHEIA, 2.9), layout="constrained")
    for ax, (nome, matriz) in zip(eixos, matrizes.items()):
        fracao = matriz / matriz.sum(axis=1, keepdims=True)
        ax.pcolormesh(fracao, cmap=ESCALA_RBO, vmin=0, vmax=1, edgecolors="white", linewidth=2)
        for i in range(2):
            for j in range(2):
                ax.text(j + 0.5, i + 0.5,
                        f"{milhar(matriz[i, j])}\n({porcentagem(fracao[i, j])})",
                        ha="center", va="center", fontsize=8,
                        color=cor_do_texto(ESCALA_RBO(fracao[i, j])))
        ax.set_xticks([0.5, 1.5], CLASSES)
        ax.set_yticks([0.5, 1.5], CLASSES)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xlabel("previsto", color=COR_TEXTO_SECUNDARIO)
        if ax is eixos[0]:
            ax.set_ylabel("real", color=COR_TEXTO_SECUNDARIO)
        ax.tick_params(length=0, colors=COR_TEXTO_SECUNDARIO)
        for lado in ax.spines.values():
            lado.set_visible(False)
        ax.set_title(ROTULO_MODELO[nome], color=COR_TEXTO)
        metricas = metricas_da_matriz(matriz)
        ax.text(0.5, -0.42, f"acurácia {decimal(metricas['acuracia'], 3)}   "
                            f"F1 {decimal(metricas['f1'], 3)}\n"
                            f"precisão {decimal(metricas['precisao'], 3)}   "
                            f"recall {decimal(metricas['recall'], 3)}",
                transform=ax.transAxes, ha="center", va="top", color=COR_TEXTO)
    total = next(iter(matrizes.values())).sum()
    salvar(fig, caminho_base, legenda=(
        f"Matrizes de confusão dos três modelos no teste (nov–dez/2025, {milhar(total)} voos, "
        "50% atrasados), com o corte em 0,5. Cor e porcentagem: fração da classe real (linha). "
        "Embaixo, as métricas calculadas das quatro células: acurácia = (VP + VN)/total, "
        "precisão = VP/(VP + FP), recall = VP/(VP + FN) e F1 = média harmônica das duas."))


def main() -> None:
    """Etapa 6b: matriz de confusão de cada modelo da Etapa 6 no teste, conferida
    contra baselines.csv; salva matriz_confusao.csv e a figura.

    Usada em: python src/avaliacao/matriz_confusao.py; main.py, etapa "desempenho".
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
    figura_matrizes(matrizes, DIR_FIGURAS / "matriz_confusao")


if __name__ == "__main__":
    main()
