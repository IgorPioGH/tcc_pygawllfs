"""Barras de importância: o valor bruto de cada método, antes de virar ranking.

Recebe results/tables/importancia_{intrinseca,permutation,shap,gawll}.csv, tira a
média das sementes e desenha uma figura por modelo, com um painel por método e
as 29 features em barras horizontais ordenadas, cada uma na unidade do seu
método (como no tutorial de importância de atributos do Machine Learning
Mastery):
    LogReg      coeficiente padronizado, com sinal (cor pelo sinal)
    RF          redução média de impureza (MDI)
    XGBoost     ganho médio por divisão (gain)
    Permutation queda média de acurácia na validação
    SHAP        média de |SHAP| na validação (log-odds)
    PyGAwLLfs   v_imp (soma das variações de fitness atribuídas à feature)
Confere que cada painel tem 29 barras e que os 3 maiores valores batem com o
ranking_consolidado.csv. Produz results/figures/importancia_barras_<modelo>.{png,pdf}.
Só lê CSVs.
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

from src.avaliacao.rankings import METODOS, MODELOS, pontuacao  # noqa: E402
from src.graficos import (  # noqa: E402
    COR_GRADE,
    COR_SERIE,
    COR_TEXTO,
    COR_TEXTO_SECUNDARIO,
    LARGURA_CHEIA,
    ROTULO_METODO,
    ROTULO_MODELO,
    eixo_com_virgula,
    salvar,
)
from src.modelagem import comum  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DIR_FIGURAS = RAIZ / "results" / "figures"
N_FEATURES = 29
COR_NEGATIVO = "#e34948"
UNIDADE = {
    ("intrinseca", "logreg"): "coeficiente padronizado",
    ("intrinseca", "random_forest"): "redução média de impureza (MDI)",
    ("intrinseca", "xgboost"): "ganho médio por divisão (gain)",
    "permutation": "queda média de acurácia",
    "shap": "média de |SHAP| (log-odds)",
    "gawll": "v_imp",
}


def importancia_media(importancias: pd.DataFrame) -> pd.DataFrame:
    """Média das sementes por (método, modelo, feature), com a pontuação que ordena o
    ranking (o valor absoluto no coeficiente da LogReg).

    Usada em: main.
    Importância: é o mesmo valor que rankings.consolidar_rankings usa.
    """
    media = (importancias.groupby(["method", "model", "feature"], as_index=False)["importance"]
             .mean())
    media["pontuacao"] = pontuacao(media)
    return media


def conferir_com_consolidado(media: pd.DataFrame, consolidado: pd.DataFrame) -> None:
    """Cada (método, modelo) tem 29 features e os 3 maiores valores batem com o
    ranking_consolidado.csv; levanta erro se não.

    Usada em: main.
    Importância: a figura mostra exatamente os números das tabelas.
    """
    for (metodo, modelo), grupo in media.groupby(["method", "model"]):
        if len(grupo) != N_FEATURES:
            raise ValueError(f"{metodo}/{modelo}: {len(grupo)} features no painel")
        referencia = consolidado[(consolidado["method"] == metodo) & (consolidado["model"] == modelo)]
        topo = grupo.nlargest(3, "pontuacao")["importance"].to_numpy()
        esperado = referencia.nlargest(3, "pontuacao")["importance"].to_numpy()
        if not np.allclose(topo, esperado):
            raise ValueError(f"{metodo}/{modelo}: top-3 {topo} difere do consolidado {esperado}")


def figura_modelo(media: pd.DataFrame, modelo: str, caminho_base: Path) -> None:
    """Um painel por método (duas colunas), barras ordenadas da mais para a menos
    importante; na LogReg intrínseca, a cor mostra o sinal do coeficiente.

    Usada em: main.
    Importância: mostra a escala e a forma de cada método, que o ranking esconde.
    """
    metodos = [m for m in METODOS if ((media["method"] == m) & (media["model"] == modelo)).any()]
    linhas = int(np.ceil(len(metodos) / 2))
    fig, eixos = plt.subplots(linhas, 2, figsize=(LARGURA_CHEIA, 4.3 * linhas),
                              layout="constrained", squeeze=False)
    for ax, metodo in zip(eixos.flat, metodos):
        grupo = (media[(media["method"] == metodo) & (media["model"] == modelo)]
                 .sort_values("pontuacao"))
        cores = [COR_NEGATIVO if valor < 0 else COR_SERIE for valor in grupo["importance"]]
        ax.barh(grupo["feature"], grupo["importance"], color=cores, height=0.7)
        ax.axvline(0, color=COR_TEXTO_SECUNDARIO, linewidth=0.6)
        ax.set_title(ROTULO_METODO[metodo], loc="left", color=COR_TEXTO)
        ax.set_xlabel(UNIDADE.get((metodo, modelo), UNIDADE.get(metodo)), color=COR_TEXTO_SECUNDARIO)
        ax.margins(y=0.01)
        eixo_com_virgula(ax.xaxis)
        ax.grid(axis="x", color=COR_GRADE, linewidth=0.6)
        ax.set_axisbelow(True)
        for lado in ["top", "right"]:
            ax.spines[lado].set_visible(False)
        ax.tick_params(colors=COR_TEXTO_SECUNDARIO)
    for ax in list(eixos.flat)[len(metodos):]:
        ax.axis("off")
    nota_sinal = (" Na intrínseca, a cor mostra o sinal do coeficiente (vermelho: negativo) e a "
                  "ordem segue o valor absoluto." if modelo == "logreg" else "")
    salvar(fig, caminho_base, legenda=(
        f"Importância de cada atributo no modelo {ROTULO_MODELO[modelo]}, na unidade de cada "
        "método e na média das sementes, do mais para o menos importante." + nota_sinal))


def main() -> None:
    """Lê as quatro tabelas de importância, confere com o consolidado e gera uma figura
    por modelo.

    Usada em: python src/interpretacao/figura_barras.py; main.py, etapa "figuras".
    Importância: é a primeira leitura dos resultados de importância, antes da
    comparação por rank.
    """
    importancias = pd.concat([pd.read_csv(comum.DIR_TABELAS / f"importancia_{m}.csv")
                              for m in METODOS], ignore_index=True)
    media = importancia_media(importancias)
    conferir_com_consolidado(media, pd.read_csv(comum.DIR_TABELAS / "ranking_consolidado.csv"))
    for modelo in MODELOS:
        figura_modelo(media, modelo, DIR_FIGURAS / f"importancia_barras_{modelo}")


if __name__ == "__main__":
    main()
