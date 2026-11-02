"""Figuras didáticas: o caminho até cada resultado (o dado, a operação e o número).

Cada figura confere os seus números com a tabela de origem, por assert:
    roc                curva ROC dos três modelos no teste; a AUC da legenda tem de
                       ser o roc_auc de baselines.csv
    convergencia_ga    melhor fitness por geração do PyGAwLLfs, 5 sementes por modelo,
                       com os resets marcados; o último ponto tem de ser o fitness de
                       selecao_gawll.csv
    ajuste_modelos     acurácia no treino, na validação e no teste de cada modelo, com o
                       trivial (0,5); os valores têm de ser os de baselines_tuning.csv e
                       baselines.csv
A ROC lê o teste: são os mesmos modelos e as mesmas previsões da avaliação final
(baselines.csv), sem escolher nada a partir dele. Produz results/figures/<nome>.{png,pdf}.
"""

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # só gera arquivo, sem abrir janela
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.graficos import (  # noqa: E402
    COR_GRADE,
    COR_TEXTO,
    COR_TEXTO_SECUNDARIO,
    LARGURA_CHEIA,
    ROTULO_MODELO,
    decimal,
    eixo_com_virgula,
    salvar,
)
from src.modelagem import comum  # noqa: E402
from src.modelagem.baselines import carregar  # noqa: E402
from src.modelagem.consolidar_gawll import DIR_EXECUCOES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DIR_FIGURAS = RAIZ / "results" / "figures"
TOLERANCIA = 1e-9
COR_MODELO = {"logreg": "#eda100", "random_forest": "#2a78d6", "xgboost": "#eb6834"}
COR_CONJUNTO = {"treino": "#9ec5f4", "validação": "#2a78d6", "teste": "#0d366b"}


def limpar_eixo(ax) -> None:
    """Grade discreta, sem as bordas de cima e da direita.

    Usada em: as figuras deste script.
    Importância: só apresentação.
    """
    ax.grid(color=COR_GRADE, linewidth=0.6)
    ax.set_axisbelow(True)
    for lado in ["top", "right"]:
        ax.spines[lado].set_visible(False)
    ax.tick_params(colors=COR_TEXTO_SECUNDARIO)


def figura_roc(baselines: pd.DataFrame, caminho_base: Path) -> None:
    """Curva ROC de cada modelo no teste, com a AUC na legenda (conferida com
    baselines.csv) e a diagonal do acaso.

    Usada em: main.
    Importância: mostra o que o ROC-AUC resume: a ordenação dos voos pelo risco,
    independente do corte em 0,5.
    """
    x_teste, y_teste = carregar("test")
    fig, ax = plt.subplots(figsize=(LARGURA_CHEIA * 0.6, 3.4), layout="constrained")
    ax.plot([0, 1], [0, 1], color=COR_TEXTO_SECUNDARIO, linewidth=0.8, linestyle=":",
            label="acaso (AUC = 0,5)")
    for nome in comum.MODELOS:
        score = comum.carregar_modelo(nome).predict_proba(x_teste)[:, 1]
        auc = roc_auc_score(y_teste, score)
        if abs(auc - baselines.loc[nome, "roc_auc"]) > TOLERANCIA:
            raise ValueError(f"AUC de {nome} ({auc}) difere de baselines.csv")
        fpr, tpr, _ = roc_curve(y_teste, score)
        ax.plot(fpr, tpr, color=COR_MODELO[nome], linewidth=1.5,
                label=f"{ROTULO_MODELO[nome]} (AUC = {decimal(auc, 3)})")
    ax.set_xlabel("taxa de falsos positivos", color=COR_TEXTO_SECUNDARIO)
    ax.set_ylabel("taxa de verdadeiros positivos", color=COR_TEXTO_SECUNDARIO)
    ax.set_aspect("equal")
    eixo_com_virgula(ax.xaxis)
    eixo_com_virgula(ax.yaxis)
    ax.legend(frameon=False, loc="lower right")
    limpar_eixo(ax)
    salvar(fig, caminho_base, legenda=(
        "Curvas ROC dos três modelos no teste (nov–dez/2025): para cada limiar de "
        "probabilidade, a fração de atrasos detectados contra a fração de voos pontuais "
        "marcados como atrasados. A AUC é a área sob a curva; a diagonal é o acaso."))


def figura_convergencia(selecao: pd.DataFrame, caminho_base: Path) -> None:
    """Melhor fitness por geração, um painel por modelo e uma linha por semente, com os
    resets marcados. Confere o último ponto com selecao_gawll.csv.

    Usada em: main.
    Importância: mostra como a busca evolui, quando estagna e o que o reset faz.
    """
    execucoes = [np.load(arquivo) for arquivo in sorted(DIR_EXECUCOES.glob("*_seed*.npz"))]
    fitness_final = selecao.set_index(["model", "seed"])["fitness"]
    fig, eixos = plt.subplots(1, len(comum.MODELOS), figsize=(LARGURA_CHEIA, 2.6),
                              layout="constrained")
    for ax, nome in zip(eixos, comum.MODELOS):
        for execucao in (e for e in execucoes if str(e["modelo"]) == nome):
            historico, semente = execucao["historico_fitness"], int(execucao["semente"])
            if abs(historico[-1] - fitness_final[(nome, semente)]) > TOLERANCIA:
                raise ValueError(f"{nome} semente {semente}: histórico não termina no fitness salvo")
            ax.plot(np.arange(len(historico)), historico, color=COR_MODELO[nome], linewidth=1,
                    alpha=0.8)
            resets = execucao["geracoes_reset"]
            ax.scatter(resets, historico[resets], marker="v", s=14, color=COR_TEXTO, zorder=3)
        ax.set_title(ROTULO_MODELO[nome], loc="left", color=COR_TEXTO)
        ax.set_xlabel("geração", color=COR_TEXTO_SECUNDARIO)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.3f}".replace(".", ",")))
        limpar_eixo(ax)
    eixos[0].set_ylabel("melhor fitness", color=COR_TEXTO_SECUNDARIO)
    salvar(fig, caminho_base, legenda=(
        "Convergência do PyGAwLLfs: melhor fitness (0,98 · acurácia na validação + 0,02 · "
        "parcimônia) a cada geração, uma linha por semente. Triângulos: gerações com reset "
        "(sem melhora por 50 gerações, a população é perturbada a partir do melhor "
        "indivíduo). Com o elitismo corrigido, o melhor fitness nunca cai."))


def acuracias_por_conjunto(tuning: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    """Acurácia no treino, na validação e no teste da configuração escolhida de cada
    modelo (inclusive o trivial), lida de baselines_tuning.csv e baselines.csv.

    Usada em: figura_ajuste.
    Importância: é o diagnóstico de ajuste: aprende acima do trivial? decorou o treino?
    """
    escolhidas = tuning.merge(baselines[["modelo", "parametros", "acuracia"]],
                              on=["modelo", "parametros"], suffixes=("", "_teste"))
    if len(escolhidas) != len(baselines):
        raise ValueError("Configuração escolhida ausente de baselines_tuning.csv")
    return escolhidas.set_index("modelo")[["acuracia_treino", "acuracia", "acuracia_teste"]] \
        .rename(columns={"acuracia_treino": "treino", "acuracia": "validação",
                         "acuracia_teste": "teste"})


def figura_ajuste(tuning: pd.DataFrame, baselines: pd.DataFrame, caminho_base: Path) -> None:
    """Barras agrupadas: acurácia no treino, na validação e no teste de cada modelo, com a
    linha do trivial (0,5).

    Usada em: main.
    Importância: mostra que os modelos aprendem acima do trivial e quanto perdem do
    treino para a validação (sobreajuste) e da validação para o teste (mudança de
    período).
    """
    acuracias = acuracias_por_conjunto(tuning, baselines).loc[comum.MODELOS]
    fig, ax = plt.subplots(figsize=(LARGURA_CHEIA * 0.7, 2.6), layout="constrained")
    x = np.arange(len(acuracias))
    largura = 0.26
    for i, conjunto in enumerate(acuracias.columns):
        barras = ax.bar(x + (i - 1) * largura, acuracias[conjunto], width=largura,
                        color=COR_CONJUNTO[conjunto], label=conjunto)
        ax.bar_label(barras, labels=[decimal(v, 3) for v in acuracias[conjunto]],
                     fontsize=8, padding=2, rotation=90, color=COR_TEXTO_SECUNDARIO)
    ax.axhline(0.5, color=COR_TEXTO, linewidth=0.8, linestyle="--", label="trivial (0,5)")
    ax.set_xticks(x, [ROTULO_MODELO[m] for m in acuracias.index])
    ax.set_ylim(0.45, max(0.8, acuracias.to_numpy().max() + 0.06))
    ax.set_ylabel("acurácia", color=COR_TEXTO_SECUNDARIO)
    eixo_com_virgula(ax.yaxis)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    limpar_eixo(ax)
    salvar(fig, caminho_base, legenda=(
        "Acurácia da configuração escolhida de cada modelo no treino (jan/2024 a ago/2025), na "
        "validação (set–out/2025) e no teste (nov–dez/2025), todos com 50% de atrasados. A "
        "linha tracejada é o classificador trivial."))


def main() -> None:
    """Gera a ROC, a convergência do GA e o diagnóstico de ajuste.

    Usada em: python src/interpretacao/figuras_didaticas.py; main.py, etapa "figuras".
    Importância: cada figura mostra de onde vem um número do texto.
    """
    baselines = pd.read_csv(comum.DIR_TABELAS / "baselines.csv")
    figura_roc(baselines.set_index("modelo"), DIR_FIGURAS / "roc")
    figura_convergencia(pd.read_csv(comum.DIR_TABELAS / "selecao_gawll.csv"),
                        DIR_FIGURAS / "convergencia_ga")
    figura_ajuste(pd.read_csv(comum.DIR_TABELAS / "baselines_tuning.csv"), baselines,
                  DIR_FIGURAS / "ajuste_modelos")


if __name__ == "__main__":
    main()
