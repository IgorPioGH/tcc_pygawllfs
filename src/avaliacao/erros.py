"""Incerteza simples e análise de erros no teste: acurácia por dia e por hora.

Recebe os modelos salvos (experiments/runs/baselines/), o test.parquet e o
baselines.csv. Não é um novo uso do teste: são os mesmos modelos e as mesmas
previsões (corte em 0,5) da avaliação final, e nada é escolhido a partir disso.

1. Acurácia por dia (61 dias, nov–dez/2025). Erro-padrão da acurácia de cada
   modelo: EP = desvio-padrão das acurácias diárias / √(nº de dias). Diferença
   entre dois modelos: média das diferenças diárias (pareada por dia) ± 2·EP
   dessa diferença. Regra de leitura: diferença menor que 2·EP = empate. Os dias
   vizinhos se parecem (o atraso de um dia carrega para o seguinte), então o EP
   é otimista: a incerteza real é um pouco maior.
2. Acurácia por hora prevista de partida.

Confere que a média das acurácias diárias, ponderada pelos voos de cada dia, é a
acurácia de baselines.csv. Produz:
    results/tables/erros_por_dia.csv       dia, modelo, voos, acurácia
    results/tables/erros_por_hora.csv      hora, modelo, voos, acurácia
    results/tables/incerteza_simples.csv   modelo, acurácia, EP, 2·EP
    results/tables/incerteza_pares.csv     diferença pareada entre modelos, EP e empate
    results/figures/erros_por_dia.{png,pdf} e erros_por_hora.{png,pdf}
"""

import logging
import sys
from itertools import combinations
from pathlib import Path

import holidays
import matplotlib

matplotlib.use("Agg")  # só gera arquivo, sem abrir janela
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.graficos import (  # noqa: E402
    COR_GRADE,
    COR_TEXTO_SECUNDARIO,
    LARGURA_CHEIA,
    ROTULO_MODELO,
    eixo_com_virgula,
    salvar,
)
from src.modelagem import comum  # noqa: E402
from src.modelagem.baselines import DIR_DADOS, carregar  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DIR_FIGURAS = RAIZ / "results" / "figures"
DIAS_NO_TESTE = 61
TOLERANCIA = 1e-9
COR_MODELO = {"logreg": "#eda100", "random_forest": "#2a78d6", "xgboost": "#eb6834"}


def acertos_no_teste() -> tuple[pd.DataFrame, pd.Series]:
    """Uma coluna de acerto (0/1) por modelo, uma linha por voo do teste, e a Partida
    Prevista de cada voo (lida direto do parquet e alinhada pelo índice, porque
    carregar() a descarta).

    Usada em: main.
    Importância: são exatamente as previsões que geraram baselines.csv.
    """
    x_teste, y_teste = carregar("test")
    partida = pd.read_parquet(DIR_DADOS / "test.parquet", columns=["Partida Prevista"])
    partida = partida["Partida Prevista"].loc[x_teste.index]
    acertos = pd.DataFrame({nome: (comum.carregar_modelo(nome).predict(x_teste) == y_teste)
                           .astype(int) for nome in comum.MODELOS}, index=x_teste.index)
    return acertos, partida


def acuracia_por_grupo(acertos: pd.DataFrame, chave: pd.Series, nome_chave: str) -> pd.DataFrame:
    """Formato longo: (chave, modelo, voos, acurácia).

    Usada em: main.
    Importância: o mesmo cálculo serve ao dia e à hora.
    """
    grupos = acertos.groupby(chave.rename(nome_chave))
    voos = grupos.size()
    return (grupos.mean().melt(ignore_index=False, var_name="model", value_name="acuracia")
            .join(voos.rename("voos")).reset_index())


def conferir_com_baselines(por_dia: pd.DataFrame, baselines: pd.DataFrame) -> None:
    """A média das acurácias diárias ponderada pelos voos é a acurácia de baselines.csv,
    e o teste tem DIAS_NO_TESTE dias; levanta erro se não.

    Usada em: main.
    Importância: prova que a análise usa as mesmas previsões da avaliação final.
    """
    for nome, grupo in por_dia.groupby("model"):
        if len(grupo) != DIAS_NO_TESTE:
            raise ValueError(f"{nome}: {len(grupo)} dias no teste, esperado {DIAS_NO_TESTE}")
        ponderada = np.average(grupo["acuracia"], weights=grupo["voos"])
        if abs(ponderada - baselines.loc[nome, "acuracia"]) > TOLERANCIA:
            raise ValueError(f"{nome}: acurácia diária ponderada {ponderada} ≠ baselines.csv")


def incerteza(por_dia: pd.DataFrame, baselines: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """EP da acurácia de cada modelo (desvio-padrão das acurácias diárias / √dias) e, para
    cada par de modelos, a diferença média diária (pareada), o EP dessa diferença e se
    ela fica abaixo de 2·EP (empate).

    Usada em: main.
    Importância: dá uma escala de ruído tirada dos próprios dados para ler as
    diferenças entre modelos e na fidelidade.
    """
    largo = por_dia.pivot(index="dia", columns="model", values="acuracia")
    n = len(largo)
    modelos = pd.DataFrame({
        "model": largo.columns,
        "acuracia": [baselines.loc[m, "acuracia"] for m in largo.columns],
        "ep": largo.std(ddof=1).to_numpy() / np.sqrt(n),
    })
    modelos["dois_ep"] = 2 * modelos["ep"]
    pares = []
    for a, b in combinations(comum.MODELOS, 2):
        diferenca = largo[a] - largo[b]
        ep = diferenca.std(ddof=1) / np.sqrt(n)
        pares.append({"comparacao": f"{a} - {b}", "diferenca_media": diferenca.mean(), "ep": ep,
                      "dois_ep": 2 * ep, "empate": abs(diferenca.mean()) < 2 * ep})
    return modelos, pd.DataFrame(pares)


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
    eixo_com_virgula(ax.yaxis)


def figura_por_dia(por_dia: pd.DataFrame, caminho_base: Path) -> None:
    """Acurácia diária de cada modelo no teste, com os feriados nacionais marcados.

    Usada em: main.
    Importância: mostra quanto a acurácia varia de um dia para o outro, que é a origem
    do erro-padrão.
    """
    fig, ax = plt.subplots(figsize=(LARGURA_CHEIA, 2.6), layout="constrained")
    dias = pd.to_datetime(por_dia["dia"].unique())
    for data in sorted(holidays.Brazil(years=sorted(set(dias.year)))):
        if dias.min() <= pd.Timestamp(data) <= dias.max():
            ax.axvline(pd.Timestamp(data), color=COR_GRADE, linewidth=4, zorder=0)
    for nome in comum.MODELOS:
        grupo = por_dia[por_dia["model"] == nome]
        ax.plot(pd.to_datetime(grupo["dia"]), grupo["acuracia"], color=COR_MODELO[nome],
                linewidth=1.2, label=ROTULO_MODELO[nome])
    ax.axhline(0.5, color=COR_TEXTO_SECUNDARIO, linewidth=0.8, linestyle="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.set_ylabel("acurácia no dia", color=COR_TEXTO_SECUNDARIO)
    ax.legend(frameon=False, ncol=3, loc="lower left")
    limpar_eixo(ax)
    salvar(fig, caminho_base, legenda=(
        "Acurácia de cada modelo em cada um dos 61 dias do teste (nov–dez/2025). As faixas "
        "cinza são feriados nacionais; a linha tracejada é o classificador trivial (0,5)."))


def figura_por_hora(por_hora: pd.DataFrame, caminho_base: Path) -> None:
    """Acurácia de cada modelo por hora prevista de partida, no teste.

    Usada em: main.
    Importância: mostra em que horas os modelos erram mais.
    """
    fig, ax = plt.subplots(figsize=(LARGURA_CHEIA, 2.4), layout="constrained")
    for nome in comum.MODELOS:
        grupo = por_hora[por_hora["model"] == nome]
        ax.plot(grupo["hora"], grupo["acuracia"], color=COR_MODELO[nome], marker="o",
                markersize=3, linewidth=1.2, label=ROTULO_MODELO[nome])
    ax.axhline(0.5, color=COR_TEXTO_SECUNDARIO, linewidth=0.8, linestyle="--")
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xlabel("hora prevista de partida", color=COR_TEXTO_SECUNDARIO)
    ax.set_ylabel("acurácia", color=COR_TEXTO_SECUNDARIO)
    ax.legend(frameon=False, ncol=3, loc="lower left")
    limpar_eixo(ax)
    salvar(fig, caminho_base, legenda=(
        "Acurácia de cada modelo no teste por hora prevista de partida; a linha tracejada é "
        "o classificador trivial (0,5)."))


def main() -> None:
    """Acurácia por dia e por hora no teste, erro-padrão por dias e as figuras.

    Usada em: python src/avaliacao/erros.py; main.py, etapa "desempenho".
    Importância: troca uma regra de empate arbitrária por uma medida tirada dos dados
    e mostra onde os modelos erram.
    """
    baselines = pd.read_csv(comum.DIR_TABELAS / "baselines.csv").set_index("modelo")
    acertos, partida = acertos_no_teste()
    por_dia = acuracia_por_grupo(acertos, partida.dt.normalize(), "dia")
    por_hora = acuracia_por_grupo(acertos, partida.dt.hour, "hora")
    conferir_com_baselines(por_dia, baselines)
    modelos, pares = incerteza(por_dia, baselines)

    comum.salvar_tabela(por_dia.round({"acuracia": 5}), "erros_por_dia.csv")
    comum.salvar_tabela(por_hora.round(5), "erros_por_hora.csv")
    comum.salvar_tabela(modelos.round(5), "incerteza_simples.csv")
    comum.salvar_tabela(pares.round(5), "incerteza_pares.csv")
    figura_por_dia(por_dia, DIR_FIGURAS / "erros_por_dia")
    figura_por_hora(por_hora, DIR_FIGURAS / "erros_por_hora")

    logger.info("Acurácia e erro-padrão por dias:\n" + modelos.round(4).to_string(index=False))
    logger.info("Diferenças pareadas:\n" + pares.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
