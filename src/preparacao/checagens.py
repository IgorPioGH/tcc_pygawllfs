"""Checagens da preparação: cada passo faz o que diz?

Recebe data/interim/vra_target.parquet (a base com o alvo, sem balancear) e os
conjuntos congelados de data/processed/. Três checagens:

1. O balanceamento preserva as relações. Descartar voos pontuais ao acaso
   multiplica as chances (odds) de atraso de todos os grupos pelo mesmo fator,
   então a ordem entre os grupos se mantém. Confere com a taxa de atraso por hora
   no treino original e no treino balanceado: Spearman entre as duas séries
   (tem de ser ≥ SPEARMAN_MINIMO) e a razão das odds por hora (quase constante).
2. O target encoding generaliza. Para empresas e aeroportos de origem, o valor
   do encoding (ajustado no treino, como no pipeline) contra a taxa de atraso
   observada na validação, na base sem balancear.
3. A divisão temporal: os três períodos, com as contagens antes e depois do
   balanceamento, conferidas com os parquets.

Produz results/tables/checagem_{balanceamento,target_encoding,divisao}.csv e
results/figures/checagem_{balanceamento,target_encoding,divisao}.{png,pdf}.
"""

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # só gera arquivo, sem abrir janela
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.graficos import (  # noqa: E402
    COR_GRADE,
    COR_SERIE,
    COR_TEXTO,
    COR_TEXTO_SECUNDARIO,
    LARGURA_CHEIA,
    decimal,
    eixo_com_virgula,
    salvar,
)
from src.preparacao.features import COLUNAS_TE, RECORTES, SUAVIZACAO, calcular_mapa_suavizado  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARQUIVO_TARGET = RAIZ / "data" / "interim" / "vra_target.parquet"
DIR_DADOS = RAIZ / "data" / "processed"
DIR_TABELAS = RAIZ / "results" / "tables"
DIR_FIGURAS = RAIZ / "results" / "figures"

SPEARMAN_MINIMO = 0.95
CONJUNTOS = {"train": "treino", "val": "validação", "test": "teste"}
COR_CONJUNTO = {"train": COR_SERIE, "val": "#eda100", "test": "#8c8b85"}
CATEGORIAS_TE = {"ICAO Empresa Aérea": "empresas", "ICAO Aeródromo Origem": "aeroportos de origem"}


def periodos() -> dict[str, tuple[str, str]]:
    """Intervalo [início, fim) de datas de cada conjunto do recorte principal.

    Usada em: base_por_conjunto e checar_divisao.
    Importância: as checagens usam exatamente os cortes da preparação.
    """
    inicio, corte_treino, corte_val, fim = RECORTES["principal"]
    return {"train": (inicio, corte_treino), "val": (corte_treino, corte_val),
            "test": (corte_val, fim)}


def base_por_conjunto(base: pd.DataFrame, conjunto: str) -> pd.DataFrame:
    """Voos da base sem balancear na janela de datas de um conjunto.

    Usada em: checar_balanceamento, checar_target_encoding e checar_divisao.
    Importância: é a referência "antes do balanceamento".
    """
    inicio, fim = periodos()[conjunto]
    return base[(base["Partida Prevista"] >= inicio) & (base["Partida Prevista"] < fim)]


def checar_balanceamento(base: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Taxa de atraso por hora no treino original e no balanceado, a razão das odds e o
    Spearman entre as duas séries. Levanta erro se o Spearman ficar abaixo de
    SPEARMAN_MINIMO.

    Usada em: main.
    Importância: um Spearman baixo indicaria balanceamento por grupo, que apagaria a
    informação da variável.
    """
    original = base_por_conjunto(base, "train")
    balanceado = pd.read_parquet(DIR_DADOS / "train.parquet", columns=["hora", "atrasado"])
    tabela = pd.DataFrame({
        "taxa_original": original["atrasado"].groupby(original["Partida Prevista"].dt.hour).mean(),
        "taxa_balanceada": balanceado["atrasado"].groupby(balanceado["hora"]).mean(),
    }).rename_axis("hora").reset_index()
    odds = lambda p: p / (1 - p)  # noqa: E731
    tabela["razao_odds"] = odds(tabela["taxa_balanceada"]) / odds(tabela["taxa_original"])
    rho = spearmanr(tabela["taxa_original"], tabela["taxa_balanceada"]).statistic
    if rho < SPEARMAN_MINIMO:
        raise ValueError(f"Spearman {rho:.3f} < {SPEARMAN_MINIMO}: o balanceamento mudou a ordem das horas")
    return tabela, float(rho)


def checar_target_encoding(base: pd.DataFrame) -> pd.DataFrame:
    """Para empresas e aeroportos de origem: o encoding ajustado no treino (o mesmo mapa
    suavizado que o pipeline aplica à validação) e a taxa de atraso observada na
    validação, por categoria presente nos dois períodos.

    Usada em: main.
    Importância: correlação alta quer dizer que o encoding carrega informação real fora
    do treino, e não só ruído do período em que foi ajustado.
    """
    treino = base_por_conjunto(base, "train")
    validacao = base_por_conjunto(base, "val")
    tabelas = []
    for coluna, rotulo in CATEGORIAS_TE.items():
        mapa, _ = calcular_mapa_suavizado(treino, coluna, "atrasado", SUAVIZACAO)
        observada = validacao.groupby(coluna)["atrasado"].agg(voos="size", taxa_validacao="mean")
        observada["encoding_treino"] = observada.index.map(mapa)
        tabelas.append(observada.dropna().reset_index(names="categoria")
                       .assign(atributo=COLUNAS_TE[coluna], rotulo=rotulo))
    return pd.concat(tabelas, ignore_index=True)


def resumo_target_encoding(tabela: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por atributo: nº de categorias, voos cobertos e as correlações (Pearson
    e Spearman) entre o encoding e a taxa na validação.

    Usada em: main.
    Importância: é o número que resume a generalização do encoding.
    """
    linhas = []
    for atributo, grupo in tabela.groupby("atributo"):
        linhas.append({"atributo": atributo, "categorias": len(grupo), "voos": int(grupo["voos"].sum()),
                       "pearson": pearsonr(grupo["encoding_treino"], grupo["taxa_validacao"]).statistic,
                       "spearman": spearmanr(grupo["encoding_treino"], grupo["taxa_validacao"]).statistic})
    return pd.DataFrame(linhas)


def checar_divisao(base: pd.DataFrame) -> pd.DataFrame:
    """Para cada conjunto: período, voos e atrasados na base sem balancear e voos depois
    do balanceamento (lidos dos parquets). Levanta erro se o balanceado não for o dobro
    dos atrasados.

    Usada em: main.
    Importância: liga as contagens do texto aos arquivos que os modelos leem.
    """
    linhas = []
    for conjunto, (inicio, fim) in periodos().items():
        original = base_por_conjunto(base, conjunto)
        balanceado = pd.read_parquet(DIR_DADOS / f"{conjunto}.parquet", columns=["atrasado"])
        if len(balanceado) != 2 * original["atrasado"].sum():
            raise ValueError(f"{conjunto}: {len(balanceado)} voos, esperado 2 × atrasados")
        linhas.append({"conjunto": conjunto, "inicio": inicio, "fim": fim,
                       "voos_original": len(original), "atrasados": int(original["atrasado"].sum()),
                       "taxa_original": original["atrasado"].mean(),
                       "voos_balanceado": len(balanceado)})
    return pd.DataFrame(linhas)


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


def figura_balanceamento(tabela: pd.DataFrame, rho: float, caminho_base: Path) -> None:
    """Taxa de atraso por hora, antes e depois do balanceamento.

    Usada em: main.
    Importância: mostra que as duas curvas têm a mesma forma, só em outro nível.
    """
    fig, ax = plt.subplots(figsize=(LARGURA_CHEIA, 2.6), layout="constrained")
    ax.plot(tabela["hora"], tabela["taxa_original"], color="#8c8b85", marker="o", markersize=3,
            label="treino original")
    ax.plot(tabela["hora"], tabela["taxa_balanceada"], color=COR_SERIE, marker="o", markersize=3,
            label="treino balanceado (50/50)")
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xlabel("hora prevista de partida", color=COR_TEXTO_SECUNDARIO)
    ax.set_ylabel("taxa de atraso", color=COR_TEXTO_SECUNDARIO)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(frameon=False)
    limpar_eixo(ax)
    salvar(fig, caminho_base, legenda=(
        "Taxa de atraso por hora prevista de partida no treino, antes e depois do "
        f"balanceamento. As duas séries têm a mesma ordem (Spearman = {decimal(rho)}): "
        "descartar voos pontuais ao acaso desloca o nível, não o padrão."))


def figura_target_encoding(tabela: pd.DataFrame, resumo: pd.DataFrame, caminho_base: Path) -> None:
    """Encoding no treino × taxa na validação, um painel por atributo, com o tamanho do
    ponto pelo volume de voos e a diagonal.

    Usada em: main.
    Importância: mostra, categoria a categoria, o quanto o encoding acerta fora do
    treino.
    """
    fig, eixos = plt.subplots(1, 2, figsize=(LARGURA_CHEIA, 3.0), layout="constrained")
    correlacao = resumo.set_index("atributo")["pearson"]
    for ax, (atributo, grupo) in zip(eixos, tabela.groupby("atributo", sort=False)):
        tamanho = 8 + 200 * grupo["voos"] / grupo["voos"].max()
        ax.scatter(grupo["encoding_treino"], grupo["taxa_validacao"], s=tamanho, color=COR_SERIE,
                   alpha=0.5, edgecolors="white", linewidths=0.5)
        limite = max(grupo["encoding_treino"].max(), grupo["taxa_validacao"].max()) * 1.05
        ax.plot([0, limite], [0, limite], color=COR_TEXTO_SECUNDARIO, linewidth=0.8, linestyle=":")
        ax.set_title(f"{grupo['rotulo'].iloc[0]} (r = {decimal(correlacao[atributo])})",
                     loc="left", color=COR_TEXTO)
        ax.set_xlabel("encoding ajustado no treino", color=COR_TEXTO_SECUNDARIO)
        eixo_com_virgula(ax.xaxis)
        eixo_com_virgula(ax.yaxis)
        limpar_eixo(ax)
    eixos[0].set_ylabel("taxa de atraso na validação", color=COR_TEXTO_SECUNDARIO)
    salvar(fig, caminho_base, legenda=(
        "Target encoding ajustado no treino (jan/2024 a ago/2025) contra a taxa de atraso "
        "observada na validação (set–out/2025, sem balancear), por empresa e por aeroporto de "
        "origem. O tamanho do ponto é o volume de voos na validação; a linha pontilhada é a "
        "diagonal; r é a correlação de Pearson."))


def figura_divisao(divisao: pd.DataFrame, caminho_base: Path) -> None:
    """Linha do tempo dos três períodos, com as contagens antes e depois do
    balanceamento.

    Usada em: main.
    Importância: mostra a divisão temporal de uma vez: o que é treino, validação e
    teste, e quantos voos sobram em cada um.
    """
    fig, ax = plt.subplots(figsize=(LARGURA_CHEIA, 1.9), layout="constrained")
    # Treino: rótulo centrado; validação e teste (curtos): alinhados ao fim do período,
    # em alturas diferentes, para não se sobreporem
    posicao = {"train": ("center", 0.35), "val": ("right", 0.35), "test": ("right", 1.15)}
    for linha in divisao.itertuples():
        inicio, fim = pd.Timestamp(linha.inicio), pd.Timestamp(linha.fim)
        ax.barh(0, fim - inicio, left=inicio, height=0.5, color=COR_CONJUNTO[linha.conjunto])
        alinhamento, altura = posicao[linha.conjunto]
        x = inicio + (fim - inicio) / 2 if alinhamento == "center" else fim
        texto = (f"{CONJUNTOS[linha.conjunto]}\n{linha.voos_original:,} → "
                 f"{linha.voos_balanceado:,}").replace(",", ".")
        ax.text(x, altura, texto, ha=alinhamento, va="bottom", color=COR_TEXTO)
    ax.set_ylim(-0.4, 2.0)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 5, 9]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    for lado in ["top", "right", "left"]:
        ax.spines[lado].set_visible(False)
    ax.tick_params(colors=COR_TEXTO_SECUNDARIO)
    salvar(fig, caminho_base, legenda=(
        "Divisão temporal: treino (jan/2024 a ago/2025), validação (set–out/2025) e teste "
        "(nov–dez/2025), com o número de voos antes → depois do balanceamento 50/50."))


def main() -> None:
    """Roda as três checagens e salva as tabelas e as figuras.

    Usada em: python src/preparacao/checagens.py; main.py, etapa "checagens".
    Importância: cada decisão da preparação passa a ter uma evidência produzida pelo
    pipeline.
    """
    base = pd.read_parquet(ARQUIVO_TARGET, columns=["Partida Prevista", "atrasado", *CATEGORIAS_TE])
    balanceamento, rho = checar_balanceamento(base)
    encoding = checar_target_encoding(base)
    resumo = resumo_target_encoding(encoding)
    divisao = checar_divisao(base)

    DIR_TABELAS.mkdir(parents=True, exist_ok=True)
    balanceamento.assign(spearman=rho).round(4).to_csv(
        DIR_TABELAS / "checagem_balanceamento.csv", index=False)
    resumo.round(4).to_csv(DIR_TABELAS / "checagem_target_encoding.csv", index=False)
    divisao.round(4).to_csv(DIR_TABELAS / "checagem_divisao.csv", index=False)

    figura_balanceamento(balanceamento, rho, DIR_FIGURAS / "checagem_balanceamento")
    figura_target_encoding(encoding, resumo, DIR_FIGURAS / "checagem_target_encoding")
    figura_divisao(divisao, DIR_FIGURAS / "checagem_divisao")

    logger.info(f"Balanceamento: Spearman por hora = {rho:.3f}; razão das odds entre "
                f"{balanceamento['razao_odds'].min():.2f} e {balanceamento['razao_odds'].max():.2f}")
    logger.info("Target encoding:\n" + resumo.round(3).to_string(index=False))
    logger.info("Divisão:\n" + divisao.to_string(index=False))


if __name__ == "__main__":
    main()
