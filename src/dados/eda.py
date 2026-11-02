"""Entendimento dos dados: qualidade da base bruta, descartes e análise exploratória.

Recebe data/interim/vra_bruto.parquet (os 24 CSVs mensais consolidados),
data/interim/vra_target.parquet (voos realizados, com o atraso e o alvo) e
data/processed/train.parquet (os 29 atributos do treino). Tudo o que envolve o
alvo usa só voos com Partida Prevista antes de FIM_ANALISE (01/11/2025): o
período de teste fica fora de qualquer decisão.

Produz, em results/tables/:
    eda_qualidade.csv          por coluna da base bruta: tipo, nulos, valores
                               distintos, categorias raras e o valor mais comum
    eda_descartes.csv          o funil da base bruta até a base com o alvo
    eda_taxa_por_atributo.csv  voos e taxa de atraso por valor de cada atributo
e, em results/figures/:
    eda_funil                  os descartes, passo a passo
    eda_serie_mensal           voos e taxa de atraso por mês, 2024 × 2025
    eda_histograma_atraso      distribuição do atraso na partida, com o limiar de 15 min
    eda_taxa_calendario        taxa de atraso por hora, dia da semana e mês
    eda_taxa_categorias        taxa de atraso das maiores empresas e aeroportos
    eda_correlacao             Spearman entre os 29 atributos, no treino

A taxa por atributo é uma relação marginal: ela não isola o efeito de cada
variável (hora e aeroporto, por exemplo, se confundem). É por isso que o trabalho
usa modelos e métodos de importância, e não só tabelas como estas.
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

from src.graficos import (  # noqa: E402
    COR_GRADE,
    COR_SERIE,
    COR_TEXTO,
    COR_TEXTO_SECUNDARIO,
    ESCALA_SPEARMAN,
    LARGURA_CHEIA,
    decimal,
    formatar_barra,
    porcentagem,
    salvar,
)
from src.modelagem.baselines import NAO_ATRIBUTOS  # noqa: E402
from src.preparacao import target  # noqa: E402
from src.preparacao.clean import remover_sem_horario, separar_realizados_cancelados  # noqa: E402
from src.preparacao.features import features_sem_alvo  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARQUIVO_BRUTO = RAIZ / "data" / "interim" / "vra_bruto.parquet"
ARQUIVO_TARGET = RAIZ / "data" / "interim" / "vra_target.parquet"
ARQUIVO_TREINO = RAIZ / "data" / "processed" / "train.parquet"
DIR_TABELAS = RAIZ / "results" / "tables"
DIR_FIGURAS = RAIZ / "results" / "figures"

FIM_ANALISE = "2025-11-01"  # início do teste: nada que envolva o alvo passa daqui
# Tamanho da base na versão dos dados usada (ANAC, "Atualizado em: 2026-10-07"):
# se mudar, os dados mudaram e todo o resto precisa ser refeito
VOOS_BRUTOS = 1_992_831
VOOS_COM_ALVO = 1_855_980
CATEGORIA_RARA = 5            # categoria com menos registros que isso é "rara"
TOP_EMPRESAS, TOP_AEROPORTOS = 10, 15
LIMIAR_ATRASO = 15
EIXO_MIN, EIXO_MAX, LARGURA_BIN = -40, 180, 5
DIAS = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]
MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
COR_ANO = {2024: "#8c8b85", 2025: COR_SERIE}


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------

def tabela_qualidade(bruto: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por coluna da base bruta: tipo, nulos (n e %), valores distintos, nº de
    categorias com menos de CATEGORIA_RARA registros (colunas de texto) e o valor mais
    comum com a sua fração.

    Usada em: main.
    Importância: documenta, com números reproduzíveis, os problemas que a limpeza
    resolve (horários vazios, coluna de justificativa vazia, alta cardinalidade).
    """
    linhas = []
    for coluna in bruto.columns:
        serie = bruto[coluna]
        contagem = serie.value_counts()
        texto = not (pd.api.types.is_numeric_dtype(serie) or pd.api.types.is_datetime64_any_dtype(serie))
        linhas.append({
            "coluna": coluna, "tipo": str(serie.dtype), "nulos": int(serie.isna().sum()),
            "nulos_pct": float(serie.isna().mean() * 100), "distintos": int(serie.nunique()),
            "categorias_raras": int((contagem < CATEGORIA_RARA).sum()) if texto else None,
            "valor_mais_comum": str(contagem.index[0]) if len(contagem) else None,
            "fracao_mais_comum": float(contagem.iloc[0] / len(serie)) if len(contagem) else None,
        })
    return pd.DataFrame(linhas)


def funil_de_descartes(bruto: pd.DataFrame) -> pd.DataFrame:
    """Passo a passo da base bruta até a base com o alvo, com as mesmas funções e
    limites da limpeza (clean.py) e da criação do alvo (target.py).

    Usada em: main.
    Importância: mostra quantos voos saem em cada critério e confere que o funil chega
    exatamente à base usada (VOOS_COM_ALVO).
    """
    realizados, _ = separar_realizados_cancelados(bruto)
    com_horario = remover_sem_horario(realizados)
    no_periodo = com_horario[(com_horario["Partida Prevista"] >= target.DATA_INICIO)
                             & (com_horario["Partida Prevista"] < target.DATA_FIM)]
    atraso = (no_periodo["Partida Real"] - no_periodo["Partida Prevista"]).dt.total_seconds() / 60
    plausiveis = no_periodo[atraso.between(target.ATRASO_MIN_PLAUSIVEL, target.ATRASO_MAX_PLAUSIVEL)]
    passos = [
        ("base bruta", "24 CSVs mensais do VRA, 2024–2025", bruto),
        ("cancelados", "Situação Voo diferente de REALIZADO", realizados),
        ("sem horário", "Partida Prevista ou Partida Real vazia", com_horario),
        ("fora do período", "Partida Prevista fora de 2024–2025", no_periodo),
        ("atraso implausível", f"atraso fora de [{target.ATRASO_MIN_PLAUSIVEL}, "
                               f"{target.ATRASO_MAX_PLAUSIVEL}] min", plausiveis),
    ]
    tabela = pd.DataFrame([{"passo": passo, "criterio": criterio, "voos": len(base)}
                           for passo, criterio, base in passos])
    tabela["removidos"] = (-tabela["voos"].diff()).fillna(0).astype(int)
    if tabela["voos"].iloc[0] != VOOS_BRUTOS or tabela["voos"].iloc[-1] != VOOS_COM_ALVO:
        raise ValueError(f"O funil mudou: {tabela['voos'].tolist()}; os dados não são a versão esperada")
    return tabela


def base_ate_outubro() -> pd.DataFrame:
    """A base com o alvo e os atributos sem alvo (features_sem_alvo, calculados na base
    inteira, como no pipeline), só com os voos antes de FIM_ANALISE.

    Usada em: main.
    Importância: é a base de todas as taxas de atraso; o teste fica de fora.
    """
    base = pd.read_parquet(ARQUIVO_TARGET)
    if len(base) != VOOS_COM_ALVO:
        raise ValueError(f"vra_target.parquet tem {len(base)} voos, esperado {VOOS_COM_ALVO}")
    base = features_sem_alvo(base)
    base = base[base["Partida Prevista"] < FIM_ANALISE]
    if base["Partida Prevista"].max() >= pd.Timestamp(FIM_ANALISE):
        raise ValueError("A EDA usou voos do período de teste")
    return base


def taxa_por_atributo(base: pd.DataFrame) -> pd.DataFrame:
    """Voos e taxa de atraso por valor de cada atributo: calendário, tipo de linha,
    feriado e véspera; as maiores empresas e aeroportos de origem; quintis da duração
    prevista e do congestionamento.

    Usada em: main.
    Importância: é a descrição marginal de cada atributo antes de qualquer modelo.
    """
    agrupamentos = {
        "hora": base["hora"], "dia_sem": base["dia_sem"], "mes": base["mes"],
        "periodo_dia": base["periodo_dia"], "tipo_linha": base["Código Tipo Linha"],
        "feriado": base["feriado"], "vespera_feriado": base["vespera_feriado"],
        "duracao_prevista_min": pd.qcut(base["duracao_prevista_min"], 5, duplicates="drop"),
        "congestion_origem": pd.qcut(base["congestion_origem"], 5, duplicates="drop"),
    }
    for nome, coluna, top in [("empresa", "ICAO Empresa Aérea", TOP_EMPRESAS),
                              ("origem", "ICAO Aeródromo Origem", TOP_AEROPORTOS)]:
        maiores = base[coluna].value_counts().index[:top]
        agrupamentos[nome] = base[coluna].where(base[coluna].isin(maiores))
    tabelas = []
    for atributo, chave in agrupamentos.items():
        grupo = base["atrasado"].groupby(chave, observed=True).agg(voos="size", taxa_atraso="mean")
        tabelas.append(grupo.reset_index(names="valor").assign(atributo=atributo))
    tabela = pd.concat(tabelas, ignore_index=True)[["atributo", "valor", "voos", "taxa_atraso"]]
    tabela["valor"] = tabela["valor"].astype(str)
    return tabela


def serie_mensal(base: pd.DataFrame) -> pd.DataFrame:
    """Voos e taxa de atraso por (ano, mês), até out/2025.

    Usada em: main.
    Importância: mostra a sazonalidade, que justifica os recortes com o mesmo
    calendário e explica por que o teste (nov–dez) é um período difícil.
    """
    data = base["Partida Prevista"]
    serie = (base["atrasado"].groupby([data.dt.year.rename("ano"), data.dt.month.rename("mes")])
             .agg(voos="size", taxa_atraso="mean").reset_index())
    if serie["voos"].sum() != len(base):
        raise ValueError("A série mensal não soma o total de voos")
    return serie


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def limpar_eixo(ax, grade: str = "y") -> None:
    """Grade discreta, sem as bordas de cima e da direita, ticks em cinza.

    Usada em: todas as figuras deste script.
    Importância: só apresentação.
    """
    ax.grid(axis=grade, color=COR_GRADE, linewidth=0.6)
    ax.set_axisbelow(True)
    for lado in ["top", "right"]:
        ax.spines[lado].set_visible(False)
    ax.tick_params(colors=COR_TEXTO_SECUNDARIO)


def formatar_porcentagem(ax, eixo: str = "y") -> None:
    """Eixo em porcentagem com vírgula decimal.

    Usada em: figuras de taxa de atraso.
    Importância: só apresentação.
    """
    formato = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}")
    (ax.yaxis if eixo == "y" else ax.xaxis).set_major_formatter(formato)


def figura_funil(funil: pd.DataFrame, caminho_base: Path) -> None:
    """Barras horizontais com os voos restantes depois de cada passo.

    Usada em: main.
    Importância: mostra de onde saem os descartes.
    """
    fig, ax = plt.subplots(figsize=(LARGURA_CHEIA, 2.4), layout="constrained")
    y = np.arange(len(funil))[::-1]
    ax.barh(y, funil["voos"] / 1e6, color=COR_SERIE, height=0.6)
    for posicao, linha in zip(y, funil.itertuples()):
        texto = f"{linha.voos:,}".replace(",", ".")
        if linha.removidos:
            texto += f"  (−{linha.removidos:,})".replace(",", ".")
        ax.text(linha.voos / 1e6 + 0.02, posicao, texto, va="center", color=COR_TEXTO)
    ax.set_yticks(y, funil["passo"])
    ax.set_xlim(0, funil["voos"].max() / 1e6 * 1.45)
    ax.set_xlabel("voos (milhões)", color=COR_TEXTO_SECUNDARIO)
    limpar_eixo(ax, grade="x")
    salvar(fig, caminho_base, legenda=(
        "Funil de descartes da base VRA: voos restantes depois de cada critério de limpeza "
        "(entre parênteses, os voos removidos no passo)."))


def figura_serie_mensal(serie: pd.DataFrame, caminho_base: Path) -> None:
    """Dois painéis: voos por mês e taxa de atraso por mês, uma linha por ano.

    Usada em: main.
    Importância: mostra a sazonalidade e a diferença entre os anos.
    """
    fig, eixos = plt.subplots(1, 2, figsize=(LARGURA_CHEIA, 2.6), layout="constrained")
    for ano, do_ano in serie.groupby("ano"):
        eixos[0].plot(do_ano["mes"], do_ano["voos"] / 1e3, color=COR_ANO[ano], marker="o",
                      markersize=3, label=str(ano))
        eixos[1].plot(do_ano["mes"], do_ano["taxa_atraso"], color=COR_ANO[ano], marker="o",
                      markersize=3, label=str(ano))
    eixos[0].set_ylabel("voos (milhares)", color=COR_TEXTO_SECUNDARIO)
    eixos[1].set_ylabel("taxa de atraso", color=COR_TEXTO_SECUNDARIO)
    formatar_porcentagem(eixos[1])
    for ax in eixos:
        ax.set_xticks(range(1, 13), [m[0].upper() for m in MESES])
        limpar_eixo(ax)
    eixos[1].legend(frameon=False)
    salvar(fig, caminho_base, legenda=(
        "Voos realizados e taxa de atraso (≥ 15 min) por mês, em 2024 e em 2025 (até outubro; "
        "novembro e dezembro de 2025 são o período de teste e ficam de fora)."))


def histograma_atraso(atraso_min: pd.Series, caminho_base: Path) -> None:
    """Histograma do atraso na partida (eixo de -40 a 180 min), com o limiar de 15 min,
    a fração de atrasados e a cauda resumida numa anotação.

    Usada em: main.
    Importância: mostra a forma do alvo e onde fica o limiar que define a classe
    positiva.
    """
    total = len(atraso_min)
    fracao_atrasados = (atraso_min >= LIMIAR_ATRASO).mean()
    fracao_cauda = (atraso_min > EIXO_MAX).mean()
    bins = np.arange(EIXO_MIN, EIXO_MAX + LARGURA_BIN, LARGURA_BIN)
    contagem, bordas = np.histogram(atraso_min.clip(upper=EIXO_MAX), bins=bins)

    fig, ax = plt.subplots(figsize=(LARGURA_CHEIA, 2.8), layout="constrained")
    cores = [COR_SERIE if borda >= LIMIAR_ATRASO else "#9ec5f4" for borda in bordas[:-1]]
    ax.bar(bordas[:-1], contagem / 1e3, width=LARGURA_BIN, align="edge", color=cores,
           edgecolor="white", linewidth=0.4)
    ax.axvline(LIMIAR_ATRASO, color=COR_TEXTO, linewidth=1, linestyle="--")
    ax.text(LIMIAR_ATRASO + 3, ax.get_ylim()[1] * 0.92,
            f"atraso ≥ {LIMIAR_ATRASO} min: {porcentagem(fracao_atrasados)} dos voos",
            color=COR_TEXTO, va="top")
    ax.text(EIXO_MAX, ax.get_ylim()[1] * 0.75,
            f"{porcentagem(fracao_cauda)} acima de {EIXO_MAX} min\n(somados na última barra)",
            color=COR_TEXTO_SECUNDARIO, ha="right", va="top")
    ax.set_xlabel("atraso na partida (min)", color=COR_TEXTO_SECUNDARIO)
    ax.set_ylabel("voos (milhares)", color=COR_TEXTO_SECUNDARIO)
    ax.set_xlim(EIXO_MIN, EIXO_MAX + LARGURA_BIN)
    limpar_eixo(ax)
    salvar(fig, caminho_base, legenda=(
        f"Distribuição do atraso na partida dos {total:,} voos realizados de jan/2024 a "
        f"out/2025. A linha tracejada é o limiar de {LIMIAR_ATRASO} min que define o voo "
        "atrasado.").replace(",", "."))


def figura_taxa_calendario(taxas: pd.DataFrame, media: float, caminho_base: Path) -> None:
    """Taxa de atraso por hora, dia da semana e mês (três painéis), com a taxa média.

    Usada em: main.
    Importância: mostra os padrões de calendário que os atributos de data carregam.
    """
    fig, eixos = plt.subplots(1, 3, figsize=(LARGURA_CHEIA, 2.4), layout="constrained",
                              sharey=True, gridspec_kw={"width_ratios": [2.4, 1, 1.6]})
    for ax, atributo, rotulo, nomes in [
            (eixos[0], "hora", "hora prevista", None),
            (eixos[1], "dia_sem", "dia da semana", [d[0].upper() for d in DIAS]),
            (eixos[2], "mes", "mês", [m[0].upper() for m in MESES])]:
        dados = taxas[taxas["atributo"] == atributo]
        x = dados["valor"].astype(int).to_numpy()
        ax.bar(x, dados["taxa_atraso"], color=COR_SERIE, width=0.75)
        ax.axhline(media, color=COR_TEXTO_SECUNDARIO, linewidth=0.8, linestyle=":")
        if nomes is not None:
            ax.set_xticks(x, [nomes[i - (1 if atributo == "mes" else 0)] for i in x])
        else:
            ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xlabel(rotulo, color=COR_TEXTO_SECUNDARIO)
        limpar_eixo(ax)
    eixos[0].set_ylabel("taxa de atraso", color=COR_TEXTO_SECUNDARIO)
    formatar_porcentagem(eixos[0])
    salvar(fig, caminho_base, legenda=(
        f"Taxa de atraso por hora prevista de partida, dia da semana e mês (jan/2024 a "
        f"out/2025). A linha pontilhada é a taxa média ({porcentagem(media)})."))


def figura_taxa_categorias(taxas: pd.DataFrame, media: float, caminho_base: Path) -> None:
    """Taxa de atraso das maiores empresas e dos maiores aeroportos de origem, em barras
    horizontais ordenadas, com a taxa média.

    Usada em: main.
    Importância: motiva o target encoding: muitas categorias, com taxas muito
    diferentes entre si.
    """
    fig, eixos = plt.subplots(1, 2, figsize=(LARGURA_CHEIA, 3.4), layout="constrained")
    for ax, atributo, titulo in [(eixos[0], "empresa", f"{TOP_EMPRESAS} maiores empresas"),
                                 (eixos[1], "origem", f"{TOP_AEROPORTOS} maiores aeroportos de origem")]:
        dados = taxas[taxas["atributo"] == atributo].sort_values("taxa_atraso")
        ax.barh(dados["valor"], dados["taxa_atraso"], color=COR_SERIE, height=0.7)
        ax.axvline(media, color=COR_TEXTO_SECUNDARIO, linewidth=0.8, linestyle=":")
        ax.set_title(titulo, loc="left", color=COR_TEXTO)
        formatar_porcentagem(ax, eixo="x")
        limpar_eixo(ax, grade="x")
    salvar(fig, caminho_base, legenda=(
        f"Taxa de atraso das {TOP_EMPRESAS} empresas e dos {TOP_AEROPORTOS} aeroportos de "
        f"origem com mais voos (jan/2024 a out/2025). A linha pontilhada é a taxa média "
        f"({porcentagem(media)})."))


def figura_correlacao(treino: pd.DataFrame, caminho_base: Path) -> None:
    """Matriz de correlação de Spearman entre os 29 atributos, no treino.

    Usada em: main.
    Importância: mostra, antes de qualquer resultado, os pares redundantes por
    construção (hora × período do dia, mês × trimestre × dia do ano, dia da semana ×
    fim de semana), que depois reaparecem nas interações do PyGAwLLfs.
    """
    atributos = treino.drop(columns=NAO_ATRIBUTOS)
    correlacao = atributos.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(LARGURA_CHEIA, 5.8), layout="constrained")
    imagem = ax.pcolormesh(correlacao.to_numpy(), cmap=ESCALA_SPEARMAN, vmin=-1, vmax=1,
                           edgecolors="white", linewidth=0.3)
    posicoes = np.arange(len(correlacao)) + 0.5
    ax.set_xticks(posicoes, correlacao.columns, rotation=90)
    ax.set_yticks(posicoes, correlacao.columns)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.tick_params(length=0, colors=COR_TEXTO_SECUNDARIO)
    for lado in ax.spines.values():
        lado.set_visible(False)
    formatar_barra(fig.colorbar(imagem, ax=ax, shrink=0.6), "Spearman")
    salvar(fig, caminho_base, legenda=(
        "Correlação de Spearman entre os 29 atributos, no conjunto de treino. Os blocos fortes "
        "fora da diagonal são pares redundantes por construção (por exemplo, hora e período do "
        "dia; mês, trimestre e dia do ano)."))


# ---------------------------------------------------------------------------

def main() -> None:
    """Gera as tabelas e as figuras da análise exploratória.

    Usada em: python src/dados/eda.py; main.py, etapa "eda".
    Importância: descreve os dados antes de qualquer modelo, de forma reproduzível.
    """
    bruto = pd.read_parquet(ARQUIVO_BRUTO)
    qualidade = tabela_qualidade(bruto)
    funil = funil_de_descartes(bruto)
    del bruto

    base = base_ate_outubro()
    taxas = taxa_por_atributo(base)
    serie = serie_mensal(base)
    media = base["atrasado"].mean()

    DIR_TABELAS.mkdir(parents=True, exist_ok=True)
    qualidade.round(4).to_csv(DIR_TABELAS / "eda_qualidade.csv", index=False)
    funil.to_csv(DIR_TABELAS / "eda_descartes.csv", index=False)
    taxas.round(4).to_csv(DIR_TABELAS / "eda_taxa_por_atributo.csv", index=False)

    figura_funil(funil, DIR_FIGURAS / "eda_funil")
    figura_serie_mensal(serie, DIR_FIGURAS / "eda_serie_mensal")
    histograma_atraso(base["atraso_min"], DIR_FIGURAS / "eda_histograma_atraso")
    figura_taxa_calendario(taxas, media, DIR_FIGURAS / "eda_taxa_calendario")
    figura_taxa_categorias(taxas, media, DIR_FIGURAS / "eda_taxa_categorias")
    figura_correlacao(pd.read_parquet(ARQUIVO_TREINO), DIR_FIGURAS / "eda_correlacao")

    with pd.option_context("display.width", 200, "display.max_columns", 10):
        logger.info("Qualidade da base bruta:\n" + qualidade.round(3).to_string(index=False))
        logger.info("Funil de descartes:\n" + funil.to_string(index=False))
        logger.info(f"Voos até out/2025: {len(base)} | taxa de atraso média {decimal(media)}")


if __name__ == "__main__":
    main()
