"""Etapa 13c — sensibilidade à agregação das sementes no PyGAwLLfs (decisão 15).

Pergunta: o que muda nos resultados de interação quando um zero do vint deixa de
contar como "sem interação" e passa a contar como "par não sondado"?

As duas agregações de graph.interacao_media passam pelas mesmas funções das
Etapas 13b e 15:
    com_zeros   média das 5 sementes com os zeros (a original)
    sondadas    média só das sementes em que o par foi sondado; ranking só com
                pares sondados em pelo menos 2 sementes (a principal)

1. Cobertura: zeros por execução × o esperado se todo zero fosse um par não
   sorteado (SONDAGENS_POR_EXECUCAO sorteios uniformes entre os 406 pares).
2. Artefato de cobertura: Spearman entre a média do par e o nº de sementes em
   que ele foi sondado (nos 406 pares e nos pares do H).
3. Concordância PyGAwLLfs × H, estabilidade entre sementes e split-half.
4. Redundância: onde caem os pares redundantes no ranking do PyGAwLLfs.
5. Top-10 e grafo reduzido: quantos pares se mantêm de uma agregação para a
   outra e como o top-10 se sai no teste de domínio da Etapa 15.
6. Seleção da Etapa 15: candidatos, corroborados e controles.
Um par testado no domínio tem um resultado só nas duas agregações
(interacoes.teste_de_dominio, com o cache da Etapa 15).

Produz:
    results/tables/sensibilidade_cobertura.csv   zeros observados × esperados, por modelo
    results/tables/sensibilidade_agregacao.csv   bloco, modelo, conjunto, métrica, valor
                                                 com zeros e valor só com as sondadas
    results/tables/sensibilidade_top10.csv       top-10 de cada modelo nas duas agregações
    results/figures/sensibilidade_agregacao.{png,pdf}
Só lê CSVs e, para o teste de domínio, treino + validação (nunca o teste). Roda
depois das etapas dominio e interacoes.
"""

import logging
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # só gera arquivo, sem abrir janela
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.compare import interacoes as it  # noqa: E402
from src.compare.figuras import (  # noqa: E402
    COR_GRADE,
    COR_TEXTO,
    COR_TEXTO_SECUNDARIO,
    ROTULO_MODELO,
    decimal,
    salvar,
)
from src.compare.rankings import MODELOS  # noqa: E402
from src.domain import validate_pairs as v  # noqa: E402
from src.importance import comum  # noqa: E402
from src.interaction import graph  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DIR_FIGURAS = RAIZ / "results" / "figures"
AGREGACOES = ["com_zeros", "sondadas"]
ROTULO_AGREGACAO = {"com_zeros": "com os zeros (original)",
                    "sondadas": "só sementes sondadas (principal)"}
COR_AGREGACAO = {"com_zeros": "#eb6834", "sondadas": "#2a78d6"}
# _generation_ll do PyGAwLLfs, com população 50 e cruzamento 0,8: 40 filhos de
# cruzamento, depois sondagens de 3 filhos até passar de 48 → 3 sondagens por
# geração; 100 gerações → 300 sorteios uniformes de par por execução
SONDAGENS_POR_EXECUCAO = 300
TOP_K = 10


# ---------------------------------------------------------------------------
# 1–2. Cobertura e artefato de cobertura
# ---------------------------------------------------------------------------

def nao_sorteados_esperados(n_pares: int, sorteios: int) -> tuple[float, float]:
    """(média, desvio-padrão) do nº de pares nunca sorteados em `sorteios` sorteios
    uniformes entre n_pares (problema da ocupação: caixas vazias).

    Usada em: cobertura.
    Importância: é a referência para dizer quantos zeros do vint são só "par não
    sondado"; o que passa disso é sondagem com efeito nulo.
    """
    p1 = (1 - 1 / n_pares) ** sorteios
    p2 = (1 - 2 / n_pares) ** sorteios
    media = n_pares * p1
    variancia = media + n_pares * (n_pares - 1) * p2 - media ** 2
    return media, float(np.sqrt(variancia))


def cobertura(gawll: pd.DataFrame) -> pd.DataFrame:
    """Por modelo (médias das 5 execuções): pares sondados (graph.marcar_sondagens) ×
    o esperado pelo sorteio, zeros do vint separados em "não sondado" e "sondado,
    efeito nulo", e quantos pares foram sondados em 0, 1, ..., 5 sementes.

    Usada em: main.
    Importância: confere a reconstrução das sondagens (os pares sondados batem com o
    sorteio, e o RF, que nunca dá efeito nulo exato, sonda os mesmos pares que os
    outros) e mede quantos zeros são medida em cada modelo.
    """
    gawll = graph.marcar_sondagens(gawll)
    media = graph.interacao_media(gawll, agregacao="sondadas")
    linhas = []
    for modelo, grupo in gawll.groupby("model"):
        n_pares = len(grupo[["feature_a", "feature_b"]].drop_duplicates())
        n_sementes = grupo["seed"].nunique()
        por_semente = grupo.groupby("seed")
        zero = grupo["interaction"] == 0
        nao_sorteados, desvio = nao_sorteados_esperados(n_pares, SONDAGENS_POR_EXECUCAO)
        contagem = media.loc[media["model"] == modelo, "n_sementes"].value_counts()
        linhas.append({
            "model": modelo, "n_pares": n_pares, "sondagens_por_execucao": SONDAGENS_POR_EXECUCAO,
            "pares_sondados_por_execucao": por_semente["sondada"].sum().mean(),
            "pares_sondados_esperados": n_pares - nao_sorteados, "desvio_esperado": desvio,
            "zeros_por_execucao": zero.groupby(grupo["seed"]).sum().mean(),
            "zeros_nao_sondados_por_execucao": (zero & ~grupo["sondada"])
                                               .groupby(grupo["seed"]).sum().mean(),
            "zeros_medidos_por_execucao": (zero & grupo["sondada"])
                                          .groupby(grupo["seed"]).sum().mean(),
            **{f"pares_sondados_{k}_sementes": int(contagem.get(k, 0))
               for k in range(n_sementes + 1)},
            "pares_nunca_sondados_esperados": n_pares * (1 - 1 / n_pares)
                                              ** (SONDAGENS_POR_EXECUCAO * n_sementes),
        })
    return pd.DataFrame(linhas)


def linha(bloco: str, modelo: str, conjunto: str, metrica: str, valores: dict) -> dict:
    """Uma linha da tabela de sensibilidade: um valor por agregação.

    Usada em: todos os blocos deste arquivo.
    Importância: deixa as duas agregações lado a lado, no mesmo formato.
    """
    return {"bloco": bloco, "model": modelo, "conjunto": conjunto, "metrica": metrica,
            **{agregacao: valores.get(agregacao) for agregacao in AGREGACOES}}


def artefato_de_cobertura(medias: dict, pares_h: dict) -> list[dict]:
    """Spearman entre a média do par e o nº de sementes em que ele foi sondado, nos
    406 pares e nos pares do H, em cada agregação (só pares com média).

    Usada em: main.
    Importância: perto de 0 = o ranking mede a interação; alto = mede a cobertura.
    """
    linhas = []
    for modelo in MODELOS:
        conjuntos = {"406 pares": None}
        if modelo in pares_h:
            conjuntos["pares do H"] = pares_h[modelo]
        for conjunto, pares in conjuntos.items():
            valores = {}
            for agregacao, media in medias.items():
                grupo = media[(media["model"] == modelo) & media["interaction"].notna()]
                if pares is not None:
                    grupo = grupo[(grupo["feature_a"] + "|" + grupo["feature_b"]).isin(pares)]
                valores[agregacao] = spearmanr(grupo["interaction"], grupo["n_sementes"]).statistic
            linhas.append(linha("artefato de cobertura", modelo, conjunto,
                                "spearman(média do par, nº de sementes sondadas)", valores))
    return linhas


# ---------------------------------------------------------------------------
# 3. Concordância, estabilidade e split-half
# ---------------------------------------------------------------------------

def split_half(gawll: pd.DataFrame, agregacao: str) -> dict[str, float]:
    """Por modelo: Spearman médio entre as médias de duas metades das sementes (2 × 3),
    nas 10 divisões possíveis, nos pares com média nas duas metades. Recebe a tabela
    de um método só (o PyGAwLLfs ou, como referência, o H_abs).

    Usada em: main.
    Importância: confiabilidade do ranking médio; é o número de docs/interacoes-h.md
    §3, agora calculado no código e nas duas agregações.
    """
    sementes = sorted(gawll["seed"].unique())
    resultado = {}
    for modelo, grupo in gawll.groupby("model"):
        correlacoes = []
        for metade in combinations(sementes, 2):
            medias = [graph.interacao_media(grupo[grupo["seed"].isin(parte) == dentro],
                                            agregacao=agregacao)
                      .set_index(["feature_a", "feature_b"])["interaction"]
                      for parte, dentro in [(metade, True), (metade, False)]]
            comuns = medias[0].notna() & medias[1].notna()
            correlacoes.append(spearmanr(medias[0][comuns], medias[1][comuns]).statistic)
        resultado[modelo] = float(np.mean(correlacoes))
    return resultado


def spearman_brown(correlacao_metade: float, fator: float) -> float:
    """Confiabilidade de uma média com `fator` vezes mais sementes que a metade:
    fator·r / (1 + (fator − 1)·r). Com o split-half, fator 2 ≈ as 5 sementes (as
    metades têm 2 e 3) e fator 4 ≈ 10 sementes.

    Usada em: main.
    Importância: diz quanto mais sementes ajudariam (decisão 13).
    """
    return fator * correlacao_metade / (1 + (fator - 1) * correlacao_metade)


def concordancia(interacao: pd.DataFrame) -> tuple[list[dict], dict]:
    """PyGAwLLfs × H_abs, PyGAwLLfs × H normalizado e estabilidade entre sementes do
    PyGAwLLfs (interacoes.comparar_modelo) nas duas agregações; devolve também as
    tabelas por par, para os blocos seguintes.

    Usada em: main.
    Importância: é a comparação principal da Etapa 13b, sob as duas leituras do zero.
    """
    por_agregacao, pares = {}, {}
    for agregacao in AGREGACOES:
        linhas_modelo, tabelas = [], []
        for modelo in it.MODELOS_H:
            linhas_m, por_par = it.comparar_modelo(interacao, modelo, agregacao=agregacao)
            linhas_modelo.extend(linhas_m)
            tabelas.append(por_par)
        por_agregacao[agregacao] = pd.DataFrame(linhas_modelo).set_index(["model", "comparacao"])
        pares[agregacao] = pd.concat(tabelas, ignore_index=True)

    linhas = []
    for modelo in it.MODELOS_H:
        for comparacao in ["gawll x friedman_h_abs", "gawll x friedman_h", "estabilidade gawll"]:
            for metrica in ["spearman", "kendall_tau_b", "rbo", "top5", "top10", "top15"]:
                linhas.append(linha("concordância" if comparacao.startswith("gawll x")
                                    else "estabilidade", modelo, "pares do H",
                                    f"{comparacao}: {metrica}",
                                    {a: por_agregacao[a].loc[(modelo, comparacao), metrica]
                                     for a in AGREGACOES}))
        linhas.append(linha("concordância", modelo, "pares do H",
                            "gawll x friedman_h_abs: pares na correlação",
                            {a: por_agregacao[a].loc[(modelo, "gawll x friedman_h_abs"),
                                                     "n_pares_correlacao"] for a in AGREGACOES}))
    return linhas, pares


# ---------------------------------------------------------------------------
# 4–6. Redundância, top-10, grafo e Etapa 15
# ---------------------------------------------------------------------------

def redundancia(pares: dict) -> list[dict]:
    """Rank médio (percentil) e nº no top-10 dos pares de cada tipo no ranking do
    PyGAwLLfs (interacoes.resumo_redundancia), nas duas agregações.

    Usada em: main.
    Importância: a hipótese da redundância (docs/sintetico.md §3) sob as duas leituras.
    """
    resumos = {a: it.resumo_redundancia(p).query("method == 'gawll'").set_index(["model", "tipo"])
               for a, p in pares.items()}
    linhas = []
    for modelo, tipo in resumos["sondadas"].index:
        for coluna, nome in [("rank_medio_percentil", "rank médio (percentil)"),
                             (f"no_top{it.TOP_K_REDUNDANCIA}", "no top-10"),
                             ("n_pares_tipo", "pares com rank")]:
            linhas.append(linha("redundância", modelo, "pares do H", f"{tipo}: {nome}",
                                {a: r.loc[(modelo, tipo), coluna] for a, r in resumos.items()}))
    return linhas


def tops_e_dominio(pares: dict, dominio: pd.DataFrame, cache: dict) -> list[dict]:
    """Top-10 de cada modelo nos pares do H (interacoes.validar_tops_no_dominio): quantos
    pares se mantêm, quantas sementes sondadas em média e a classificação no domínio.

    Usada em: main.
    Importância: o top-10 é o que vai para o texto e para o teste de domínio.
    """
    tops = {a: it.validar_tops_no_dominio(pares[a], dominio, cache).query("method == 'gawll'")
            for a in ["sondadas", "com_zeros"]}  # sondadas primeiro: igual à Etapa 13b
    linhas = []
    for modelo in it.MODELOS_H:
        do_modelo = {a: t[t["model"] == modelo] for a, t in tops.items()}
        chaves = {a: set(zip(t["feature_a"], t["feature_b"])) for a, t in do_modelo.items()}
        linhas.append(linha("top-10 (pares do H)", modelo, "pares do H",
                            "pares que estão no top-10 das duas agregações",
                            {a: len(chaves["com_zeros"] & chaves["sondadas"]) for a in AGREGACOES}))
        linhas.append(linha("top-10 (pares do H)", modelo, "pares do H",
                            "média de sementes sondadas no top-10",
                            {a: t["n_sementes_gawll"].mean() for a, t in do_modelo.items()}))
        for classificacao in sorted(set(tops["sondadas"]["classificacao"])
                                    | set(tops["com_zeros"]["classificacao"])):
            linhas.append(linha("top-10 (pares do H)", modelo, "pares do H",
                                f"domínio: {classificacao}",
                                {a: int((t["classificacao"] == classificacao).sum())
                                 for a, t in do_modelo.items()}))
    return linhas


def grafo(medias: dict, dominio: pd.DataFrame) -> list[dict]:
    """Grafo reduzido (graph.arestas_reduzidas, 20 arestas por modelo): arestas em comum
    entre as agregações e quantas são redundantes ou estruturais.

    Usada em: main.
    Importância: o grafo é a figura que o artigo do PyGAwLLfs apresenta.
    """
    arestas = {a: graph.arestas_reduzidas(m) for a, m in medias.items()}
    tipos = {}
    linhas = []
    for modelo in MODELOS:
        chaves = {a: list(zip(ar.loc[ar["model"] == modelo, "feature_a"],
                              ar.loc[ar["model"] == modelo, "feature_b"]))
                  for a, ar in arestas.items()}
        for chave in set(chaves["com_zeros"]) | set(chaves["sondadas"]):
            if chave not in tipos:
                tipos[chave] = v.tipo_do_par(*chave, dominio)[0]
        linhas.append(linha("grafo reduzido", modelo, "406 pares", "arestas em comum",
                            {a: len(set(chaves["com_zeros"]) & set(chaves["sondadas"]))
                             for a in AGREGACOES}))
        for tipo in ["redundante", "estrutural", "operacional"]:
            linhas.append(linha("grafo reduzido", modelo, "406 pares", f"arestas {tipo}s",
                                {a: sum(tipos[c] == tipo for c in ch) for a, ch in chaves.items()}))
    return linhas


def etapa_15(medias_dominio: dict, dominio: pd.DataFrame, cache: dict) -> tuple[list[dict], pd.DataFrame]:
    """Seleção da Etapa 15 (validate_pairs.selecionar_pares) nas duas agregações, com o
    teste de domínio de cada par; devolve as linhas da tabela e o top-10 de cada modelo
    nas duas agregações.

    Usada em: main.
    Importância: mostra se o número de corroborados e o resultado dos controles
    dependem da leitura do zero.
    """
    rng = np.random.default_rng(v.SEMENTE)
    selecoes, tops = {}, []
    for agregacao in ["sondadas", "com_zeros"]:
        pares = v.selecionar_pares(medias_dominio[agregacao])
        testes = [it.teste_de_dominio(p.feature_a, p.feature_b, dominio, cache, rng)
                  for p in pares.itertuples()]
        selecoes[agregacao] = pares.assign(tipo=[t["tipo"] for t in testes],
                                           classificacao=[t["classificacao"] for t in testes])
        media = medias_dominio[agregacao]
        for modelo in MODELOS:
            top = media[media["model"] == modelo].dropna(subset=["rank"]).nsmallest(TOP_K, "rank")
            for par in top.itertuples():
                teste = it.teste_de_dominio(par.feature_a, par.feature_b, dominio, cache, rng)
                tops.append({"agregacao": agregacao, "model": modelo, "rank": par.rank,
                             "feature_a": par.feature_a, "feature_b": par.feature_b,
                             "interaction": par.interaction, "n_sementes": par.n_sementes,
                             "tipo": teste["tipo"], "classificacao": teste["classificacao"]})
    tops = pd.DataFrame(tops)
    chaves = {a: set(zip(t["model"], t["feature_a"], t["feature_b"]))
              for a, t in tops.groupby("agregacao")}
    outra = {"com_zeros": "sondadas", "sondadas": "com_zeros"}
    tops["no_top10_da_outra"] = [(m, a, b) in chaves[outra[ag]] for ag, m, a, b in
                                 zip(tops["agregacao"], tops["model"], tops["feature_a"],
                                     tops["feature_b"])]

    linhas = []
    for grupo in ["candidato", "controle"]:
        do_grupo = {a: s[s["grupo"] == grupo] for a, s in selecoes.items()}
        linhas.append(linha("Etapa 15", "todos", "406 pares", f"{grupo}s",
                            {a: len(s) for a, s in do_grupo.items()}))
        for classificacao in sorted(set(selecoes["sondadas"]["classificacao"])
                                    | set(selecoes["com_zeros"]["classificacao"])):
            linhas.append(linha("Etapa 15", "todos", "406 pares", f"{grupo}s: {classificacao}",
                                {a: int((s["classificacao"] == classificacao).sum())
                                 for a, s in do_grupo.items()}))
        n_sementes = [f"n_sementes_{m}" for m in MODELOS]
        linhas.append(linha("Etapa 15", "todos", "406 pares",
                            f"{grupo}s: média de sementes sondadas (por modelo)",
                            {a: s[n_sementes].to_numpy().mean() for a, s in do_grupo.items()}))
        linhas.append(linha("Etapa 15", "todos", "406 pares",
                            f"{grupo}s: nunca sondados em nenhum modelo",
                            {a: int((s[n_sementes] == 0).all(axis=1).sum())
                             for a, s in do_grupo.items()}))
    for modelo in MODELOS:
        do_modelo = tops[tops["model"] == modelo]
        linhas.append(linha("Etapa 15", modelo, "406 pares",
                            "pares que estão no top-10 das duas agregações",
                            {a: int(t["no_top10_da_outra"].sum())
                             for a, t in do_modelo.groupby("agregacao")}))
        linhas.append(linha("Etapa 15", modelo, "406 pares", "média de sementes sondadas no top-10",
                            {a: t["n_sementes"].mean() for a, t in do_modelo.groupby("agregacao")}))
        for k in [5, TOP_K]:
            linhas.append(linha("Etapa 15", modelo, "406 pares", f"redundantes no top-{k}",
                                {a: int(((t["rank"] <= k) & (t["tipo"] == "redundante")).sum())
                                 for a, t in do_modelo.groupby("agregacao")}))
    return linhas, tops


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------

def figura(medias: dict, caminho_base: Path) -> None:
    """Para cada modelo: posição mediana do par no ranking (percentil, 0 = mais forte),
    com a faixa interquartil, pelo nº de sementes em que o par foi sondado, nas duas
    agregações.

    Usada em: main.
    Importância: mostra o artefato numa imagem: com os zeros, a posição do par segue
    a cobertura; só com as sondadas, bem menos.
    """
    fig, eixos = plt.subplots(1, len(MODELOS), figsize=(13, 4.8), sharey=True,
                              layout="constrained")
    for ax, modelo in zip(eixos, MODELOS):
        for agregacao in AGREGACOES:
            media = medias[agregacao]
            grupo = media[(media["model"] == modelo) & media["interaction"].notna()].copy()
            grupo["percentil"] = grupo["interaction"].rank(ascending=False, pct=True) * 100
            resumo = grupo.groupby("n_sementes")["percentil"].quantile([0.25, 0.5, 0.75]).unstack()
            cor = COR_AGREGACAO[agregacao]
            ax.fill_between(resumo.index, resumo[0.25], resumo[0.75], color=cor, alpha=0.14,
                            linewidth=0)
            ax.plot(resumo.index, resumo[0.5], color=cor, linewidth=2, marker="o", markersize=6,
                    markeredgecolor="white", markeredgewidth=1.5,
                    label=ROTULO_AGREGACAO[agregacao])
            rho = spearmanr(grupo["interaction"], grupo["n_sementes"]).statistic
            ax.annotate(f"ρ = {decimal(rho)}", (resumo.index[-1], resumo[0.5].iloc[-1]),
                        xytext=(8, -4 if agregacao == "com_zeros" else 6),
                        textcoords="offset points", fontsize=8.5, color=COR_TEXTO_SECUNDARIO)
        contagem = medias["sondadas"].query("model == @modelo")["n_sementes"].value_counts()
        ax.set_xticks(range(6), [f"{k}\n({contagem.get(k, 0)})" for k in range(6)], fontsize=8)
        ax.set_xlim(-0.3, 5.9)
        ax.set_ylim(102, -2)  # 0 = mais forte, no alto
        ax.set_title(ROTULO_MODELO[modelo], loc="left", color=COR_TEXTO, fontsize=11)
        ax.set_xlabel("sementes (de 5) em que o par foi sondado (nº de pares)",
                      color=COR_TEXTO_SECUNDARIO, fontsize=9)
        ax.grid(axis="y", color=COR_GRADE, linewidth=0.6)
        ax.set_axisbelow(True)
        for lado in ["top", "right"]:
            ax.spines[lado].set_visible(False)
        ax.tick_params(colors=COR_TEXTO_SECUNDARIO, labelsize=8)
    eixos[0].set_ylabel("posição no ranking (percentil; 0 = mais forte)",
                        color=COR_TEXTO_SECUNDARIO, fontsize=9)
    alcas, rotulos = eixos[0].get_legend_handles_labels()
    fig.legend(alcas, rotulos, frameon=False, fontsize=8.5, loc="outside upper right", ncol=2)
    fig.suptitle("Com os zeros na média, a posição do par acompanha quantas vezes ele foi sondado",
                 x=0.01, ha="left", color=COR_TEXTO, fontsize=12)
    fig.supxlabel("Linha = mediana; faixa = quartis; ρ = Spearman entre a média do par e o nº de "
                  "sementes sondadas (406 pares por modelo). Só com as sondadas, os pares nunca "
                  "sondados ficam sem média.", x=0.01, ha="left", color=COR_TEXTO_SECUNDARIO,
                  fontsize=8.5)
    salvar(fig, caminho_base)


# ---------------------------------------------------------------------------

def main() -> None:
    """Etapa 13c: passa as duas agregações pelas Etapas 13b e 15 e salva
    sensibilidade_cobertura.csv, sensibilidade_agregacao.csv, sensibilidade_top10.csv
    e a figura.

    Usada em: python src/compare/sensibilidade_agregacao.py; main.py, etapa
    "interacoes"; tests/test_interacao.py.
    Importância: é a análise de sensibilidade da decisão 15: separa as conclusões que
    dependem da leitura do zero das que não dependem.
    """
    interacao = it.carregar_interacoes()
    gawll = interacao[interacao["method"] == "gawll"].drop(columns="par")
    dominio = v.carregar_base()
    medias = {a: graph.interacao_media(gawll, agregacao=a) for a in AGREGACOES}
    h = interacao[interacao["method"] == "friedman_h_abs"]
    pares_h = {m: set(g["par"]) for m, g in h.groupby("model")}

    tabela_cobertura = cobertura(gawll)
    linhas = artefato_de_cobertura(medias, pares_h)
    linhas_concordancia, pares = concordancia(interacao)
    linhas += linhas_concordancia
    split = {a: split_half(gawll, a) for a in AGREGACOES}
    # Referência: o H não tem zero "não sondado", então as duas agregações coincidem
    split_h = split_half(h.drop(columns="par"), "com_zeros")
    linhas += [linha("estabilidade", m, "pares do H", "H_abs, split-half (2 × 3 amostras): spearman",
                     {a: split_h[m] for a in AGREGACOES}) for m in it.MODELOS_H]
    for modelo in MODELOS:
        linhas.append(linha("estabilidade", modelo, "406 pares",
                            "split-half (2 × 3 sementes): spearman",
                            {a: split[a][modelo] for a in AGREGACOES}))
        for fator, sementes in [(2, 5), (4, 10)]:
            linhas.append(linha("estabilidade", modelo, "406 pares",
                                f"confiabilidade da média de {sementes} sementes (Spearman-Brown)",
                                {a: spearman_brown(split[a][modelo], fator) for a in AGREGACOES}))

    tipos = it.tipos_dos_pares(pares["sondadas"].drop_duplicates("par"), dominio)
    for tabela in pares.values():
        tabela["tipo"] = tabela["par"].map(tipos)
    linhas += redundancia(pares)
    cache = it.cache_da_etapa_15()
    linhas += tops_e_dominio(pares, dominio, cache)
    linhas += grafo(medias, dominio)
    linhas_15, tops = etapa_15(medias, dominio, cache)
    linhas += linhas_15

    sensibilidade = pd.DataFrame(linhas)
    sensibilidade["diferenca"] = sensibilidade["sondadas"] - sensibilidade["com_zeros"]
    comum.salvar_tabela(tabela_cobertura.round(3), "sensibilidade_cobertura.csv")
    comum.salvar_tabela(sensibilidade.round(4), "sensibilidade_agregacao.csv")
    comum.salvar_tabela(tops.round(6), "sensibilidade_top10.csv")
    figura(medias, DIR_FIGURAS / "sensibilidade_agregacao")

    with pd.option_context("display.width", 220, "display.max_columns", 20,
                           "display.max_rows", 500, "display.max_colwidth", 60):
        logger.info("Cobertura:\n" + tabela_cobertura.round(2).to_string(index=False))
        logger.info("Com zeros × só sondadas:\n" + sensibilidade.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
