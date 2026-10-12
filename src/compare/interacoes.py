"""Etapa 13 — PyGAwLLfs × H de Friedman: os dois concordam sobre quais pares interagem?

Recebe interacao_gawll.csv (vint, 5 execuções) e interacao_h.csv (H, 5
amostras), e classifica cada par com a mesma função da Etapa 15
(domain/validate_pairs.tipo_do_par: estrutural, redundante ou operacional).
Para o RF e o XGBoost (a LogReg tem H = 0 por construção):

1. Média das sementes de cada método, restrita aos pares em que o H foi
   calculado (o conjunto do §4.7).
2. Concordância entre os rankings de pares (Spearman, Kendall tau-b, RBO,
   top-5/10/15): PyGAwLLfs × H_abs (a comparação principal), PyGAwLLfs × H
   normalizado e H_abs × H normalizado.
3. Estabilidade entre sementes de cada método, nos mesmos pares.
4. Hipótese da redundância (docs/sintetico.md §3): onde os pares
   redundantes caem em cada ranking — rank médio (em percentil) e quantos
   entram no top-10.
5. Validação de domínio dos dois tops: o top-10 de cada método passa pelo
   mesmo teste descritivo da Etapa 15. Pares já testados lá têm o resultado
   reaproveitado de dominio_pares.csv (mesmos números nos dois documentos).

Produz:
    results/tables/interacoes.csv        concordância e estabilidade, por modelo
    results/tables/interacoes_redundancia.csv  posição dos pares de cada tipo
    results/tables/interacoes_pares.csv  cada par: valor e rank em cada método, tipo
    results/tables/interacoes_dominio.csv      top-10 de cada método no teste de domínio
    results/figures/interacao_gawll_x_h.{png,pdf}
Só lê CSVs e, para o tipo dos pares, treino + validação (nunca o teste).
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

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.compare.figuras import COR_GRADE, COR_TEXTO, COR_TEXTO_SECUNDARIO, ROTULO_MODELO, salvar  # noqa: E402
from src.compare.rankings import comparar  # noqa: E402
from src.compare.stability import concordancia_entre_sementes  # noqa: E402
from src.domain.validate_pairs import (  # noqa: E402
    COR_TIPO,
    SEMENTE,
    carregar_base,
    classificar,
    testar_par,
    tipo_do_par,
    variavel,
)
from src.importance import comum  # noqa: E402
from src.interaction.graph import interacao_media  # noqa: E402
from src.interaction.friedman_h import MODELOS_H  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DIR_FIGURAS = RAIZ / "results" / "figures"
METODOS = ["gawll", "friedman_h_abs", "friedman_h"]
ROTULO = {"gawll": "PyGAwLLfs", "friedman_h_abs": "H não normalizado",
          "friedman_h": "H normalizado"}
TOP_K_REDUNDANCIA = 10
TOP_K_DOMINIO = 10


def carregar_interacoes() -> pd.DataFrame:
    """Junta interacao_gawll.csv e interacao_h.csv, com a chave do par ("a|b", na mesma
    ordem nas duas).

    Usada em: main.
    Importância: a chave comum é o que permite comparar par a par o PyGAwLLfs e o H.
    """
    interacao = pd.concat([pd.read_csv(comum.DIR_TABELAS / "interacao_gawll.csv"),
                           pd.read_csv(comum.DIR_TABELAS / "interacao_h.csv")],
                          ignore_index=True)
    interacao["par"] = interacao["feature_a"] + "|" + interacao["feature_b"]
    return interacao


def tipos_dos_pares(pares: pd.DataFrame, dominio: pd.DataFrame) -> dict[str, str]:
    """Tipo de cada par (estrutural, redundante ou operacional), via
    validate_pairs.tipo_do_par, com cache porque várias dummies caem na mesma
    variável.

    Usada em: main.
    Importância: separa redundância de interação operacional, a hipótese central
    sobre o que o PyGAwLLfs detecta.
    """
    cache, tipos = {}, {}
    for a, b, par in zip(pares["feature_a"], pares["feature_b"], pares["par"]):
        if (a, b) not in cache:
            cache[(a, b)] = tipo_do_par(a, b, dominio)[0]
        tipos[par] = cache[(a, b)]
    return tipos


def comparar_modelo(interacao: pd.DataFrame, modelo: str) -> tuple[list[dict], pd.DataFrame]:
    """Para um modelo: concordância PyGAwLLfs × H nos pares em que o H existe, e
    estabilidade de cada método entre sementes.

    Usada em: main.
    Importância: produz o resultado principal da Etapa 13 (os dois métodos apontam as
    mesmas interações?).
    """
    do_modelo = interacao[interacao["model"] == modelo]
    pares_h = set(do_modelo.loc[do_modelo["method"] == "friedman_h_abs", "par"])
    do_modelo = do_modelo[do_modelo["par"].isin(pares_h)]

    media = interacao_media(do_modelo.drop(columns="par"))
    media["par"] = media["feature_a"] + "|" + media["feature_b"]
    valores = {m: media[media["method"] == m].set_index("par")["interaction"] for m in METODOS}

    linhas = []
    for metodo_a, metodo_b in combinations(METODOS, 2):
        linhas.append({"model": modelo, "comparacao": f"{metodo_a} x {metodo_b}",
                       "n_pares": len(pares_h),
                       **comparar(valores[metodo_a], valores[metodo_b])})
    for metodo in METODOS:
        grupo = (do_modelo[do_modelo["method"] == metodo]
                 .rename(columns={"par": "feature", "interaction": "pontuacao"}))
        linhas.append({"model": modelo, "comparacao": f"estabilidade {metodo}",
                       "n_pares": len(pares_h),
                       **{k: v for k, v in concordancia_entre_sementes(grupo).items()
                          if not k.endswith("_min") and not k.startswith("n_")}})

    # Uma linha por par: valor e rank (1 = mais forte) em cada método
    por_par = media.pivot_table(index=["par", "feature_a", "feature_b"], columns="method",
                                values=["interaction", "rank"])
    por_par.columns = [f"{coluna}_{metodo}" for coluna, metodo in por_par.columns]
    return linhas, por_par.reset_index().assign(model=modelo)


def resumo_redundancia(pares: pd.DataFrame) -> pd.DataFrame:
    """Para cada (modelo, método, tipo de par): rank médio em percentil e quantos pares
    do tipo estão no top-10.

    Usada em: main.
    Importância: testa nos dados reais se o PyGAwLLfs põe pares redundantes no topo,
    como no sintético.
    """
    linhas = []
    for modelo, grupo in pares.groupby("model"):
        n = len(grupo)
        for metodo in METODOS:
            for tipo, do_tipo in grupo.groupby("tipo"):
                ranks = do_tipo[f"rank_{metodo}"]
                linhas.append({
                    "model": modelo, "method": metodo, "tipo": tipo, "n_pares_tipo": len(do_tipo),
                    "rank_medio_percentil": float((ranks / n).mean() * 100),
                    f"no_top{TOP_K_REDUNDANCIA}": int((ranks <= TOP_K_REDUNDANCIA).sum()),
                })
    return pd.DataFrame(linhas)


def validar_tops_no_dominio(pares: pd.DataFrame, dominio: pd.DataFrame) -> pd.DataFrame:
    """Top-10 de pares do PyGAwLLfs e do H_abs, por modelo, passados pelo teste
    descritivo da Etapa 15 (validate_pairs.testar_par e classificar).

    Usada em: main.
    Importância: julga os dois métodos contra os dados, não só um contra o outro.
    """
    ja_testados = pd.read_csv(comum.DIR_TABELAS / "dominio_pares.csv")
    ja_testados = ja_testados.set_index(["feature_a", "feature_b"])
    rng = np.random.default_rng(SEMENTE)
    cache, linhas = {}, []
    for modelo, grupo in pares.groupby("model"):
        for metodo in ["gawll", "friedman_h_abs"]:
            for par in grupo.nsmallest(TOP_K_DOMINIO, f"rank_{metodo}").itertuples():
                chave = (par.feature_a, par.feature_b)
                if chave not in cache:
                    if chave in ja_testados.index:
                        cache[chave] = ja_testados.loc[chave].to_dict()
                    else:
                        tipo, incerteza = tipo_do_par(*chave, dominio)
                        resultado = {"tipo": tipo, "coef_incerteza": incerteza}
                        if tipo != "estrutural":
                            resultado.update(testar_par(dominio, variavel(par.feature_a),
                                                        variavel(par.feature_b), rng)[0])
                        resultado["classificacao"] = classificar(tipo, resultado)
                        cache[chave] = resultado
                teste = cache[chave]
                linhas.append({
                    "model": modelo, "method": metodo, "rank": getattr(par, f"rank_{metodo}"),
                    "feature_a": par.feature_a, "feature_b": par.feature_b,
                    "tipo": teste["tipo"], "desvio_2024_pp": teste.get("desvio_2024_pp"),
                    "desvio_2025_pp": teste.get("desvio_2025_pp"),
                    "replicacao_anos": teste.get("replicacao_anos"),
                    "classificacao": teste["classificacao"],
                })
    return pd.DataFrame(linhas)


def figura_gawll_x_h(pares: pd.DataFrame, caminho_base: Path) -> None:
    """Rank no PyGAwLLfs (x) × rank no H não normalizado (y), cor = tipo do par.

    Usada em: main.
    Importância: mostra a falta de concordância e onde ficam os pares redundantes.
    """
    fig, eixos = plt.subplots(1, len(MODELOS_H), figsize=(12, 5.6), layout="constrained")
    for ax, modelo in zip(eixos, MODELOS_H):
        grupo = pares[pares["model"] == modelo]
        n = len(grupo)
        ax.plot([1, n], [1, n], color=COR_GRADE, linewidth=1, zorder=0)
        # Operacionais primeiro (fundo); redundantes e estruturais por cima, em destaque
        for tipo in ["operacional", "estrutural", "redundante"]:
            do_tipo = grupo[grupo["tipo"] == tipo]
            ax.scatter(do_tipo["rank_gawll"], do_tipo["rank_friedman_h_abs"],
                       s=22 if tipo == "operacional" else 60, color=COR_TIPO[tipo],
                       alpha=0.55 if tipo == "operacional" else 0.95,
                       edgecolors="white", linewidths=0.8, label=tipo, zorder=2)
        # Rótulo só nos redundantes do top-30 de algum dos dois (os que importam)
        redundantes = grupo[(grupo["tipo"] == "redundante")
                            & ((grupo["rank_gawll"] <= 30) | (grupo["rank_friedman_h_abs"] <= 30))]
        for linha in redundantes.itertuples():
            ax.annotate(f"{linha.feature_a} × {linha.feature_b}",
                        (linha.rank_gawll, linha.rank_friedman_h_abs),
                        xytext=(6, -3), textcoords="offset points",
                        fontsize=7, color=COR_TEXTO_SECUNDARIO)
        ax.set_xlim(0, n + 5)
        ax.set_ylim(0, n + 5)
        ax.invert_xaxis()
        ax.invert_yaxis()  # rank 1 (mais forte) no canto superior direito
        ax.set_xlabel("rank no PyGAwLLfs (vint)", color=COR_TEXTO_SECUNDARIO, fontsize=9)
        ax.set_ylabel("rank no H de Friedman (não normalizado)", color=COR_TEXTO_SECUNDARIO,
                      fontsize=9)
        ax.set_title(f"{ROTULO_MODELO[modelo]} — {n} pares", loc="left", color=COR_TEXTO, fontsize=11)
        ax.grid(color=COR_GRADE, linewidth=0.6)
        ax.set_axisbelow(True)
        for lado in ["top", "right"]:
            ax.spines[lado].set_visible(False)
        ax.tick_params(colors=COR_TEXTO_SECUNDARIO, labelsize=8)
    alcas, rotulos = eixos[-1].get_legend_handles_labels()
    fig.legend(alcas, rotulos, frameon=False, fontsize=8.5, title="tipo do par",
               title_fontsize=8.5, loc="outside upper right", ncol=3)
    fig.suptitle("Pares de features: interação segundo o PyGAwLLfs × segundo o H de Friedman",
                 x=0.01, ha="left", color=COR_TEXTO, fontsize=12)
    fig.supxlabel("Canto superior direito = forte nos dois; a diagonal é a concordância perfeita. "
                  "Rotulados: pares redundantes no top-30 de algum dos dois métodos (média das "
                  "5 sementes).", x=0.01, ha="left", color=COR_TEXTO_SECUNDARIO, fontsize=8.5)
    salvar(fig, caminho_base)


def main() -> None:
    """Etapa 13b: carrega as interações, compara por modelo, resume a redundância,
    valida os tops no domínio e salva interacoes*.csv e a figura. Roda depois da
    Etapa 15.

    Usada em: python src/compare/interacoes.py; main.py, etapa "interacoes";
    tests/test_interacao.py.
    Importância: responde se o PyGAwLLfs e o H de Friedman apontam as mesmas
    interações.
    """
    interacao = carregar_interacoes()
    dominio = carregar_base()

    linhas, tabelas_pares = [], []
    for modelo in MODELOS_H:
        linhas_modelo, por_par = comparar_modelo(interacao, modelo)
        linhas.extend(linhas_modelo)
        tabelas_pares.append(por_par)
    pares = pd.concat(tabelas_pares, ignore_index=True)
    tipos = tipos_dos_pares(pares.drop_duplicates("par"), dominio)
    pares["tipo"] = pares["par"].map(tipos)

    concordancia = pd.DataFrame(linhas)
    redundancia = resumo_redundancia(pares)
    dominio_tops = validar_tops_no_dominio(pares, dominio)
    comum.salvar_tabela(concordancia.round(4), "interacoes.csv")
    comum.salvar_tabela(redundancia.round(2), "interacoes_redundancia.csv")
    comum.salvar_tabela(pares.drop(columns="par").round(6), "interacoes_pares.csv")
    comum.salvar_tabela(dominio_tops.round(4), "interacoes_dominio.csv")
    figura_gawll_x_h(pares, DIR_FIGURAS / "interacao_gawll_x_h")

    with pd.option_context("display.width", 220, "display.max_columns", 20):
        logger.info("Concordância e estabilidade:\n"
                    + concordancia[["model", "comparacao", "n_pares", "spearman", "rbo",
                                    "top5", "top10"]].round(3).to_string(index=False))
        logger.info("Onde caem os pares de cada tipo:\n" + redundancia.round(1).to_string(index=False))
        logger.info("Top-10 de cada método no teste de domínio:\n"
                    + dominio_tops.groupby(["model", "method", "classificacao"]).size()
                    .to_string())


if __name__ == "__main__":
    main()
