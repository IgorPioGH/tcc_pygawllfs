"""Etapa 11 — concordância entre os rankings de importância.

Recebe as quatro tabelas da Etapa 9 (results/tables/importancia_*.csv, no
formato padrão feature,importance,rank,method,model,seed) e faz três coisas:

1. Consolida cada (método, modelo) num ranking só, pela média da importância
   nas sementes — o pipeline original do PyGAwLLfs também tira a média do
   vimp entre execuções. A variação entre sementes fica para o stability.py.
   → ranking_consolidado.csv (com a magnitude normalizada, só para figuras, §4.8)

2. Compara todos os pares de rankings (11 rankings → 55 pares) com as métricas
   do §4.5: Spearman, Kendall tau-b, RBO (p = 0,9) e sobreposição top-5/10/15.
   A comparação é por rank, nunca por valor: as escalas dos métodos não se
   comparam (§4.8).
   → concordancia.csv

3. Classifica cada par para separar os dois efeitos do §4.11:
     mesmo_modelo   métodos diferentes sobre o mesmo modelo (comparação primária)
     mesmo_metodo   o mesmo método sobre modelos diferentes
     diferentes     método e modelo diferentes
   → concordancia_resumo.csv, que responde: "os métodos concordam mais entre
     si quando aplicados ao mesmo modelo do que a modelos diferentes?"

Quem consome: src/compare/figuras.py (heatmaps), Etapa 12 (top-k de cada
método, em ranking_consolidado.csv), Etapa 13 (top-15 por consenso) e o Cap. 4.
"""

import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.importance import comum  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

METODOS = ["intrinseca", "permutation", "shap", "gawll"]
MODELOS = ["logreg", "random_forest", "xgboost"]
P_RBO = 0.9
TOP_K = [5, 10, 15]


# ---------------------------------------------------------------------------
# Pontuação e normalização
# ---------------------------------------------------------------------------

def pontuacao(tabela: pd.DataFrame) -> pd.Series:
    """Valor que ordena o ranking: a importância, exceto no coeficiente da
    LogReg, em que o que importa é o tamanho, não o sinal (mesma regra do
    rank_por_absoluto em comum.tabela_importancia)."""
    coeficiente = (tabela["method"] == "intrinseca") & (tabela["model"] == "logreg")
    return tabela["importance"].where(~coeficiente, tabela["importance"].abs())


def normalizar_magnitude(valores: pd.Series) -> pd.Series:
    """Para figuras de magnitude (§4.8): negativos viram 0 ("sem evidência de
    importância"; só a Permutation tem) e depois min-max para [0, 1].
    Nunca soma-1, porque há valores com sinal."""
    valores = valores.clip(lower=0)
    amplitude = valores.max() - valores.min()
    if amplitude == 0:
        return valores * 0.0
    return (valores - valores.min()) / amplitude


def ordenar(pontuacoes: pd.Series) -> list[str]:
    """Features da mais para a menos importante. Empate: ordem alfabética,
    para o resultado não depender da ordem das linhas."""
    tabela = pontuacoes.rename("valor").reset_index()
    tabela.columns = ["feature", "valor"]
    return list(tabela.sort_values(["valor", "feature"], ascending=[False, True])["feature"])


# ---------------------------------------------------------------------------
# Métricas de concordância
# ---------------------------------------------------------------------------

def rbo(lista_a: list[str], lista_b: list[str], p: float = P_RBO) -> float:
    """Rank-Biased Overlap estendido (RBO_ext; Webber, Moffat e Zobel, 2010, eq. 32).

    Em cada profundidade d, mede a sobreposição dos d primeiros itens das duas
    listas (X_d). A média é ponderada por p^d, então o topo pesa mais:
    com p = 0,9, as 10 primeiras posições respondem por ~86% do valor.
        RBO_ext = (X_k / k) * p^k + (1 - p) / p * soma_{d=1..k} (X_d / d) * p^d
    Vale 1 para listas iguais e fica perto de 0 quando os topos não se cruzam.
    """
    k = min(len(lista_a), len(lista_b))
    vistos_a, vistos_b = set(), set()
    soma = 0.0
    sobreposicao = 0
    for d in range(1, k + 1):
        vistos_a.add(lista_a[d - 1])
        vistos_b.add(lista_b[d - 1])
        sobreposicao = len(vistos_a & vistos_b)
        soma += (sobreposicao / d) * p**d
    return (sobreposicao / k) * p**k + (1 - p) / p * soma


def sobreposicao_top_k(lista_a: list[str], lista_b: list[str], k: int) -> float:
    """Fração das k primeiras de uma lista que também estão nas k primeiras da outra."""
    return len(set(lista_a[:k]) & set(lista_b[:k])) / k


def comparar(pontuacoes_a: pd.Series, pontuacoes_b: pd.Series) -> dict:
    """Todas as métricas entre dois rankings (Series indexadas pela feature)."""
    pontuacoes_b = pontuacoes_b.loc[pontuacoes_a.index]  # mesma ordem de features
    lista_a, lista_b = ordenar(pontuacoes_a), ordenar(pontuacoes_b)
    resultado = {
        "spearman": spearmanr(pontuacoes_a, pontuacoes_b).statistic,
        # scipy usa tau-b por padrão; explícito porque empates importam (§4.5)
        "kendall_tau_b": kendalltau(pontuacoes_a, pontuacoes_b, variant="b").statistic,
        "rbo": rbo(lista_a, lista_b),
    }
    for k in TOP_K:
        resultado[f"top{k}"] = sobreposicao_top_k(lista_a, lista_b, k)
    return resultado


def tipo_de_par(chave_a: tuple[str, str], chave_b: tuple[str, str]) -> str:
    """Chaves (método, modelo) -> mesmo_modelo, mesmo_metodo ou diferentes."""
    (metodo_a, modelo_a), (metodo_b, modelo_b) = chave_a, chave_b
    if modelo_a == modelo_b:
        return "mesmo_modelo"
    if metodo_a == metodo_b:
        return "mesmo_metodo"
    return "diferentes"


# ---------------------------------------------------------------------------
# Etapas
# ---------------------------------------------------------------------------

def carregar_importancias() -> pd.DataFrame:
    """As quatro tabelas da Etapa 9 juntas, com a coluna 'pontuacao'."""
    tabelas = [pd.read_csv(comum.DIR_TABELAS / f"importancia_{m}.csv") for m in METODOS]
    importancias = pd.concat(tabelas, ignore_index=True)
    importancias["pontuacao"] = pontuacao(importancias)
    return importancias


def consolidar_rankings(importancias: pd.DataFrame) -> pd.DataFrame:
    """Um ranking por (método, modelo): média das sementes."""
    consolidado = (importancias
                   .groupby(["method", "model", "feature"], as_index=False)
                   .agg(importance=("importance", "mean"),
                        pontuacao=("pontuacao", "mean"),
                        n_sementes=("seed", "nunique")))
    grupos = consolidado.groupby(["method", "model"])
    consolidado["rank"] = (grupos["pontuacao"]
                           .rank(ascending=False, method="min").astype(int))
    consolidado["importance_norm"] = grupos["pontuacao"].transform(normalizar_magnitude)
    return consolidado.sort_values(["model", "method", "rank"]).reset_index(drop=True)


def concordancia_entre_rankings(consolidado: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por par de rankings (método, modelo) × (método, modelo)."""
    pontuacoes = {
        chave: grupo.set_index("feature")["pontuacao"]
        for chave, grupo in consolidado.groupby(["method", "model"])
    }
    # Ordem fixa (modelo, depois método), para a tabela ser fácil de ler
    chaves = [(metodo, modelo) for modelo in MODELOS for metodo in METODOS
              if (metodo, modelo) in pontuacoes]

    linhas = []
    for chave_a, chave_b in combinations(chaves, 2):
        linhas.append({
            "method_a": chave_a[0], "model_a": chave_a[1],
            "method_b": chave_b[0], "model_b": chave_b[1],
            "tipo_par": tipo_de_par(chave_a, chave_b),
            **comparar(pontuacoes[chave_a], pontuacoes[chave_b]),
        })
    return pd.DataFrame(linhas)


def resumir(concordancia: pd.DataFrame) -> pd.DataFrame:
    """Média (e extremos do Spearman) por tipo de par."""
    metricas = ["spearman", "kendall_tau_b", "rbo"] + [f"top{k}" for k in TOP_K]
    resumo = concordancia.groupby("tipo_par")[metricas].mean()
    resumo["spearman_min"] = concordancia.groupby("tipo_par")["spearman"].min()
    resumo["spearman_max"] = concordancia.groupby("tipo_par")["spearman"].max()
    resumo["n_pares"] = concordancia.groupby("tipo_par").size()
    ordem = ["mesmo_modelo", "mesmo_metodo", "diferentes"]
    return resumo.reindex(ordem).reset_index()


def main() -> None:
    importancias = carregar_importancias()
    consolidado = consolidar_rankings(importancias)
    concordancia = concordancia_entre_rankings(consolidado)
    resumo = resumir(concordancia)

    comum.salvar_tabela(consolidado, "ranking_consolidado.csv")
    comum.salvar_tabela(concordancia, "concordancia.csv")
    comum.salvar_tabela(resumo, "concordancia_resumo.csv")

    colunas = ["method_a", "method_b", "spearman", "kendall_tau_b", "rbo", "top5", "top10"]
    with pd.option_context("display.width", 200):
        for modelo in MODELOS:
            intra = concordancia[(concordancia["tipo_par"] == "mesmo_modelo")
                                 & (concordancia["model_a"] == modelo)]
            logger.info(f"{modelo} — métodos entre si:\n"
                        + intra[colunas].round(3).to_string(index=False))
        logger.info("Resumo por tipo de par:\n" + resumo.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
