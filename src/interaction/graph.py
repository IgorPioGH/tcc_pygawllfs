"""Etapa 13 — grafo de interação reduzido, como no artigo do PyGAwLLfs.

Recebe uma tabela de interação em formato longo (feature_a, feature_b,
interaction, method, model, seed) — a do PyGAwLLfs (interacao_gawll.csv) ou a
do H de Friedman (interacao_h.csv) — e faz duas coisas:
    interacao_media()   média das sementes por (método, modelo, par), com o rank
                        de cada par (1 = interação mais forte)
    arestas_reduzidas() as FRACAO_ARESTAS arestas mais fortes de cada
                        (método, modelo): o "grafo reduzido" (5% no artigo)
Quem consome: friedman_h.py (features do grafo entram no conjunto do H, §4.7),
compare/interacoes.py (PyGAwLLfs × H) e domain/validate_pairs.py (Etapa 15).
"""

import pandas as pd

FRACAO_ARESTAS = 0.05


def interacao_media(interacao: pd.DataFrame) -> pd.DataFrame:
    """Média das sementes por (método, modelo, par) e o rank do par no grupo (1 = mais
    forte).

    Usada em: validate_pairs, friedman_h.conjunto_de_features,
    interacoes.comparar_modelo e testes.
    Importância: o ranking médio de pares é a base do grafo reduzido e das
    comparações de interação.
    """
    media = (interacao.groupby(["method", "model", "feature_a", "feature_b"], as_index=False)
             ["interaction"].mean())
    media["rank"] = (media.groupby(["method", "model"])["interaction"]
                     .rank(ascending=False, method="min").astype(int))
    return media


def arestas_reduzidas(media: pd.DataFrame, fracao: float = FRACAO_ARESTAS) -> pd.DataFrame:
    """As arestas mais fortes de cada (método, modelo): round(fracao × nº de pares), 5%
    por padrão, como no artigo.

    Usada em: friedman_h.conjunto_de_features; tests/test_interacao.py.
    Importância: define o "grafo reduzido" usado no §4.7 e no Diferencial 4.
    """
    grupos = []
    for _, grupo in media.groupby(["method", "model"]):
        n_arestas = int(round(len(grupo) * fracao))
        grupos.append(grupo.nlargest(n_arestas, "interaction"))
    return pd.concat(grupos, ignore_index=True)


def features_do_grafo(arestas: pd.DataFrame) -> set[str]:
    """Todas as features que aparecem em alguma aresta.

    Usada em: friedman_h.conjunto_de_features; tests/test_interacao.py.
    Importância: é a parte (b) da união do §4.7.
    """
    return set(arestas["feature_a"]) | set(arestas["feature_b"])
