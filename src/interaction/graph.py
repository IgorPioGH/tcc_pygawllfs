"""Etapa 13 — grafo de interação reduzido, como no artigo do PyGAwLLfs.

Recebe uma tabela de interação em formato longo (feature_a, feature_b,
interaction, method, model, seed) — a do PyGAwLLfs (interacao_gawll.csv) ou a
do H de Friedman (interacao_h.csv) — e faz duas coisas:
    interacao_media()   média das sementes por (método, modelo, par), com o nº de
                        sementes em que o par foi sondado e o rank (1 = mais forte)
    arestas_reduzidas() as FRACAO_ARESTAS arestas mais fortes de cada
                        (método, modelo): o "grafo reduzido" (5% no artigo)
Agregação das sementes (docs/decisoes.md, decisão 15): no PyGAwLLfs, o vint de
uma execução começa em 0 e só registra o par quando uma sondagem do linkage
learning o sorteia; um zero mistura "par não sondado" com "sondado, efeito
nulo". Os três modelos de uma mesma semente sondam os mesmos pares (o sorteio
usa o gerador random do Python, e o nº de chamadas a ele não depende do
fitness), então o par foi sondado na semente s se teve vint > 0 em algum modelo
nessa semente (marcar_sondagens). A agregação principal ("sondadas") tira a
média só das sementes em que o par foi sondado e só ranqueia pares sondados em
pelo menos MIN_SEMENTES_RANKING sementes. A original ("com_zeros") fica para a
análise de sensibilidade.
Quem consome: friedman_h.py (features do grafo entram no conjunto do H, §4.7),
compare/interacoes.py (PyGAwLLfs × H), compare/sensibilidade_agregacao.py e
domain/validate_pairs.py (Etapa 15).
"""

import pandas as pd

FRACAO_ARESTAS = 0.05
AGREGACOES = ("sondadas", "com_zeros")
# Ranking, top-k e grafo só com pares sondados em pelo menos 2 sementes: uma
# sondagem única guarda o máximo de um retreino só, que pode ser puro ruído
MIN_SEMENTES_RANKING = 2
# Métodos em que interaction = 0 pode querer dizer "par não sondado naquela
# semente". No H de Friedman todo par é calculado, e um zero é uma medida.
METODOS_ZERO_SEM_SONDAGEM = {"gawll"}


def marcar_sondagens(interacao: pd.DataFrame) -> pd.DataFrame:
    """Cópia da tabela com a coluna booleana "sondada": o par foi medido naquela
    (método, modelo, semente)? No PyGAwLLfs, sim se teve vint > 0 em algum dos
    modelos da mesma semente; nos outros métodos, sempre. Precisa da tabela com
    todos os modelos: filtrar um modelo antes cairia na regra "vint > 0".

    Usada em: marcar_nao_sondadas; interacao_media; interacoes.carregar_interacoes;
    sensibilidade_agregacao.
    Importância: separa "não medido" de "medido, efeito nulo" (no XGBoost, ~15 pares
    por execução, como os de periodo_dia, que o modelo não usa).
    """
    if "sondada" in interacao.columns:
        return interacao
    positivo = interacao["interaction"] > 0
    chaves = [interacao[c] for c in ["method", "seed", "feature_a", "feature_b"]]
    em_algum_modelo = positivo.groupby(chaves).transform("any")
    return interacao.assign(
        sondada=em_algum_modelo | ~interacao["method"].isin(METODOS_ZERO_SEM_SONDAGEM))


def marcar_nao_sondadas(interacao: pd.DataFrame) -> pd.DataFrame:
    """Cópia da tabela com interaction = NaN onde o par não foi sondado.

    Usada em: interacao_media; interacoes.comparar_modelo (estabilidade entre
    sementes).
    Importância: um par não medido sai da conta em vez de entrar como "sem interação".
    """
    interacao = marcar_sondagens(interacao)
    return interacao.assign(interaction=interacao["interaction"].where(interacao["sondada"]))


def interacao_media(interacao: pd.DataFrame, agregacao: str = "sondadas") -> pd.DataFrame:
    """Média das sementes por (método, modelo, par), o nº de sementes em que o par foi
    sondado (n_sementes) e o rank do par no grupo (1 = mais forte).
        sondadas    média só das sementes sondadas (NaN se nenhuma); rank só para
                    pares com n_sementes >= MIN_SEMENTES_RANKING (NaN nos demais)
        com_zeros   média das 5 sementes com os zeros; rank para todos os pares
                    (a agregação original, mantida para a análise de sensibilidade)

    Usada em: validate_pairs, friedman_h.conjunto_de_features,
    interacoes.comparar_modelo, sensibilidade_agregacao e testes.
    Importância: o ranking médio de pares é a base do grafo reduzido e das
    comparações de interação; com os zeros, ele acompanha a cobertura das sondagens
    em vez da força da interação (decisão 15).
    """
    if agregacao not in AGREGACOES:
        raise ValueError(f"agregacao deve ser uma de {AGREGACOES}, não {agregacao!r}")
    interacao = marcar_sondagens(interacao)
    valores = marcar_nao_sondadas(interacao) if agregacao == "sondadas" else interacao
    media = (valores.groupby(["method", "model", "feature_a", "feature_b"], as_index=False)
             .agg(interaction=("interaction", "mean"), n_sementes=("sondada", "sum")))
    no_ranking = media["interaction"]
    if agregacao == "sondadas":
        no_ranking = no_ranking.where(media["n_sementes"] >= MIN_SEMENTES_RANKING)
    media["rank"] = (no_ranking.groupby([media["method"], media["model"]])
                     .rank(ascending=False, method="min"))
    return media


def arestas_reduzidas(media: pd.DataFrame, fracao: float = FRACAO_ARESTAS) -> pd.DataFrame:
    """As arestas mais fortes de cada (método, modelo): round(fracao × nº de pares), 5%
    por padrão, como no artigo. Só entram pares com rank (na agregação "sondadas",
    os sondados em pelo menos MIN_SEMENTES_RANKING sementes).

    Usada em: friedman_h.conjunto_de_features; validate_pairs.main;
    sensibilidade_agregacao; tests/test_interacao.py.
    Importância: define o "grafo reduzido" usado no §4.7 e no Diferencial 4.
    """
    grupos = []
    for _, grupo in media.groupby(["method", "model"]):
        n_arestas = int(round(len(grupo) * fracao))
        grupos.append(grupo[grupo["rank"].notna()].nlargest(n_arestas, "interaction"))
    return pd.concat(grupos, ignore_index=True)


def features_do_grafo(arestas: pd.DataFrame) -> set[str]:
    """Todas as features que aparecem em alguma aresta.

    Usada em: friedman_h.conjunto_de_features; tests/test_interacao.py.
    Importância: é a parte (b) da união do §4.7.
    """
    return set(arestas["feature_a"]) | set(arestas["feature_b"])
