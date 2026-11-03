"""Avaliação — grafo de interação reduzido, como no artigo do PyGAwLLfs.

Recebe uma tabela de interação em formato longo (feature_a, feature_b,
interaction, method, model, seed, sondada) — a do PyGAwLLfs (interacao_gawll.csv)
ou a do H de Friedman (interacao_h.csv) — e faz três coisas:
    interacao_media()      média das sementes por (método, modelo, par), com o nº de
                           sementes em que o par foi sondado e o rank (1 = mais forte)
    arestas_reduzidas()    as FRACAO_ARESTAS arestas mais fortes de cada
                           (método, modelo): o "grafo reduzido" (5% no artigo)
    tipo_por_construcao()  o tipo de cada par, lido da definição dos atributos
Agregação das sementes (docs/decisoes.md, decisão 13): no PyGAwLLfs, o vint de
uma execução começa em 0 e só registra o par quando uma sondagem do linkage
learning o sorteia; um zero mistura "par não sondado" com "sondado, efeito
nulo". A coluna "sondada", gravada pelo próprio GA (contador de sondagens),
separa os dois casos. A média de um par usa só as sementes em que ele foi
sondado, e só entram no ranking os pares sondados em pelo menos
MIN_SEMENTES_RANKING sementes. No H de Friedman todo par é calculado
(sondada = True sempre).
Quem consome: compare/interacoes.py (PyGAwLLfs × H) e
interpretacao/figura_grafo.py.
"""

import pandas as pd

FRACAO_ARESTAS = 0.05
# Ranking, top-k e grafo só com pares sondados em pelo menos 2 sementes: uma
# sondagem única guarda o máximo de um retreino só, que pode ser puro ruído
MIN_SEMENTES_RANKING = 2

# Pares em que um atributo é função do outro, pela forma como features.py os cria
PARES_REDUNDANTES = {frozenset(par) for par in [
    ("hora", "periodo_dia"),      # periodo_dia = hora // 6
    ("dia_sem", "fim_semana"),    # fim_semana = dia_sem >= 5
    ("mes", "trimestre"), ("mes", "dia_ano"),
    ("trimestre", "dia_ano"), ("dia_mes", "dia_ano"),
]}
COR_TIPO = {"operacional": "#2a78d6", "redundante": "#eb6834", "estrutural": "#8c8b85"}


def variavel_bruta(feature: str) -> str:
    """Coluna do VRA de onde a feature sai: as dummies de tipo de linha vêm do tipo de
    linha; as de porte e o target encoding do aeroporto vêm do aeroporto (origem ou
    destino); as demais são a própria feature.

    Usada em: tipo_por_construcao.
    Importância: duas features da mesma coluna bruta não interagem no sentido
    operacional; elas descrevem a mesma coisa.
    """
    if feature.startswith("tipo_linha_"):
        return "tipo_linha"
    if feature.startswith("porte_origem_") or feature == "origem_te":
        return "origem"
    if feature.startswith("porte_destino_") or feature == "destino_te":
        return "destino"
    return feature


def tipo_por_construcao(feature_a: str, feature_b: str) -> str:
    """Tipo do par, pela definição dos atributos (nada é estimado nos dados):
        estrutural   as duas codificam a mesma coluna bruta (ex.: duas dummies de porte)
        redundante   uma é função da outra (PARES_REDUNDANTES; ex.: mês e trimestre)
        operacional  os demais

    Usada em: interacoes.tipos_dos_pares; figura_grafo.main; tests/test_interacao.py.
    Importância: separa redundância de interação operacional, a hipótese central
    sobre o que o PyGAwLLfs detecta.
    """
    if variavel_bruta(feature_a) == variavel_bruta(feature_b):
        return "estrutural"
    if frozenset((feature_a, feature_b)) in PARES_REDUNDANTES:
        return "redundante"
    return "operacional"


def marcar_nao_sondadas(interacao: pd.DataFrame) -> pd.DataFrame:
    """Cópia da tabela com interaction = NaN onde o par não foi sondado (coluna
    "sondada"). Levanta erro se a coluna faltar.

    Usada em: interacao_media; interacoes.comparar_modelo (estabilidade entre
    sementes).
    Importância: um par não medido sai da conta em vez de entrar como "sem interação".
    """
    if "sondada" not in interacao.columns:
        raise ValueError("tabela sem 'sondada': rode consolidar_gawll / friedman_h de novo")
    return interacao.assign(interaction=interacao["interaction"].where(interacao["sondada"]))


def interacao_media(interacao: pd.DataFrame) -> pd.DataFrame:
    """Média das sementes por (método, modelo, par), só nas sementes em que o par foi
    sondado (NaN se nenhuma), o nº dessas sementes (n_sementes) e o rank do par no
    grupo (1 = mais forte), só para pares com n_sementes >= MIN_SEMENTES_RANKING.

    Usada em: interacoes.comparar_modelo, figura_grafo.main e testes.
    Importância: o ranking médio de pares é a base do grafo reduzido e das
    comparações de interação.
    """
    valores = marcar_nao_sondadas(interacao)
    media = (valores.groupby(["method", "model", "feature_a", "feature_b"], as_index=False)
             .agg(interaction=("interaction", "mean"), n_sementes=("sondada", "sum")))
    no_ranking = media["interaction"].where(media["n_sementes"] >= MIN_SEMENTES_RANKING)
    media["rank"] = (no_ranking.groupby([media["method"], media["model"]])
                     .rank(ascending=False, method="min"))
    return media


def arestas_reduzidas(media: pd.DataFrame, fracao: float = FRACAO_ARESTAS) -> pd.DataFrame:
    """As arestas mais fortes de cada (método, modelo): round(fracao × nº de pares), 5%
    por padrão, como no artigo. Só entram pares com rank (sondados em pelo menos
    MIN_SEMENTES_RANKING sementes).

    Usada em: figura_grafo.main; tests/test_interacao.py.
    Importância: define o "grafo reduzido" do artigo.
    """
    grupos = []
    for _, grupo in media.groupby(["method", "model"]):
        n_arestas = int(round(len(grupo) * fracao))
        grupos.append(grupo[grupo["rank"].notna()].nlargest(n_arestas, "interaction"))
    return pd.concat(grupos, ignore_index=True)
