"""Etapa 10 — Diferencial 1: mede o quanto cada método recupera o gabarito.

Recebe as tabelas de aplicar_metodos.py (results/tables/sintetico_*.csv) e
o GABARITO de generate.py. Responde "qual método recupera a verdade?" em
três partes, sempre por (método, modelo) — a comparação é intra-modelo
primeiro (§4.11): a LogReg, linear, não consegue usar x1*x2 em nenhum método.

1. Importância (recuperacao_sintetico.csv), média das sementes. Relevantes =
   diretas + redundantes (redundante não é erro, docs/metodologia.md §5); ruído = x7..x10.
     auc_relevante_ruido    fração dos pares (relevante, ruído) em que a
                            relevante fica à frente no ranking; 1 = separação
                            perfeita, 0,5 = acaso. Não depende de limiar.
     precisao_top6          das 6 primeiras, quantas são relevantes. Com
                            k = nº de relevantes, precision = recall = F1.
     precisao_top4_diretas  das 4 primeiras, quantas são diretas (critério
                            mais duro: põe as diretas à frente das proxies?)
     rank_x1x2, rank_x3x4, rank_x5x6, rank_ruido   rank médio de cada grupo
     → distinção direta × redundante: rank_x3x4 < rank_x5x6.

2. Interação (recuperacao_sintetico_interacao.csv), sobre a matriz MÉDIA das
   sementes (a matriz final do protocolo):
     rank_par_interacao     posição do par (x1, x2) entre os 45 pares (1 = topo)
     rank_pares_redundancia posição média de (x3, x5) e (x4, x6)
     precisao_top3          dos 3 pares do topo, quantos estão em
                            {(x1,x2), (x3,x5), (x4,x6)}
     acerto_por_semente     fração das sementes com (x1, x2) em 1º lugar
   Hipótese central: o PyGAwLLfs põe os pares de redundância no topo (tirar
   uma só quase não custa, tirar as duas custa); o H de Friedman, que mede
   interação na função do modelo, não deveria (entra na Etapa 13).

3. Seleção do PyGAwLLfs (recuperacao_sintetico_selecao.csv): o x* contra o
   conjunto mínimo suficiente, as 4 diretas: precision, recall e F1, mais
   quantas redundantes e quantas de ruído entraram.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.importance.comum import salvar_tabela  # noqa: E402
from src.models.baselines import DIR_TABELAS  # noqa: E402
from src.synthetic.generate import GABARITO, PAR_INTERACAO, PARES_REDUNDANCIA  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DIRETAS = [f for f, c in GABARITO.items() if c == "direta"]
REDUNDANTES = [f for f, c in GABARITO.items() if c == "redundante"]
RUIDO = [f for f, c in GABARITO.items() if c == "ruido"]
RELEVANTES = DIRETAS + REDUNDANTES

# Par como conjunto: (x1, x2) e (x2, x1) são o mesmo par
PAR_VERDADEIRO = frozenset(PAR_INTERACAO)
PARES_HIPOTESE = [frozenset(p) for p in PARES_REDUNDANCIA]


def metricas_importancia(tabela: pd.DataFrame) -> dict:
    """Uma execução (método, modelo, semente) no formato padrão -> métricas."""
    # Ordena pelo rank já calculado (que trata sinal e empates) e desempata
    # pelo nome, para o resultado não depender da ordem das linhas
    ordem = tabela.sort_values(["rank", "feature"])
    rank = dict(zip(ordem["feature"], ordem["rank"]))
    top = list(ordem["feature"])

    comparacoes = [
        1.0 if rank[r] < rank[s] else 0.5 if rank[r] == rank[s] else 0.0
        for r in RELEVANTES for s in RUIDO
    ]
    return {
        "auc_relevante_ruido": float(np.mean(comparacoes)),
        "precisao_top6": len(set(top[:6]) & set(RELEVANTES)) / 6,
        "precisao_top4_diretas": len(set(top[:4]) & set(DIRETAS)) / 4,
        "rank_x1x2": float(np.mean([rank[f] for f in PAR_INTERACAO])),
        "rank_x3x4": float(np.mean([rank["x3"], rank["x4"]])),
        "rank_x5x6": float(np.mean([rank[f] for f in REDUNDANTES])),
        "rank_ruido": float(np.mean([rank[f] for f in RUIDO])),
    }


def ranks_dos_pares(pares: pd.DataFrame) -> dict[frozenset, int]:
    """Tabela longa (feature_a, feature_b, interaction) -> {par: rank}, 1 = mais forte."""
    rank = pares["interaction"].rank(ascending=False, method="min").astype(int)
    return {
        frozenset((a, b)): r
        for a, b, r in zip(pares["feature_a"], pares["feature_b"], rank)
    }


def metricas_interacao(interacao: pd.DataFrame) -> dict:
    """Todas as sementes de um (método, modelo) -> métricas sobre a matriz média."""
    media = (interacao.groupby(["feature_a", "feature_b"], as_index=False)["interaction"]
             .mean())
    rank = ranks_dos_pares(media)
    top3 = sorted(rank, key=rank.get)[:3]

    acertos = [
        ranks_dos_pares(grupo)[PAR_VERDADEIRO] == 1
        for _, grupo in interacao.groupby("seed")
    ]
    return {
        "rank_par_interacao": rank[PAR_VERDADEIRO],
        "rank_pares_redundancia": float(np.mean([rank[p] for p in PARES_HIPOTESE])),
        "precisao_top3": len(set(top3) & {PAR_VERDADEIRO, *PARES_HIPOTESE}) / 3,
        "acerto_por_semente": float(np.mean(acertos)),
        "n_pares": len(rank),
    }


def metricas_selecao(features_selecionadas: str) -> dict:
    """x* de uma execução ("x1;x3;...") contra as 4 diretas."""
    x_estrela = set(features_selecionadas.split(";")) if features_selecionadas else set()
    acertos = len(x_estrela & set(DIRETAS))
    precisao = acertos / len(x_estrela) if x_estrela else 0.0
    recall = acertos / len(DIRETAS)
    f1 = 2 * precisao * recall / (precisao + recall) if acertos else 0.0
    return {
        "precisao": precisao,
        "recall": recall,
        "f1": f1,
        "n_redundantes": len(x_estrela & set(REDUNDANTES)),
        "n_ruido": len(x_estrela & set(RUIDO)),
    }


def main() -> None:
    importancia = pd.read_csv(DIR_TABELAS / "sintetico_importancia.csv")
    interacao = pd.read_csv(DIR_TABELAS / "sintetico_interacao.csv")
    selecao = pd.read_csv(DIR_TABELAS / "sintetico_selecao.csv", keep_default_na=False)

    # 1. Importância: métricas por semente, depois média por (método, modelo)
    por_semente = pd.DataFrame([
        {"method": metodo, "model": modelo, "seed": semente, **metricas_importancia(grupo)}
        for (metodo, modelo, semente), grupo in importancia.groupby(["method", "model", "seed"])
    ])
    resumo_importancia = (por_semente.drop(columns="seed")
                          .groupby(["method", "model"]).mean().round(3).reset_index())
    resumo_importancia["n_sementes"] = (por_semente.groupby(["method", "model"])
                                        .size().to_numpy())

    # 2. Interação: por (método, modelo), sobre a matriz média
    resumo_interacao = pd.DataFrame([
        {"method": metodo, "model": modelo, **metricas_interacao(grupo)}
        for (metodo, modelo), grupo in interacao.groupby(["method", "model"])
    ]).round(3)

    # 3. Seleção do PyGAwLLfs: média das sementes por modelo
    selecao_metricas = pd.DataFrame([
        {"model": modelo, "seed": semente, **metricas_selecao(features)}
        for modelo, semente, features in zip(selecao["model"], selecao["seed"], selecao["features"])
    ])
    resumo_selecao = (selecao_metricas.drop(columns="seed")
                      .groupby("model").mean().round(3).reset_index())

    salvar_tabela(resumo_importancia, "recuperacao_sintetico.csv")
    salvar_tabela(resumo_interacao, "recuperacao_sintetico_interacao.csv")
    salvar_tabela(resumo_selecao, "recuperacao_sintetico_selecao.csv")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        logger.info("Importância (média das sementes):\n" + resumo_importancia.to_string(index=False))
        logger.info("Interação (matriz média):\n" + resumo_interacao.to_string(index=False))
        logger.info("Seleção do PyGAwLLfs contra as 4 diretas:\n" + resumo_selecao.to_string(index=False))


if __name__ == "__main__":
    main()
