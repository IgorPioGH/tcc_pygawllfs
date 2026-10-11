"""Etapa 9 — consolida as 15 execuções do PyGAwLLfs em tabelas.

Recebe os checkpoints .npz da Etapa 8 (experiments/runs/gawll/) e produz:
    results/tables/importancia_gawll.csv  vimp no formato padrão (§7)
    results/tables/interacao_gawll.csv    vint em formato longo, um par (i < j)
                                          por linha: feature_a, feature_b,
                                          interaction, method, model, seed
    results/tables/selecao_gawll.csv      o subconjunto x* de cada execução
    custo_importancia.csv                 tempo e nº de treinos de cada execução

Cada execução vira uma "semente" das tabelas. A média entre as 5 sementes
fica para quem consome: Etapa 11 (rankings e estabilidade), Etapa 12
(fidelidade do x*), Etapa 13 e Diferencial 4 (interações).

A função consolidar() também é usada pelo Diferencial 1
(src/synthetic/aplicar_metodos.py), sobre os checkpoints do dataset sintético.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.importance.comum import (  # noqa: E402
    salvar_custo,
    salvar_tabela,
    tabela_importancia,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

METODO = "gawll"
DIR_EXECUCOES = RAIZ / "experiments" / "runs" / "gawll"
COLUNAS_INTERACAO = ["feature_a", "feature_b", "interaction", "method", "model", "seed"]


def tabela_interacao(
    features: list[str], matriz: np.ndarray, method: str, model: str, seed: int
) -> pd.DataFrame:
    """Matriz simétrica N x N -> uma linha por par (i < j)."""
    linhas_i, colunas_j = np.triu_indices(len(features), k=1)
    return pd.DataFrame({
        "feature_a": [features[i] for i in linhas_i],
        "feature_b": [features[j] for j in colunas_j],
        "interaction": matriz[linhas_i, colunas_j],
        "method": method,
        "model": model,
        "seed": seed,
    })


def consolidar(dir_execucoes: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    """Lê todos os .npz de dir_execucoes e devolve (importância, interação, seleção, custos)."""
    arquivos = sorted(dir_execucoes.glob("*_seed*.npz"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum checkpoint em {dir_execucoes}")

    importancias, interacoes, selecoes, custos = [], [], [], []
    for arquivo in arquivos:
        execucao = np.load(arquivo)
        features = [str(f) for f in execucao["features"]]
        modelo = str(execucao["modelo"])
        semente = int(execucao["semente"])

        if not np.allclose(execucao["vint"], execucao["vint"].T):
            raise ValueError(f"vint não simétrica em {arquivo.name}")

        importancias.append(
            tabela_importancia(features, execucao["vimp"], METODO, modelo, semente)
        )
        interacoes.append(
            tabela_interacao(features, execucao["vint"], METODO, modelo, semente)
        )

        selecionadas = [f for f, bit in zip(features, execucao["cromossomo"]) if bit]
        selecoes.append({
            "model": modelo,
            "seed": semente,
            "n_features": len(selecionadas),
            "features": ";".join(selecionadas),
            "f1_validacao": float(execucao["f1_validacao"]),
            "fitness": float(execucao["fitness"]),
        })
        custos.append({
            "method": METODO,
            "model": modelo,
            "seed": semente,
            "tempo_s": float(execucao["tempo_s"]),
            "n_treinos": int(execucao["n_treinos"]),
        })

    return (
        pd.concat(importancias),
        pd.concat(interacoes),
        pd.DataFrame(selecoes),
        custos,
    )


def main() -> None:
    importancia, interacao, selecao, custos = consolidar(DIR_EXECUCOES)
    salvar_tabela(importancia, "importancia_gawll.csv")
    salvar_tabela(interacao, "interacao_gawll.csv")
    salvar_tabela(selecao, "selecao_gawll.csv")
    salvar_custo(custos)
    logger.info(f"{len(selecao)} execuções consolidadas")
    logger.info("Subconjuntos:\n" + selecao[
        ["model", "seed", "n_features", "f1_validacao"]].to_string(index=False))


if __name__ == "__main__":
    main()
