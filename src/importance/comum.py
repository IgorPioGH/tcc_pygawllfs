"""Funções compartilhadas pelos métodos de importância da Etapa 9.

Recebe valores de importância já calculados por cada método e os coloca no
formato padrão do projeto (docs/metodologia.md §7), feature,importance,rank,method,
model,seed, além de registrar tempo e número de treinos de cada cálculo em
results/tables/custo_importancia.csv. Quem consome: permutation.py,
shap_global.py, intrinsic.py, consolidar_gawll.py e, depois, a Etapa 11.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

# Raiz do projeto: comum.py -> importance/ -> src/ -> raiz
RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.models.baselines import DIR_MODELOS, DIR_TABELAS  # noqa: E402

MODELOS = ["logreg", "random_forest", "xgboost"]
SEMENTES = [42, 43, 44, 45, 46]
COLUNAS_PADRAO = ["feature", "importance", "rank", "method", "model", "seed"]
ARQUIVO_CUSTO = "custo_importancia.csv"


def carregar_modelo(nome: str):
    """Modelo treinado na Etapa 6 (treino completo), o mesmo avaliado no teste."""
    return joblib.load(DIR_MODELOS / f"{nome}.joblib")


def tabela_importancia(
    features: list[str],
    valores,
    method: str,
    model: str,
    seed: int,
    rank_por_absoluto: bool = False,
) -> pd.DataFrame:
    """Monta a tabela no formato padrão. Rank 1 = feature mais importante.

    rank_por_absoluto=True para valores com sinal cujo tamanho é o que importa
    (coeficientes da LogReg). O valor bruto, com sinal, fica em 'importance'
    (§4.8). Na Permutation o rank usa o valor bruto: negativo fica no fim.
    """
    tabela = pd.DataFrame({"feature": features, "importance": valores})
    criterio = tabela["importance"].abs() if rank_por_absoluto else tabela["importance"]
    tabela["rank"] = criterio.rank(ascending=False, method="min").astype(int)
    tabela["method"] = method
    tabela["model"] = model
    tabela["seed"] = seed
    return tabela.sort_values("rank")[COLUNAS_PADRAO]


def salvar_tabela(tabela: pd.DataFrame, nome_arquivo: str) -> Path:
    """Grava em results/tables/ e devolve o caminho."""
    DIR_TABELAS.mkdir(parents=True, exist_ok=True)
    caminho = DIR_TABELAS / nome_arquivo
    tabela.to_csv(caminho, index=False)
    return caminho


def salvar_custo(linhas: list[dict]) -> None:
    """Atualiza custo_importancia.csv com linhas {method, model, seed, tempo_s, n_treinos}.

    Cada script substitui só as linhas do próprio método, então a ordem em que
    os scripts rodam não importa e rodar de novo não duplica nada.
    """
    novas = pd.DataFrame(linhas)
    caminho = DIR_TABELAS / ARQUIVO_CUSTO
    if caminho.exists():
        antigas = pd.read_csv(caminho)
        antigas = antigas[~antigas["method"].isin(novas["method"].unique())]
        novas = pd.concat([antigas, novas], ignore_index=True)
    salvar_tabela(novas, ARQUIVO_CUSTO)
