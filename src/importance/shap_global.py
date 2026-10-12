"""Etapa 9 — importância global por SHAP (TreeSHAP).

Recebe o RF e o XGBoost treinados na Etapa 6 e o conjunto de VALIDAÇÃO
(nunca o teste, §4.9). Sorteia TAMANHO_AMOSTRA linhas da validação, calcula
o valor SHAP de cada feature em cada linha com shap.TreeExplainer (exato e
rápido para árvores, §4.6) e resume a importância global como a média do
valor absoluto: mean(|SHAP|).

Detalhes que vão para o Cap. 3:
    - feature_perturbation="tree_path_dependent": o TreeSHAP original, que
      usa a contagem de amostras de cada nó (vinda do treino) no lugar de um
      conjunto de background.
    - Unidades diferentes: o RF explica a probabilidade da classe 1; o XGBoost
      explica a margem (log-odds). Por isso a comparação é por rank (§4.8).
    - A LogReg não entra: seus coeficientes padronizados já são a importância
      intrínseca (§4.6, intrinsic.py).

A amostra é sorteada, então roda com as 5 sementes. Produz
results/tables/importancia_shap.csv no formato padrão. Quem consome:
Etapa 11 (comparação de rankings e estabilidade entre sementes).
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.importance.comum import (  # noqa: E402
    SEMENTES,
    carregar_modelo,
    salvar_custo,
    salvar_tabela,
    tabela_importancia,
)
from src.models.baselines import carregar  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

METODO = "shap"
MODELOS_ARVORE = ["random_forest", "xgboost"]
TAMANHO_AMOSTRA = 10_000


def calcular_shap(modelo, amostra: pd.DataFrame) -> list[float]:
    """mean(|SHAP|) de cada coluna da amostra, para a classe positiva, com o
    TreeExplainer.

    Usada em: main; aplicar_metodos.importancias_modelo_fixo.
    Importância: importância global a partir de explicações locais; TreeSHAP é exato
    em árvores, e KernelSHAP seria inviável nesta escala (§4.6).
    """
    explicador = shap.TreeExplainer(modelo, feature_perturbation="tree_path_dependent")
    valores = explicador(amostra).values

    # O RF do sklearn devolve uma matriz por classe: (linhas, features, 2).
    # Fica a da classe 1 (atrasou). O XGBoost já devolve (linhas, features).
    if valores.ndim == 3:
        valores = valores[:, :, 1]
    return list(np.abs(valores).mean(axis=0))


def main() -> None:
    """Roda o TreeSHAP no RF e no XGBoost, numa amostra de 10 mil linhas da validação
    por semente, e salva importancia_shap.csv e o custo.

    Usada em: python src/importance/shap_global.py; main.py, etapa "importancias";
    tests/test_importancia.py.
    Importância: um dos quatro métodos; a LogReg fica de fora porque os coeficientes
    padronizados já cumprem esse papel (§4.6).
    """
    x_val, _ = carregar("val")
    logger.info(f"Validação: {x_val.shape}, amostra de {TAMANHO_AMOSTRA} linhas por semente")

    tabelas, custos = [], []
    for nome in MODELOS_ARVORE:
        modelo = carregar_modelo(nome)
        for semente in SEMENTES:
            amostra = x_val.sample(n=TAMANHO_AMOSTRA, random_state=semente)

            inicio = time.perf_counter()
            valores = calcular_shap(modelo, amostra)
            tempo = time.perf_counter() - inicio

            tabelas.append(
                tabela_importancia(list(x_val.columns), valores, METODO, nome, semente)
            )
            custos.append(
                {"method": METODO, "model": nome, "seed": semente,
                 "tempo_s": tempo, "n_treinos": 0}
            )
            logger.info(f"{nome} semente {semente}: {tempo:.1f}s")
        logger.info(f"{nome} — top 5 (semente 42):\n"
                    + tabelas[-len(SEMENTES)].head(5).to_string(index=False))

    caminho = salvar_tabela(pd.concat(tabelas), "importancia_shap.csv")
    salvar_custo(custos)
    logger.info(f"Salvo em {caminho}")


if __name__ == "__main__":
    main()
