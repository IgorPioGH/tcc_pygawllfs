"""Etapa 9 — importância intrínseca de cada modelo.

Recebe os três modelos treinados na Etapa 6 (experiments/runs/baselines/) e
lê a importância que o próprio modelo já carrega depois do treino. Não usa
nenhum conjunto de dados, nem o teste.

    LogReg: coeficiente padronizado (com sinal; o rank usa o valor absoluto).
            As features passam pelo StandardScaler do pipeline, então os
            coeficientes estão na mesma escala e são comparáveis entre si.
    RF:     MDI (redução média de impureza), feature_importances_ do sklearn.
    XGBoost: ganho médio por divisão ("gain"), o padrão do feature_importances_.

É determinística: um valor por modelo, com seed = 42 (o random_state do
modelo). Produz results/tables/importancia_intrinseca.csv no formato padrão.
Quem consome: Etapa 11 (comparação de rankings).
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.importance.comum import (  # noqa: E402
    MODELOS,
    carregar_modelo,
    salvar_custo,
    salvar_tabela,
    tabela_importancia,
)
from src.models.baselines import SEMENTE  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

METODO = "intrinseca"


def importancia_intrinseca(nome: str, modelo) -> tuple[list[str], list[float]]:
    """Devolve (features, valores) da importância embutida no modelo: coeficientes
    padronizados da LogReg (com sinal), MDI do RF ou gain do XGBoost.

    Usada em: main; aplicar_metodos.importancias_modelo_fixo.
    Importância: o método mais barato (não treina nada); referência para os outros.
    """
    if nome == "logreg":
        regressao = modelo.named_steps["logisticregression"]
        return list(modelo.feature_names_in_), list(regressao.coef_[0])
    # RF (MDI) e XGBoost (gain) expõem o mesmo atributo
    return list(modelo.feature_names_in_), list(modelo.feature_importances_)


def main() -> None:
    """Calcula a importância intrínseca dos três modelos (determinística, uma semente
    só) e salva importancia_intrinseca.csv e o custo.

    Usada em: python src/importance/intrinsic.py; main.py, etapa "importancias";
    tests/test_importancia.py.
    Importância: um dos quatro métodos comparados na Etapa 11.
    """
    tabelas, custos = [], []
    for nome in MODELOS:
        inicio = time.perf_counter()
        features, valores = importancia_intrinseca(nome, carregar_modelo(nome))
        tempo = time.perf_counter() - inicio

        tabela = tabela_importancia(
            features, valores, METODO, nome, SEMENTE,
            rank_por_absoluto=(nome == "logreg"),
        )
        tabelas.append(tabela)
        custos.append(
            {"method": METODO, "model": nome, "seed": SEMENTE,
             "tempo_s": tempo, "n_treinos": 0}
        )
        logger.info(f"{nome} — top 5:\n" + tabela.head(5).to_string(index=False))

    caminho = salvar_tabela(pd.concat(tabelas), "importancia_intrinseca.csv")
    salvar_custo(custos)
    logger.info(f"Salvo em {caminho}")


if __name__ == "__main__":
    main()
