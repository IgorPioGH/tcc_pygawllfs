"""Etapa 9 — Permutation Feature Importance.

Recebe os três modelos treinados na Etapa 6 e o conjunto de VALIDAÇÃO
(nunca o teste, §4.9). Para cada feature, embaralha a coluna e mede quanto
o F1 cai; repete n_repeats vezes e tira a média. A métrica é o F1 por
coerência com o fitness do PyGAwLLfs (§4.2): os dois métodos medem
"quanto o F1 piora sem essa informação".

O valor pode ser negativo (embaralhar pode melhorar o F1 por acaso). Ele é
salvo bruto (§4.8); o rank usa o valor bruto, então negativo vai para o fim.

É estocástica (o embaralhamento), então roda com as 5 sementes. Produz
results/tables/importancia_permutation.csv no formato padrão. Quem consome:
Etapa 11 (comparação de rankings e estabilidade entre sementes).
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.importance.comum import (  # noqa: E402
    MODELOS,
    SEMENTES,
    carregar_modelo,
    salvar_custo,
    salvar_tabela,
    tabela_importancia,
)
from src.models.baselines import carregar  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

METODO = "permutation"
N_REPEATS = 10
METRICA = "f1"


def calcular_permutation(modelo, x: pd.DataFrame, y: pd.Series, semente: int,
                         n_repeats: int = N_REPEATS) -> list[float]:
    """Queda média de F1 ao embaralhar cada coluna, n_repeats vezes, na ordem de
    x.columns.

    Usada em: main; aplicar_metodos.importancias_modelo_fixo.
    Importância: método agnóstico ao modelo; mede o quanto o modelo treinado depende
    de cada feature.
    """
    resultado = permutation_importance(
        modelo, x, y,
        scoring=METRICA,
        n_repeats=n_repeats,
        random_state=semente,
        n_jobs=1,  # o RF já usa todos os núcleos dentro do predict
    )
    return list(resultado.importances_mean)


def main() -> None:
    """Roda a Permutation nos três modelos e nas 5 sementes, sempre na validação (§4.9),
    e salva importancia_permutation.csv e o custo.

    Usada em: python src/importance/permutation.py; main.py, etapa "importancias";
    tests/test_importancia.py.
    Importância: um dos quatro métodos; as sementes permitem medir a estabilidade.
    """
    x_val, y_val = carregar("val")
    logger.info(f"Validação: {x_val.shape}, {N_REPEATS} repetições, métrica {METRICA}")

    tabelas, custos = [], []
    for nome in MODELOS:
        modelo = carregar_modelo(nome)
        for semente in SEMENTES:
            inicio = time.perf_counter()
            valores = calcular_permutation(modelo, x_val, y_val, semente)
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

    caminho = salvar_tabela(pd.concat(tabelas), "importancia_permutation.csv")
    salvar_custo(custos)
    logger.info(f"Salvo em {caminho}")


if __name__ == "__main__":
    main()
