"""Checagem da subamostra do PyGAwLLfs: treinar em ~45 mil voos muda o que é importante?

O PyGAwLLfs explica modelos treinados na parte de treino da subamostra
(data/processed/subamostra.parquet), porque retreina o modelo milhares de vezes;
intrínseca, Permutation e SHAP explicam o modelo treinado no treino completo.
Para separar o efeito do método do efeito do tamanho dos dados, cada modelo é
treinado na subamostra (mesmos hiperparâmetros) e a sua importância intrínseca e
a Permutation (validação completa, semente 42) são comparadas com as do modelo
completo (importancia_intrinseca.csv e importancia_permutation.csv), com as
mesmas métricas da comparação de rankings.

Leitura: Spearman alto (≥ 0,8) quer dizer que a subamostra preserva a estrutura
de importância, e as diferenças entre o PyGAwLLfs e os outros métodos vêm do
método; baixo quer dizer que parte da divergência vem do tamanho dos dados.

Produz results/tables/checagem_subamostra.csv (modelo, método, spearman, top5,
top10). Nunca lê o teste.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.avaliacao.rankings import comparar, pontuacao  # noqa: E402
from src.modelagem import comum  # noqa: E402
from src.modelagem.baselines import (  # noqa: E402
    NAO_ATRIBUTOS,
    SEMENTE,
    carregar,
    criar_modelo,
    ler_configuracoes_escolhidas,
)
from src.modelagem.intrinsic import importancia_intrinseca  # noqa: E402
from src.modelagem.permutation import calcular_permutation  # noqa: E402
from src.preparacao.clean import assert_sem_vazamento  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARQUIVO_SUBAMOSTRA = RAIZ / "data" / "processed" / "subamostra.parquet"


def treino_da_subamostra() -> tuple[pd.DataFrame, pd.Series]:
    """(x, y) da parte de treino da subamostra, com as mesmas colunas de carregar().

    Usada em: main.
    Importância: é exatamente o treino que o PyGAwLLfs usa.
    """
    df = pd.read_parquet(ARQUIVO_SUBAMOSTRA)
    df = df[df["origem"] == "treino"]
    x = df.drop(columns=NAO_ATRIBUTOS + ["origem"])
    assert_sem_vazamento(x)
    return x, df["atrasado"]


def comparar_com_completo(tabela_sub: pd.DataFrame, tabela_completa: pd.DataFrame) -> dict:
    """Métricas de rankings.comparar entre a importância do modelo da subamostra e a do
    modelo completo, cada uma ordenada por rankings.pontuacao.

    Usada em: main.
    Importância: usa a mesma régua da concordância entre métodos.
    """
    sub = pontuacao(tabela_sub).set_axis(tabela_sub["feature"])
    completa = pontuacao(tabela_completa).set_axis(tabela_completa["feature"])
    return comparar(completa, sub)


def main() -> None:
    """Treina os três modelos na subamostra, calcula a intrínseca e a Permutation e
    compara com as do modelo completo; salva checagem_subamostra.csv.

    Usada em: python src/avaliacao/checagem_subamostra.py; main.py, etapa "rankings".
    Importância: justifica comparar o PyGAwLLfs (subamostra) com os outros métodos
    (treino completo).
    """
    x_sub, y_sub = treino_da_subamostra()
    x_val, y_val = carregar("val")
    configuracoes = ler_configuracoes_escolhidas()
    completas = {
        "intrinseca": pd.read_csv(comum.DIR_TABELAS / "importancia_intrinseca.csv"),
        "permutation": pd.read_csv(comum.DIR_TABELAS / "importancia_permutation.csv")
        .query("seed == @SEMENTE"),
    }
    logger.info(f"Subamostra: treino {x_sub.shape} | validação {x_val.shape}")

    linhas = []
    for nome in comum.MODELOS:
        modelo = criar_modelo(nome, configuracoes[nome])
        modelo.fit(x_sub, y_sub)
        features, valores = importancia_intrinseca(nome, modelo)
        da_subamostra = {
            "intrinseca": comum.tabela_importancia(features, valores, "intrinseca", nome,
                                                   SEMENTE, rank_por_absoluto=(nome == "logreg")),
            "permutation": comum.tabela_importancia(
                list(x_val.columns), calcular_permutation(modelo, x_val, y_val, SEMENTE),
                "permutation", nome, SEMENTE),
        }
        for metodo, tabela in da_subamostra.items():
            completa = completas[metodo][completas[metodo]["model"] == nome]
            resultado = comparar_com_completo(tabela, completa)
            linhas.append({"model": nome, "method": metodo, "spearman": resultado["spearman"],
                           "top5": resultado["top5"], "top10": resultado["top10"]})
        logger.info(f"{nome}: pronto")

    tabela = pd.DataFrame(linhas)
    if len(tabela) != 2 * len(comum.MODELOS):
        raise ValueError(f"checagem_subamostra com {len(tabela)} linhas, esperado 6")
    comum.salvar_tabela(tabela.round(4), "checagem_subamostra.csv")
    logger.info("Subamostra × treino completo:\n" + tabela.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
