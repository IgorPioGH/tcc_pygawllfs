import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def separar_realizados_cancelados(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa o DataFrame em voos realizados e cancelados
       A separação usa a coluna 'Situação Voo': voos com valor "REALIZADO" vão para
       o primeiro DataFrame retornado, todos os demais vão para o segundo


    Args:
        df (pd.DataFrame): DataFrame bruto do VRA, contendo a coluna
        'Situação Voo'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (realizados, cancelados)
    """
    realizados = df[df["Situação Voo"] == "REALIZADO"]
    cancelados = df[df["Situação Voo"] != "REALIZADO"]
    logger.info(f"Quantidade voos cancelados: {cancelados.shape[0]}")
    return realizados, cancelados


def remover_sem_horario(df: pd.DataFrame) -> pd.DataFrame:
    """Remove as linhas em 'Partida Real' e 'Partida Prevista'

    Args:
        df (pd.DataFrame): DataFrame já filtrado por voos que aconteceram (Não cancelados)

    Returns:
        pd.DataFrame: DataFrame filtrado sem as linhas que faltam informações sobre partidas
    """
    total_antes = df.shape[0]
    df_filtrado = df[(df["Partida Prevista"].notna() & df["Partida Real"].notna())]
    total_depois = df_filtrado.shape[0]
    logger.info(
        f"Número de voos removidos por falta de horário: {total_antes - total_depois}"
    )
    return df_filtrado


def assert_sem_vazamento(df: pd.DataFrame) -> None:
    """Verifica se alguma coluna pós-hoc (vazamento de dados) ainda estão no DataFrame

        Compara a lista de colunas que não devem estar no DataFrame, com as próprias colunas
        do DataFrame, se encontrar alguma informa como erro.

    Args:
        df (pd.DataFrame): DataFrame para ser verificado antes de ser utilizado como conjunto
        de atributos

    Raises:
        ValueError: Se qualquer coluna proibida estiver presente no DataFrame
    """
    proibidas = [
        "Partida Real",
        "Chegada Real",
        "Situação Voo",
        "Código Justificativa",
        "atraso_min",
        "Chegada Prevista",
    ]
    encontradas = [coluna for coluna in proibidas if coluna in df.columns]

    if encontradas:
        raise ValueError(f"Colunas proibidas encontradas: {encontradas}")


if __name__ == "__main__":
    df = pd.read_parquet(path=Path("data/interim/vra_bruto.parquet"))
    (realizados, cancelados) = separar_realizados_cancelados(df)
    realizados_limpo = remover_sem_horario(realizados)
    realizados_limpo.to_parquet("data/interim/vra_limpo.parquet")
