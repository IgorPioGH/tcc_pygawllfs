import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def separar_realizados_cancelados(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa o DataFrame em voos realizados e cancelados pela coluna 'Situação Voo':
    "REALIZADO" vai para o primeiro DataFrame, todo o resto para o segundo.

    Usada em: bloco __main__ deste script (Etapa 3; main.py, etapa "dados").
    Importância: só voo realizado tem atraso. Os cancelados saem aqui, e por isso o
    congestionamento conta só realizados (limitação em docs/dicionario-features.md).

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
    """Remove as linhas sem 'Partida Prevista' ou sem 'Partida Real'.

    Usada em: bloco __main__ deste script (Etapa 3).
    Importância: sem os dois horários não dá para calcular o atraso, então essas
    linhas ficariam sem alvo.

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
    """Levanta erro se sobrar no DataFrame alguma coluna pós-hoc (Partida Real, Chegada
    Real, Situação Voo, Código Justificativa) ou o próprio alvo (atraso_min,
    atrasado).

    Usada em: baselines.carregar e gawll.carregar_subamostra, antes de todo treino.
    Importância: trava automática contra o vazamento do §4.3; se uma coluna proibida
    voltar, o pipeline para.

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
        "atrasado",
    ]
    encontradas = [coluna for coluna in proibidas if coluna in df.columns]

    if encontradas:
        raise ValueError(f"Colunas proibidas encontradas: {encontradas}")


if __name__ == "__main__":
    df = pd.read_parquet(path=Path("data/interim/vra_bruto.parquet"))
    (realizados, cancelados) = separar_realizados_cancelados(df)
    realizados_limpo = remover_sem_horario(realizados)
    realizados_limpo.to_parquet("data/interim/vra_limpo.parquet")
