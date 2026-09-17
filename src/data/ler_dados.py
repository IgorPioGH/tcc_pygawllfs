import logging
from builtins import sorted
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def carregar_mes(caminho: Path) -> pd.DataFrame:
    return pd.read_csv(
        caminho,
        sep=";",
        skiprows=1,
        encoding="utf-8-sig",
        parse_dates=[
            "Partida Prevista",
            "Partida Real",
            "Chegada Prevista",
            "Chegada Real",
        ],
    )


def carregar_todos_os_meses(pasta: Path) -> pd.DataFrame:
    """Função que recebe o caminho de um diretório que possui os arquivos
    csv que serão lidos

    Args:
        pasta (Path): Diretório que possui os arquivos csv

    Returns:
        pd.DataFrame: DataFrame pandas com todos os csv concatenados
    """
    csv_meses = sorted(pasta.glob("VRA_*.csv"))
    data_frames = []
    for csv in csv_meses:
        data_frames.append(carregar_mes(csv))
    df_completo = pd.concat(data_frames, ignore_index=True)
    colunas_datas = [
        "Partida Prevista",
        "Partida Real",
        "Chegada Prevista",
        "Chegada Real",
    ]
    for coluna in colunas_datas:
        df_completo[coluna] = pd.to_datetime(df_completo[coluna], errors="coerce")

    df_completo["Código Autorização (DI)"] = df_completo[
        "Código Autorização (DI)"
    ].astype("string")
    return df_completo


def verificar_meses_completos(pasta: Path) -> None:
    """Verifica todos os meses esperados (2024-2025) estão presentes na pasta,
    sem meses faltando ou duplicados

    Compara os pares (ano,mes) esperados com os efetivamente encontrados a
    partir dos nomes dos arquivos no padrão VRA_{ano}_{mes:02d}.csv, e registra
    um relatório indicando meses ausentes, arquivos inesperados e duplicos

    Args:
        pasta (Path): Diretório contendo os arquivos CSV do VRA.

    Returns:
        None: a função apenas registra o resultado (log), não retorna um valor.
    """
    arquivos = sorted(pasta.glob("VRA_*.csv"))
    # Meses que deveriam existir
    esperados = set()
    for ano in [2024, 2025]:
        for mes in range(1, 13):
            esperados.add((ano, mes))
    # Quais arquivos existem de verdade
    encontrados = []
    for arquivo in arquivos:
        nome = arquivo.stem
        partes = nome.split("_")
        ano_do_arquivo = int(partes[1])
        mes_do_arquivo = int(partes[2])
        encontrados.append((ano_do_arquivo, mes_do_arquivo))
    # Verificar duplicatas
    contagens = Counter(encontrados)
    duplicados = [x[0] for x in contagens.items() if x[1] > 1]

    # Comparacao entre o que era esperado, com o que foi encontrado
    encontrados_unicos = set(encontrados)
    faltando = list(esperados - encontrados_unicos)
    inesperados = list(encontrados_unicos - esperados)

    # Relato dos resultados
    if not faltando and not inesperados and not duplicados:
        logger.info("Todos os dados foram baixados corretamente!")
    else:
        logger.error(f"""Erro no download dos dados:
              faltando: {faltando}
              inesperados: {inesperados}
              duplicados: {duplicados}""")


if __name__ == "__main__":
    verificar_meses_completos(Path("data/raw"))
    df = carregar_todos_os_meses(Path("data/raw"))
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    print(df.head())
    logger.info(f"Total de linhas: {len(df)}")
    logger.info(
        f"Periodo {df['Partida Prevista'].min()} até {df['Partida Prevista'].max()}"
    )
    logger.info(
        f"Baixado em: {datetime.fromtimestamp(next(Path('data/raw').glob('VRA_*.csv')).stat().st_mtime, tz=ZoneInfo('America/Sao_Paulo'))}"
    )
    df.to_parquet("data/interim/vra_bruto.parquet")
