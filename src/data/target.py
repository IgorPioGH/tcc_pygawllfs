import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Limites de plausibilidade para o atraso, em minutos.
# Assimétricos de propósito: companhia não antecipa partida de forma
# significativa (passageiro perderia o voo), mas atraso de horas é real.
ATRASO_MIN_PLAUSIVEL = -40
ATRASO_MAX_PLAUSIVEL = 720

# Recorte temporal do estudo
DATA_INICIO = "2024-01-01"
DATA_FIM = "2026-01-01"


if __name__ == "__main__":
    # Carregar parquet
    df = pd.read_parquet(Path("data/interim/vra_limpo.parquet"))

    # Recorte temporal: descarta registros fora de 2024-2025 (datas de
    # 2026 vindas de registros com data de partida mal preenchida)
    total_antes = len(df)
    df = df[
        (df["Partida Prevista"] >= DATA_INICIO) & (df["Partida Prevista"] < DATA_FIM)
    ]
    logger.info(f"Removidas fora do recorte 2024-2025: {total_antes - len(df)}")

    # atraso_min -> Subtrair Partida Prevista de Partida Real
    # Passar de timedelta para minutos
    duracao = df["Partida Real"] - df["Partida Prevista"]
    df["atraso_min"] = duracao.dt.total_seconds() / 60

    # Descarta atrasos fisicamente impossíveis (erro de registro de data
    # na Partida Real: aparecem como -4 dias de adiantamento ou 31 dias
    # de atraso)
    total_antes = len(df)
    df = df[df["atraso_min"].between(ATRASO_MIN_PLAUSIVEL, ATRASO_MAX_PLAUSIVEL)]
    logger.info(f"Removidas por atraso implausível: {total_antes - len(df)}")

    # Atraso -> comparar atraso_min >=15
    df["atrasado"] = df["atraso_min"] >= 15
    df["atrasado"] = df["atrasado"].astype(int)

    # Chacagem com logger
    # Soma dos atrasados / total
    logger.info(f"Média de atraso: {df['atrasado'].mean()}")
    logger.info(f"Mediana de atraso: {df['atraso_min'].median()}")

    logger.info(
        f"Minimo e Máximo de atraso: {df['atraso_min'].min()}, {df['atraso_min'].max()}"
    )

    logger.info(
        f"Taxa de atraso por mês: {df.groupby(df['Partida Prevista'].dt.month)['atrasado'].mean()}"
    )

    # Remover as 4 colunas proibidas com .drop(columns=[...])
    # mantendo atraso_min (ele sai em clean.py)
    df = df.drop(
        columns=["Partida Real", "Chegada Real", "Situação Voo", "Código Justificativa"]
    )
    df.to_parquet("data/interim/vra_target.parquet")
