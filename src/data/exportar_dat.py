import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

N_AMOSTRA = 50_000
SEMENTE = 42
DESTINO = Path("PyGAwLLfs/data/vra_piloto.dat")
if __name__ == "__main__":
    df = pd.read_parquet("data/interim/vra_target.parquet")

    dados = pd.DataFrame(index=df.index)
    dados["empresa"] = df["ICAO Empresa Aérea"].astype("category").cat.codes
    dados["origem"] = df["ICAO Aeródromo Origem"].astype("category").cat.codes
    dados["destino"] = df["ICAO Aeródromo Destino"].astype("category").cat.codes
    dados["hora"] = df["Partida Prevista"].dt.hour
    dados["dia_sem"] = df["Partida Prevista"].dt.dayofweek
    dados["mes"] = df["Partida Prevista"].dt.month
    dados["dia_mes"] = df["Partida Prevista"].dt.day
    dados["dia_ano"] = df["Partida Prevista"].dt.dayofyear
    dados["trimestre"] = df["Partida Prevista"].dt.quarter
    dados["ano"] = df["Partida Prevista"].dt.year
    dados["minuto"] = df["Partida Prevista"].dt.minute
    dados["fim_semana"] = (dados["dia_sem"] >= 5).astype(int)
    dados["periodo_dia"] = dados["hora"] // 6
    dados["hora_chegada"] = df["Chegada Prevista"].dt.hour
    dados["dia_sem_chegada"] = df["Chegada Prevista"].dt.dayofweek
    duracao_prevista = df["Chegada Prevista"] - df["Partida Prevista"]
    dados["duracao_prevista"] = duracao_prevista.dt.total_seconds() / 60
    dados["tipo_linha"] = df["Código Tipo Linha"].astype("category").cat.codes
    dados["di"] = df["Código Autorização (DI)"].astype("category").cat.codes
    dados["numero_voo"] = df["Número Voo"].astype("category").cat.codes
    dados["rota"] = (
        (
            df["ICAO Aeródromo Origem"].astype(str)
            + " "
            + df["ICAO Aeródromo Destino"].astype(str)
        )
        .astype("category")
        .cat.codes
    )
    dados["alvo"] = df["atrasado"]

    nulos = dados.isna().sum().sum()
    if nulos:
        raise ValueError(f"Há {nulos} valores nulos nas features do piloto")

    amostra, _ = train_test_split(
        dados, train_size=N_AMOSTRA, stratify=dados["alvo"], random_state=SEMENTE
    )

    n_exemplos, n_colunas = amostra.shape
    n_atributos = n_colunas - 1

    logger.info(f"Amostra: {n_exemplos} linhas, {n_atributos} atributos")
    logger.info(f"Taxa de atraso na amostra: {amostra['alvo'].mean():.4f}")

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    with open(DESTINO, "w") as f:
        f.write("TYPE: 1\n")
        f.write(f"N_ATTRIBUTES: {n_atributos}\n")
        f.write(f"N_EXAMPLES: {n_exemplos}\n")
        f.write("DATASET\n")
        amostra.to_csv(f, sep=" ", header=False, index=False)

    logger.info(f"Gravando: {DESTINO}")
