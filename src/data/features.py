# Transforma as colunas cruas de VRA nos atributos que os modelos vão realmente usar.
import logging
from pathlib import Path

import holidays
import pandas as pd

from sklearn.model_selection import KFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CORTE_TREINO_FIM = "2025-09-01"
CORTE_VAL_FIM = "2025-11-01"
SEMENTE = 42
K_FOLDS = 5
SUAVIZACAO = 20

COLUNAS_TE = {
    "ICAO Empresa Aérea": "empresa_te",
    "ICAO Aeródromo Origem": "origem_te",
    "ICAO Aeródromo Destino": "destino_te",
}

COLUNAS_PARA_REMOVER = [
    "Código Tipo Linha",
    "porte_origem",
    "porte_destino",
    "ICAO Empresa Aérea",
    "ICAO Aeródromo Origem",
    "ICAO Aeródromo Destino",
    "Número Voo",
    "Código Autorização (DI)",
]


def features_sem_alvo(df: pd.DataFrame) -> pd.DataFrame:
    df["hora"] = df["Partida Prevista"].dt.hour
    df["dia_sem"] = df["Partida Prevista"].dt.dayofweek
    df["mes"] = df["Partida Prevista"].dt.month
    df["dia_mes"] = df["Partida Prevista"].dt.day
    df["dia_ano"] = df["Partida Prevista"].dt.dayofyear
    df["trimestre"] = df["Partida Prevista"].dt.quarter
    df["ano"] = df["Partida Prevista"].dt.year
    df["minuto"] = df["Partida Prevista"].dt.minute
    df["fim_semana"] = (df["dia_sem"] >= 5).astype(int)
    df["periodo_dia"] = df["hora"] // 6
    # Duração programada
    duracao_prevista = df["Chegada Prevista"] - df["Partida Prevista"]
    df["duracao_prevista_min"] = duracao_prevista.dt.total_seconds() / 60

    # Feriados nacionais e césperas
    datas = df["Partida Prevista"].dt.normalize()
    feriados_br = holidays.Brazil(years=[2024, 2025, 2026])
    dias_feriado = pd.to_datetime(sorted(feriados_br.keys()))
    df["feriado"] = datas.isin(dias_feriado).astype(int)
    df["vespera_feriado"] = (
        (datas + pd.Timedelta(days=1)).isin(dias_feriado).astype(int)
    )

    # Congestionamento programado: voos na mesma fatia horario do mesmo aeroporto
    slot_partida = df["Partida Prevista"].dt.floor("h")
    slot_chegada = df["Chegada Prevista"].dt.floor("h")
    df["congestion_origem"] = df.groupby(["ICAO Aeródromo Origem", slot_partida])[
        "Partida Prevista"
    ].transform("size")
    df["congestion_destino"] = df.groupby(["ICAO Aeródromo Destino", slot_chegada])[
        "Chegada Prevista"
    ].transform("size")

    # Tipo de linha: One-hot enconde é seguro, pois tem poucas linhas
    tipo_linha = pd.get_dummies(df["Código Tipo Linha"], prefix="tipo_linha", dtype=int)
    df = pd.concat([df, tipo_linha], axis=1)
    return df


def dividir_temporal(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = df["Partida Prevista"]

    treino = df[data < CORTE_TREINO_FIM]
    validacao = df[(data >= CORTE_TREINO_FIM) & (data < CORTE_VAL_FIM)]
    teste = df[data >= CORTE_VAL_FIM]

    # Cópias explícitas, para evitar SettingWithCopyWarning
    treino = treino.copy()
    validacao = validacao.copy()
    teste = teste.copy()

    # Garantindo que nenhum conjunto fique vazio para .min() e .max() funcionarem
    for nome, conjunto in [
        ("treino", treino),
        ("validacao", validacao),
        ("teste", teste),
    ]:
        if conjunto.empty:
            raise ValueError(f"Conjunto '{nome}' ficou vazio, confira os cortes")

    total = len(treino) + len(validacao) + len(teste)
    if total != len(df):
        raise ValueError("Partição perdeu ou duplicou linhas")
    if treino["Partida Prevista"].max() > validacao["Partida Prevista"].min():
        raise ValueError("Invariante Violado: Treino invade validação")
    if validacao["Partida Prevista"].max() > teste["Partida Prevista"].min():
        raise ValueError("Invariante Violado: Validação invade teste")

    for nome, conjunto in [
        ("treino", treino),
        ("validacao", validacao),
        ("teste", teste),
    ]:
        logger.info(f"nome: {nome}")
        logger.info(f"Número de linhas: {len(conjunto)}")
        logger.info(f"Taxa de atraso: {conjunto['atrasado'].mean()}")
        logger.info(f"Data Mínima: {conjunto['Partida Prevista'].min()}")
        logger.info(f"Data Máxima: {conjunto['Partida Prevista'].max()}")
    return treino, validacao, teste


def ajustar_porte_aeroporto(treino: pd.DataFrame) -> dict:
    vol_origem = treino["ICAO Aeródromo Origem"].value_counts()
    vol_destino = treino["ICAO Aeródromo Destino"].value_counts()
    volume = vol_origem.add(vol_destino, fill_value=0)

    faixas = pd.qcut(
        volume,
        q=4,
        labels=["pequeno", "medio", "grande", "hub"],
    )
    mapa = faixas.to_dict()
    return mapa


def aplicar_porte(df: pd.DataFrame, mapa: dict) -> pd.DataFrame:
    df["porte_origem"] = df["ICAO Aeródromo Origem"].map(mapa)
    df["porte_destino"] = df["ICAO Aeródromo Destino"].map(mapa)

    df["porte_destino"] = df["porte_destino"].fillna("pequeno")
    df["porte_origem"] = df["porte_origem"].fillna("pequeno")

    porte_dummies_origem = pd.get_dummies(
        df["porte_origem"], prefix="porte_origem", dtype=int
    )
    porte_dummies_destino = pd.get_dummies(
        df["porte_destino"], prefix="porte_destino", dtype=int
    )
    df = pd.concat([df, porte_dummies_origem, porte_dummies_destino], axis=1)

    return df


def calcular_mapa_suavizado(df, coluna, alvo, m):
    media_global = df[alvo].mean()
    agregado = df.groupby(coluna, observed=True)[alvo].agg(["sum", "count"])
    valores = (agregado["sum"] + m * media_global) / (agregado["count"] + m)
    return valores.to_dict(), media_global


def target_encoding_oof(treino, coluna, alvo, k=K_FOLDS, m=SUAVIZACAO):
    codificado = pd.Series(index=treino.index, dtype=float)
    kf = KFold(n_splits=k, shuffle=True, random_state=SEMENTE)

    for idx_fora, idx_dentro in kf.split(treino):
        mapa_parcial, media_parcial = calcular_mapa_suavizado(
            treino.iloc[idx_fora], coluna, alvo, m
        )
        valores = (
            treino.iloc[idx_dentro][coluna].map(mapa_parcial).fillna(media_parcial)
        )
        codificado.iloc[idx_dentro] = valores.to_numpy()

    mapa_final, media_global = calcular_mapa_suavizado(treino, coluna, alvo, m)
    return codificado, mapa_final, media_global


def remover_colunas_cruas(df):
    return df.drop(columns=COLUNAS_PARA_REMOVER, errors="ignore")


def conferir_tudo_numerico(df, permitidas):
    nao_numericas = df.select_dtypes(exclude="number").columns
    inesperadas = [c for c in nao_numericas if c not in permitidas]
    if inesperadas:
        raise ValueError(f"COlunas não numéricas inesperadas: {inesperadas}")


if __name__ == "__main__":
    df = pd.read_parquet(path=Path("data/interim/vra_target.parquet"))

    df = features_sem_alvo(df=df)
    treino, validacao, teste = dividir_temporal(df=df)

    mapa_porte = ajustar_porte_aeroporto(treino=treino)
    treino = aplicar_porte(df=treino, mapa=mapa_porte)
    validacao = aplicar_porte(df=validacao, mapa=mapa_porte)
    teste = aplicar_porte(df=teste, mapa=mapa_porte)

    for coluna, nome_novo in COLUNAS_TE.items():
        codificado, mapa, media = target_encoding_oof(treino, coluna, "atrasado")
        treino[nome_novo] = codificado
        validacao[nome_novo] = validacao[coluna].map(mapa).fillna(media)
        teste[nome_novo] = teste[coluna].map(mapa).fillna(media)
        logger.info(f"{nome_novo}: media global do treino = {media:.4f}")

    permitidas = ["Partida Prevista", "Chegada Prevista"]
    conjuntos = {"train": treino, "val": validacao, "test": teste}

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    for nome, conjunto in conjuntos.items():
        conjunto = remover_colunas_cruas(conjunto)
        conferir_tudo_numerico(conjunto, permitidas)
        n_features = (
            len(conjunto.columns) - len(permitidas) - 2
        )  # menos alvo e atraso_min
        logger.info(f"{nome}: {len(conjunto)} linhas, {n_features} atributos")
        conjunto.to_parquet(f"data/processed/{nome}.parquet")
