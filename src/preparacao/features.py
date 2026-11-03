# Transforma as colunas cruas de VRA nos atributos que os modelos vão realmente usar.
import logging
from pathlib import Path

import holidays
import pandas as pd

from sklearn.model_selection import KFold, train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Janelas de cada recorte: (início, fim do treino, fim da validação, fim do teste),
# intervalos [a, b). Os recortes de 2024 e 2025 usam o mesmo calendário do principal
# (treino jan–ago, validação set–out, teste nov–dez), um ano de cada vez.
RECORTES = {
    "principal": ("2024-01-01", "2025-09-01", "2025-11-01", "2026-01-01"),
    "2024": ("2024-01-01", "2024-09-01", "2024-11-01", "2025-01-01"),
    "2025": ("2025-01-01", "2025-09-01", "2025-11-01", "2026-01-01"),
}
SEMENTE = 42
K_FOLDS = 5
SUAVIZACAO = 20
# Entre o ruído entre folds (0,001)
TOLERANCIA = 0.01

# Colunas não numéricas que ficam nos conjuntos: os horários previstos, para a
# divisão temporal e as análises por dia e hora
PERMITIDAS = ["Partida Prevista", "Chegada Prevista"]
TAMANHO_SUBAMOSTRA = 50_000

# Faixas de porte de aeroporto, do menor para o maior volume
PORTES = ["pequeno", "medio", "grande", "hub"]

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
    """Cria os atributos que não usam o alvo: derivados da Partida Prevista (hora, dia
    da semana, mês, dia do mês, dia do ano, trimestre, minuto, fim de semana, período
    do dia), duração prevista, feriado e véspera, congestionamento na origem e no
    destino e one-hot do tipo de linha.

    Usada em: bloco __main__ deste script (main.py, etapa "preparacao") e
    recortes.py, uma vez sobre a base inteira.
    Importância: só usa informação conhecida antes da partida e não usa o
    alvo, por isso pode rodar antes do split sem vazamento. O congestionamento conta
    os voos da malha inteira, então tem de ser calculado antes de qualquer recorte.
    """
    df["hora"] = df["Partida Prevista"].dt.hour
    df["dia_sem"] = df["Partida Prevista"].dt.dayofweek
    df["mes"] = df["Partida Prevista"].dt.month
    df["dia_mes"] = df["Partida Prevista"].dt.day
    df["dia_ano"] = df["Partida Prevista"].dt.dayofyear
    df["trimestre"] = df["Partida Prevista"].dt.quarter
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
    df: pd.DataFrame, inicio: str, corte_treino: str, corte_val: str, fim: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recorta a janela [inicio, fim) da Partida Prevista e a divide por data absoluta:
    treino antes de corte_treino, validação até corte_val e teste até fim. Confere que
    nenhum conjunto ficou vazio, que nenhuma linha da janela se perdeu e que
    max(treino) ≤ min(validação) e max(validação) ≤ min(teste).

    Usada em: montar_conjuntos.
    Importância: implementa o split temporal (prever o futuro a partir do
    passado, nada de shuffle), para qualquer recorte de RECORTES.
    """
    janela = df[(df["Partida Prevista"] >= inicio) & (df["Partida Prevista"] < fim)]
    data = janela["Partida Prevista"]

    treino = janela[data < corte_treino]
    validacao = janela[(data >= corte_treino) & (data < corte_val)]
    teste = janela[data >= corte_val]

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
    if total != len(janela):
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
    """Classifica cada aeroporto em pequeno, médio, grande ou hub pelos quartis do
    volume de pousos e decolagens no treino.

    Usada em: montar_conjuntos.
    Importância: resume a alta cardinalidade dos aeroportos em 4 faixas (N ≤ 40);
    ajustado só no treino, para não vazar volume de validação e teste.
    """
    vol_origem = treino["ICAO Aeródromo Origem"].value_counts()
    vol_destino = treino["ICAO Aeródromo Destino"].value_counts()
    volume = vol_origem.add(vol_destino, fill_value=0)

    faixas = pd.qcut(
        volume,
        q=4,
        labels=PORTES,
    )
    mapa = faixas.to_dict()
    return mapa


def aplicar_porte(df: pd.DataFrame, mapa: dict) -> pd.DataFrame:
    """Aplica o mapa de porte à origem e ao destino (aeroporto fora do mapa vira
    "pequeno") e cria as dummies com categorias fixas.

    Usada em: montar_conjuntos, nos três conjuntos.
    Importância: as categorias fixas garantem as mesmas 4 colunas em treino,
    validação e teste, mesmo que uma faixa falte num deles.
    """
    df["porte_origem"] = df["ICAO Aeródromo Origem"].map(mapa)
    df["porte_destino"] = df["ICAO Aeródromo Destino"].map(mapa)

    df["porte_destino"] = df["porte_destino"].fillna("pequeno")
    df["porte_origem"] = df["porte_origem"].fillna("pequeno")

    # Categorias fixas: o get_dummies cria as 4 colunas mesmo que uma faixa não
    # apareça neste conjunto, então treino, validação e teste têm sempre o mesmo
    # esquema. Em ordem alfabética, a mesma que o get_dummies usava sobre texto.
    tipo_porte = pd.CategoricalDtype(categories=sorted(PORTES))
    df["porte_origem"] = df["porte_origem"].astype(tipo_porte)
    df["porte_destino"] = df["porte_destino"].astype(tipo_porte)

    porte_dummies_origem = pd.get_dummies(
        df["porte_origem"], prefix="porte_origem", dtype=int
    )
    porte_dummies_destino = pd.get_dummies(
        df["porte_destino"], prefix="porte_destino", dtype=int
    )
    df = pd.concat([df, porte_dummies_origem, porte_dummies_destino], axis=1)

    return df


def calcular_mapa_suavizado(df, coluna, alvo, m):
    """Média do alvo por categoria, suavizada: (soma + m · média global) / (contagem +
    m). Devolve o mapa e a média global.

    Usada em: target_encoding_oof.
    Importância: a suavização puxa categorias raras para a média global e reduz o
    ruído do encoding.
    """
    media_global = df[alvo].mean()
    agregado = df.groupby(coluna, observed=True)[alvo].agg(["sum", "count"])
    valores = (agregado["sum"] + m * media_global) / (agregado["count"] + m)
    return valores.to_dict(), media_global


def target_encoding_oof(treino, coluna, alvo, k=K_FOLDS, m=SUAVIZACAO):
    """Target encoding out-of-fold (K = 5): cada fold do treino recebe o mapa ajustado
    nos outros K − 1. No fim, ajusta o mapa final no treino inteiro, para aplicar sem
    refit em validação e teste.

    Usada em: montar_conjuntos, para empresa, origem e destino.
    Importância: é o target encoding sem vazamento; nenhuma linha de treino usa o próprio rótulo
    no seu encoding.
    """
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


def verifica_vazamento(df, coluna, codificado, media_global):
    """Teste de vazamento do target encoding: nas categorias com uma única linha, o encoding OOF não pode
    depender do rótulo dela, então a média do encoding com atrasado = 0 e com
    atrasado = 1 tem de ser igual (dentro da TOLERANCIA). Levanta erro se não for.

    Usada em: montar_conjuntos, uma vez por coluna codificada.
    Importância: prova, a cada execução, que o encoding não vazou o alvo dentro do
    treino.
    """
    singletons = df[coluna].value_counts()[lambda x: x == 1].index
    mascara = df[coluna].isin(singletons)
    treino_filtrado = df.loc[mascara]
    codificado_filtrado = codificado.loc[mascara]
    medias_0_e_1 = codificado_filtrado.groupby(treino_filtrado["atrasado"]).mean()
    try:
        media_0 = medias_0_e_1[0]
    except KeyError:
        media_0 = media_global
        logger.warning("Media 0 foi substituída pela média global")
    try:
        media_1 = medias_0_e_1[1]
    except KeyError:
        media_1 = media_global
        logger.warning("Media 1 foi substituída pela média global")
    # Se a diferença das médias for menor do que a tolerancia esta ok
    if abs(media_0 - media_1) > TOLERANCIA:
        raise ValueError(
            f"A diferença entre as médias é muito grande: {abs(media_0 - media_1)}"
        )
    logger.info(f"Média 0: {media_0}")
    logger.info(f"Média 1: {media_1}")
    logger.info(f"Média Global: {media_global}")


def remover_colunas_cruas(df):
    """Remove as colunas já transformadas (códigos ICAO, porte, tipo de linha) e os
    identificadores (número do voo, DI).

    Usada em: montar_conjuntos, em cada conjunto.
    Importância: deixa só atributos que os modelos usam e impede que um identificador
    entre como feature.
    """
    return df.drop(columns=COLUNAS_PARA_REMOVER, errors="ignore")


def conferir_tudo_numerico(df, permitidas):
    """Levanta erro se sobrar coluna não numérica além das permitidas (Partida Prevista
    e Chegada Prevista, mantidas para o split e para a validação de domínio).

    Usada em: montar_conjuntos, em cada conjunto.
    Importância: garante que o dataset congelado entra direto nos modelos e que
    nenhuma coluna crua passou.
    """
    nao_numericas = df.select_dtypes(exclude="number").columns
    inesperadas = [c for c in nao_numericas if c not in permitidas]
    if inesperadas:
        raise ValueError(f"COlunas não numéricas inesperadas: {inesperadas}")


def balancear(conjunto: pd.DataFrame, semente: int = SEMENTE) -> pd.DataFrame:
    """Mantém todos os voos atrasados e sorteia, sem reposição, o mesmo número de voos
    pontuais; devolve o conjunto em ordem de Partida Prevista. O sorteio é global no
    conjunto, nunca por grupo (mês, aeroporto, empresa): balancear dentro de um grupo
    deixaria a taxa de atraso de todos os grupos em 50% e apagaria a informação da
    variável.

    Usada em: montar_conjuntos, em treino, validação e teste.
    Importância: descartar pontuais ao acaso multiplica as chances de atraso de todos
    os grupos pelo mesmo fator, então a ordem entre os grupos se mantém; com 50/50, a
    acurácia passa a ser uma métrica legível (o trivial acerta 0,5).
    """
    atrasados = conjunto[conjunto["atrasado"] == 1]
    pontuais = conjunto[conjunto["atrasado"] == 0]
    if len(atrasados) > len(pontuais):
        raise ValueError(f"Mais atrasados ({len(atrasados)}) que pontuais ({len(pontuais)})")
    sorteados = pontuais.sample(n=len(atrasados), random_state=semente)
    return pd.concat([atrasados, sorteados]).sort_values("Partida Prevista", kind="stable")


def montar_conjuntos(df: pd.DataFrame, recorte: str = "principal") -> dict[str, pd.DataFrame]:
    """Treino, validação e teste de um recorte, prontos para os modelos: divide no
    tempo, ajusta o porte dos aeroportos no treino e aplica nos três, faz o target
    encoding fora da dobra (com a checagem de vazamento em cada coluna), balanceia
    cada conjunto em 50/50, remove as colunas cruas e confere que só sobraram colunas
    numéricas. df já tem de vir de features_sem_alvo. O porte e o encoding são
    ajustados no treino completo, antes do balanceamento: as taxas que o encoding
    carrega são as reais.

    Usada em: bloco __main__ deste script; recortes.py.
    Importância: a mesma preparação serve ao experimento principal e aos recortes
    temporais; o que muda é só a janela de datas.
    """
    treino, validacao, teste = dividir_temporal(df, *RECORTES[recorte])

    mapa_porte = ajustar_porte_aeroporto(treino=treino)
    treino = aplicar_porte(df=treino, mapa=mapa_porte)
    validacao = aplicar_porte(df=validacao, mapa=mapa_porte)
    teste = aplicar_porte(df=teste, mapa=mapa_porte)

    for coluna, nome_novo in COLUNAS_TE.items():
        codificado, mapa, media = target_encoding_oof(treino, coluna, "atrasado")
        # Roda uma vez por coluna e verifica que nenhuma linha de treino usou o próprio rótulo
        verifica_vazamento(treino, coluna, codificado, media)
        treino[nome_novo] = codificado
        validacao[nome_novo] = validacao[coluna].map(mapa).fillna(media)
        teste[nome_novo] = teste[coluna].map(mapa).fillna(media)
        logger.info(f"{nome_novo}: media global do treino = {media:.4f}")

    conjuntos = {}
    for nome, conjunto in {"train": treino, "val": validacao, "test": teste}.items():
        conjunto = remover_colunas_cruas(balancear(conjunto))
        conferir_tudo_numerico(conjunto, PERMITIDAS)
        conjuntos[nome] = conjunto
    return conjuntos


def criar_subamostra(treino: pd.DataFrame, validacao: pd.DataFrame,
                     n: int = TAMANHO_SUBAMOSTRA, semente: int = SEMENTE) -> pd.DataFrame:
    """Sorteia n linhas de treino + validação, estratificadas pela origem (treino ou
    validação) e pelo alvo; a coluna "origem" diz de onde veio cada linha.

    Usada em: bloco __main__ deste script; recortes.py.
    Importância: é a base do PyGAwLLfs, que retreina o modelo milhares de vezes e não
    cabe no treino completo; nunca inclui o teste.
    """
    treino = treino.assign(origem="treino")
    validacao = validacao.assign(origem="validacao")
    treino_validacao = pd.concat([treino, validacao])
    chave = treino_validacao["origem"] + " " + treino_validacao["atrasado"].astype(str)
    amostra, _ = train_test_split(
        treino_validacao, stratify=chave, train_size=n, random_state=semente
    )
    logger.info(f"Linhas por origem: {amostra['origem'].value_counts()}")
    logger.info(f"Taxa de atraso por origem: {amostra.groupby('origem')['atrasado'].mean()}")
    return amostra


if __name__ == "__main__":
    df = features_sem_alvo(pd.read_parquet(Path("data/interim/vra_target.parquet")))
    conjuntos = montar_conjuntos(df, "principal")

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    for nome, conjunto in conjuntos.items():
        n_features = len(conjunto.columns) - len(PERMITIDAS) - 2  # menos alvo e atraso_min
        logger.info(f"{nome}: {len(conjunto)} linhas, {n_features} atributos")
        conjunto.to_parquet(f"data/processed/{nome}.parquet")
    subamostra = criar_subamostra(conjuntos["train"], conjuntos["val"])
    subamostra.to_parquet("data/processed/subamostra.parquet")
