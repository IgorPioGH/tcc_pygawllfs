"""Etapa 10 — Diferencial 1: gerador do dataset sintético com gabarito.

Recebe só os parâmetros abaixo (nenhum dado real). Gera um problema de
classificação cuja estrutura é conhecida por construção:

    z = BETA_INTERACAO * x1 * x2  +  x3  +  x4  +  SIGMA_RUIDO * ruído
    y = 1 se z > 0, senão 0

    x1, x2   interação pura: sozinhas não dizem nada sobre y (o produto é
             simétrico em torno de zero), juntas são o sinal mais forte
    x3, x4   informativas com efeito puramente aditivo
    x5, x6   redundantes: cópias ruidosas de x3 e x4 (correlação RHO), que
             não entram em z — são proxies, não ruído
    x7..x10  ruído puro

Os parâmetros foram calibrados para que, com os modelos da Etapa 6, a
interação valha +0,09 de F1 no RF e +0,19 no XGBoost (a LogReg, linear,
não consegue usá-la), e para que tirar x3 custe pouco quando x5 está
presente (é a redundância).

Produz data/processed/sintetico.parquet (x1..x10, y, conjunto), com a
divisão treino/validação estratificada por y. Quem consome:
aplicar_metodos.py (roda os métodos) e recovery.py (usa o GABARITO).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parents[2]
ARQUIVO_SINTETICO = RAIZ / "data" / "processed" / "sintetico.parquet"

N_AMOSTRAS = 5000
SEMENTE_DADOS = 42
BETA_INTERACAO = 2.0
SIGMA_RUIDO = 0.5
RHO_REDUNDANCIA = 0.9
FRACAO_VALIDACAO = 0.3

FEATURES = [f"x{i}" for i in range(1, 11)]

# Gabarito em três categorias (docs/metodologia.md §5): redundante não é erro
GABARITO = {
    "x1": "direta", "x2": "direta", "x3": "direta", "x4": "direta",
    "x5": "redundante", "x6": "redundante",
    "x7": "ruido", "x8": "ruido", "x9": "ruido", "x10": "ruido",
}
PAR_INTERACAO = ("x1", "x2")
# Pares direta-redundante: aditivos no modelo gerador, mas a hipótese é que o
# PyGAwLLfs acuse interação neles (tirar um só quase não custa; tirar os dois custa)
PARES_REDUNDANCIA = [("x3", "x5"), ("x4", "x6")]


def gerar(
    n: int = N_AMOSTRAS,
    semente: int = SEMENTE_DADOS,
    beta: float = BETA_INTERACAO,
    sigma: float = SIGMA_RUIDO,
    rho: float = RHO_REDUNDANCIA,
) -> pd.DataFrame:
    """Gera n linhas com as colunas x1..x10 e y."""
    rng = np.random.default_rng(semente)
    x = {nome: rng.standard_normal(n) for nome in ["x1", "x2", "x3", "x4"]}

    # Correlação exata rho com a original, variância 1
    x["x5"] = rho * x["x3"] + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    x["x6"] = rho * x["x4"] + np.sqrt(1 - rho**2) * rng.standard_normal(n)

    for nome in ["x7", "x8", "x9", "x10"]:
        x[nome] = rng.standard_normal(n)

    z = beta * x["x1"] * x["x2"] + x["x3"] + x["x4"] + sigma * rng.standard_normal(n)

    df = pd.DataFrame({nome: x[nome] for nome in FEATURES})
    df["y"] = (z > 0).astype(int)
    return df


def dividir(df: pd.DataFrame, fracao_validacao: float = FRACAO_VALIDACAO,
            semente: int = SEMENTE_DADOS) -> pd.DataFrame:
    """Acrescenta a coluna 'conjunto' (treino/validacao), estratificada por y.

    Dado sintético não tem tempo, então aqui o split aleatório é legítimo.
    Não há conjunto de teste: quem julga é o gabarito, não uma métrica.
    """
    _, indice_validacao = train_test_split(
        df.index, test_size=fracao_validacao, stratify=df["y"], random_state=semente
    )
    df = df.copy()
    df["conjunto"] = "treino"
    df.loc[indice_validacao, "conjunto"] = "validacao"
    return df


def main() -> None:
    df = dividir(gerar())
    logger.info(f"Linhas por conjunto:\n{df['conjunto'].value_counts().to_string()}")
    logger.info(f"Taxa de y=1: {df['y'].mean():.3f}")
    logger.info(f"corr(x1, y) = {df['x1'].corr(df['y']):+.3f}  (interação pura: ~0)")
    logger.info(f"corr(x3, x5) = {df['x3'].corr(df['x5']):.3f}  (alvo: {RHO_REDUNDANCIA})")

    ARQUIVO_SINTETICO.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ARQUIVO_SINTETICO)
    logger.info(f"Salvo em {ARQUIVO_SINTETICO}")


if __name__ == "__main__":
    main()
