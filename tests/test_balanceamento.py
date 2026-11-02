"""Testes do balanceamento 50/50 (src/preparacao/features.py).

Script simples com assert (o pytest não está instalado). Rodar da raiz,
depois de features.py:
    python tests/test_balanceamento.py

Conferem que treino, validação e teste têm exatamente 50% de atrasados, que
nenhum voo atrasado se perdeu, que a divisão temporal continua valendo, que o
congestionamento foi calculado antes do descarte e que a subamostra do
PyGAwLLfs segue as mesmas regras.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Raiz do projeto: test_balanceamento.py -> tests/ -> raiz
RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ))

from src.preparacao.features import RECORTES, features_sem_alvo  # noqa: E402

logger = logging.getLogger(__name__)

DIR_DADOS = RAIZ / "data" / "processed"
ARQUIVO_TARGET = RAIZ / "data" / "interim" / "vra_target.parquet"
CONJUNTOS = ["train", "val", "test"]


def carregar_conjuntos() -> dict[str, pd.DataFrame]:
    """Os três conjuntos congelados, por nome.

    Usada em: todos os testes desta suíte.
    Importância: os testes olham os arquivos que os modelos de fato leem.
    """
    return {nome: pd.read_parquet(DIR_DADOS / f"{nome}.parquet") for nome in CONJUNTOS}


def janelas_principais() -> dict[str, tuple[str, str]]:
    """Intervalo [início, fim) de datas de cada conjunto do recorte principal.

    Usada em: teste_nenhum_atrasado_perdido.
    Importância: compara cada conjunto com a mesma janela da base sem balancear.
    """
    inicio, corte_treino, corte_val, fim = RECORTES["principal"]
    return {"train": (inicio, corte_treino), "val": (corte_treino, corte_val),
            "test": (corte_val, fim)}


def teste_cinquenta_por_cento():
    """Exatamente metade de atrasados em treino, validação e teste, sem índice repetido.

    Usada em: bloco __main__ desta suíte (python tests/test_balanceamento.py; também
    em python main.py --testes).
    Importância: o 50/50 é o que torna a acurácia legível (o trivial acerta 0,5).
    """
    for nome, conjunto in carregar_conjuntos().items():
        assert 2 * conjunto["atrasado"].sum() == len(conjunto), nome
        assert not conjunto.index.duplicated().any(), nome
        logger.info(f"  {nome}: {len(conjunto)} voos, {conjunto['atrasado'].sum()} atrasados")


def teste_nenhum_atrasado_perdido():
    """Cada conjunto tem todos os atrasados da sua janela de datas na base sem
    balancear: o balanceamento só descarta voos pontuais.

    Usada em: bloco __main__ desta suíte (python tests/test_balanceamento.py; também
    em python main.py --testes).
    Importância: descartar atrasados mudaria o que o modelo aprende sobre atraso.
    """
    base = pd.read_parquet(ARQUIVO_TARGET, columns=["Partida Prevista", "atrasado"])
    for nome, conjunto in carregar_conjuntos().items():
        inicio, fim = janelas_principais()[nome]
        janela = base[(base["Partida Prevista"] >= inicio) & (base["Partida Prevista"] < fim)]
        atrasados = janela.index[janela["atrasado"] == 1]
        assert conjunto["atrasado"].sum() == len(atrasados), nome
        assert set(conjunto.index[conjunto["atrasado"] == 1]) == set(atrasados), nome


def teste_divisao_temporal():
    """max(treino) < min(validação) e max(validação) < min(teste), depois do
    balanceamento.

    Usada em: bloco __main__ desta suíte (python tests/test_balanceamento.py; também
    em python main.py --testes).
    Importância: o balanceamento não pode misturar períodos.
    """
    conjuntos = carregar_conjuntos()
    datas = {nome: c["Partida Prevista"] for nome, c in conjuntos.items()}
    assert datas["train"].max() < datas["val"].min()
    assert datas["val"].max() < datas["test"].min()
    for nome, serie in datas.items():
        assert serie.is_monotonic_increasing, f"{nome} fora da ordem de Partida Prevista"


def teste_congestionamento_antes_do_descarte():
    """congestion_origem das linhas mantidas é o mesmo calculado na base inteira.

    Usada em: bloco __main__ desta suíte (python tests/test_balanceamento.py; também
    em python main.py --testes).
    Importância: o congestionamento conta os voos programados da malha inteira;
    calculado depois do descarte, ele contaria só metade dos voos.
    """
    referencia = features_sem_alvo(pd.read_parquet(ARQUIVO_TARGET))
    for nome, conjunto in carregar_conjuntos().items():
        for coluna in ["congestion_origem", "congestion_destino"]:
            esperado = referencia.loc[conjunto.index, coluna]
            assert (conjunto[coluna] == esperado).all(), (nome, coluna)


def teste_subamostra():
    """Subamostra do PyGAwLLfs: 50 mil linhas, só de treino e validação, sem linha do
    teste e com taxa de atraso entre 49,5% e 50,5%.

    Usada em: bloco __main__ desta suíte (python tests/test_balanceamento.py; também
    em python main.py --testes).
    Importância: o GA explica modelos treinados nessa amostra; ela tem de seguir as
    mesmas regras dos conjuntos completos.
    """
    subamostra = pd.read_parquet(DIR_DADOS / "subamostra.parquet")
    teste = pd.read_parquet(DIR_DADOS / "test.parquet", columns=[])
    assert len(subamostra) == 50_000
    assert set(subamostra["origem"]) <= {"treino", "validacao"}
    assert subamostra.index.intersection(teste.index).empty
    assert 0.495 <= subamostra["atrasado"].mean() <= 0.505, subamostra["atrasado"].mean()


if __name__ == "__main__":
    testes = [
        teste_cinquenta_por_cento,
        teste_nenhum_atrasado_perdido,
        teste_divisao_temporal,
        teste_congestionamento_antes_do_descarte,
        teste_subamostra,
    ]
    for teste in testes:
        logger.info(f"{teste.__name__} ...")
        teste()
        logger.info(f"{teste.__name__} OK")
    logger.info(f"Todos os {len(testes)} testes passaram")
