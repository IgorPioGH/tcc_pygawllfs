"""Testes dos recortes temporais 2024 × 2025 (src/avaliacao/recortes.py).

Script simples com assert (o pytest não está instalado). Rodar da raiz,
depois de recortes.py:
    python tests/test_recortes.py

Conferem os conjuntos de cada recorte (janela de datas, divisão temporal,
50/50 e colunas), rodam o script inteiro em miniatura sem acesso ao teste de
nenhum recorte e conferem a tabela real.
"""

import logging
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd

# Raiz do projeto: test_recortes.py -> tests/ -> raiz
RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ))
sys.path.append(str(RAIZ / "PyGAwLLfs"))

from src.avaliacao import recortes  # noqa: E402
from src.modelagem import comum, consolidar_gawll, gawll, permutation  # noqa: E402
from src.preparacao.features import RECORTES, features_sem_alvo, montar_conjuntos  # noqa: E402

logger = logging.getLogger(__name__)

DIR_TABELAS = RAIZ / "results" / "tables"
DIR_DADOS = RAIZ / "data" / "processed"


def teste_conjuntos_dos_recortes():
    """Em cada recorte: datas dentro da janela, treino antes da validação antes do
    teste, 50/50 em todos os conjuntos e as mesmas 29 colunas, na mesma ordem, do
    train.parquet principal.

    Usada em: bloco __main__ desta suíte (python tests/test_recortes.py; também em
    python main.py --testes).
    Importância: os recortes só são comparáveis ao principal se a preparação for a
    mesma.
    """
    df = features_sem_alvo(pd.read_parquet(recortes.ARQUIVO_TARGET))
    colunas = list(pd.read_parquet(DIR_DADOS / "train.parquet").columns)
    for recorte in recortes.RECORTES_COMPARADOS:
        inicio, _, _, fim = RECORTES[recorte]
        conjuntos = montar_conjuntos(df, recorte)
        datas = {nome: c["Partida Prevista"] for nome, c in conjuntos.items()}
        for nome, conjunto in conjuntos.items():
            assert datas[nome].between(inicio, fim, inclusive="left").all(), (recorte, nome)
            assert 2 * conjunto["atrasado"].sum() == len(conjunto), (recorte, nome)
            assert list(conjunto.columns) == colunas, (recorte, nome)
        assert datas["train"].max() < datas["val"].min() < datas["val"].max() < datas["test"].min()


def teste_script_nao_le_o_teste():
    """O script roda inteiro em miniatura (LogReg, 2 gerações do GA, 1 repetição da
    Permutation) sem abrir test.parquet e sem acessar o teste de nenhum recorte.

    Usada em: bloco __main__ desta suíte (python tests/test_recortes.py; também em
    python main.py --testes).
    Importância: as importâncias dos recortes são calculadas na validação, como no
    principal; o teste do recorte nunca entra.
    """
    ler_parquet_original = pd.read_parquet

    def ler_parquet_vigiado(caminho, *args, **kwargs):
        """Substituto de pandas.read_parquet que falha se o arquivo for um teste.

        Usada em: teste_script_nao_le_o_teste.
        Importância: fiscal automático do isolamento do teste.
        """
        if Path(caminho).name.startswith("test"):
            raise AssertionError(f"Os recortes tentaram ler o teste: {caminho}")
        return ler_parquet_original(caminho, *args, **kwargs)

    def sem_teste(df, recorte):
        """montar_conjuntos sem a chave "test": qualquer acesso a ela vira KeyError.

        Usada em: teste_script_nao_le_o_teste.
        Importância: prova que o script não usa o teste do recorte.
        """
        conjuntos = montar_conjuntos(df, recorte)
        del conjuntos["test"]
        return conjuntos

    with tempfile.TemporaryDirectory() as pasta:
        pasta = Path(pasta)
        shutil.copy(DIR_TABELAS / "estabilidade.csv", pasta)
        with mock.patch("pandas.read_parquet", side_effect=ler_parquet_vigiado), \
                mock.patch.object(recortes, "montar_conjuntos", side_effect=sem_teste), \
                mock.patch.object(recortes, "MODELOS", ["logreg"]), \
                mock.patch.object(recortes, "DIR_EXECUCOES", pasta / "runs"), \
                mock.patch.object(recortes, "DIR_FIGURAS", pasta / "figuras"), \
                mock.patch.object(comum, "DIR_TABELAS", pasta), \
                mock.patch.object(permutation, "N_REPEATS", 1), \
                mock.patch.object(gawll, "GERACOES", 2), \
                mock.patch.object(consolidar_gawll, "GERACOES", 2):
            recortes.main()
        importancias = pd.read_csv(pasta / "importancia_recortes.csv")
        assert set(importancias["recorte"].astype(str)) == set(recortes.RECORTES_COMPARADOS)
        assert set(importancias["method"]) == {"intrinseca", "permutation", "gawll"}
        assert len(pd.read_csv(pasta / "recortes_concordancia.csv")) == 3
        assert (pasta / "figuras" / "recortes.png").exists()


def teste_tabela_real():
    """importancia_recortes.csv: 2 recortes × 11 tabelas (3 intrínseca + 3 Permutation
    + 2 SHAP + 3 PyGAwLLfs) × 29 atributos = 638 linhas; acurácia de validação acima
    de 0,5 nos dois recortes; uma linha de concordância por (método, modelo).

    Usada em: bloco __main__ desta suíte (python tests/test_recortes.py; também em
    python main.py --testes).
    Importância: confere que nenhum (recorte, método, modelo) ficou de fora.
    """
    importancias = pd.read_csv(DIR_TABELAS / "importancia_recortes.csv")
    assert len(importancias) == 638, len(importancias)
    assert importancias.groupby(["recorte", "method", "model"]).size().eq(29).all()
    desempenho = pd.read_csv(DIR_TABELAS / "recortes_desempenho.csv")
    assert len(desempenho) == 6 and (desempenho["acuracia"] > 0.5).all()
    concordancia = pd.read_csv(DIR_TABELAS / "recortes_concordancia.csv")
    assert len(concordancia) == 11 and concordancia["spearman"].between(-1, 1).all()


if __name__ == "__main__":
    testes = [
        teste_conjuntos_dos_recortes,
        teste_script_nao_le_o_teste,
        teste_tabela_real,
    ]
    for teste in testes:
        logger.info(f"{teste.__name__} ...")
        teste()
        logger.info(f"{teste.__name__} OK")
    logger.info(f"Todos os {len(testes)} testes passaram")
