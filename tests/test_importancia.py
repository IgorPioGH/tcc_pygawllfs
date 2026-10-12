"""Testes da Etapa 9 — métodos de importância (src/importance/).

Script simples com assert (o pytest não está instalado). Rodar da raiz,
depois de rodar os scripts da Etapa 9:
    python tests/test_importancia.py

Verifica a regra de rank do formato padrão, o isolamento do teste (§4.9),
a coerência da importância intrínseca com a Etapa 6 e a consolidação das
execuções do PyGAwLLfs.
"""

import logging
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

# Raiz do projeto: test_importancia.py -> tests/ -> raiz
RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ))

from src.importance import comum, consolidar_gawll, intrinsic, permutation, shap_global  # noqa: E402

logger = logging.getLogger(__name__)

DIR_TABELAS = RAIZ / "results" / "tables"


def teste_regra_de_rank():
    """Rank 1 = mais importante; coeficiente pelo absoluto; negativo no fim.

    Usada em: bloco __main__ desta suíte (python tests/test_importancia.py; também em
    python main.py --testes).
    Importância: a regra de rank é a mesma para todos os métodos comparados.
    """
    features = ["a", "b", "c"]

    tabela = comum.tabela_importancia(features, [0.1, -0.5, 0.3], "x", "m", 42)
    assert dict(zip(tabela["feature"], tabela["rank"])) == {"c": 1, "a": 2, "b": 3}

    tabela = comum.tabela_importancia(
        features, [0.1, -0.5, 0.3], "x", "m", 42, rank_por_absoluto=True
    )
    assert dict(zip(tabela["feature"], tabela["rank"])) == {"b": 1, "c": 2, "a": 3}
    assert tabela.loc[tabela["feature"] == "b", "importance"].item() == -0.5, "sinal perdido"
    assert list(tabela.columns) == comum.COLUNAS_PADRAO


def teste_nenhum_metodo_le_o_teste():
    """§4.9: os quatro scripts rodam inteiros (em miniatura) sem abrir test.parquet.

    Usada em: bloco __main__ desta suíte (python tests/test_importancia.py; também em
    python main.py --testes).
    Importância: garante que nenhuma importância é calculada no teste.
    """
    ler_parquet_original = pd.read_parquet

    def ler_parquet_vigiado(caminho, *args, **kwargs):
        """Substituto de pandas.read_parquet (via mock.patch) que falha se o arquivo for
        o conjunto de teste e, senão, lê normalmente.

        Usada em: teste_nenhum_metodo_le_o_teste.
        Importância: é o fiscal automático do §4.9 (docs/metodologia.md).
        """
        if Path(caminho).name.startswith("test"):
            raise AssertionError(f"Um método de importância tentou ler o teste: {caminho}")
        return ler_parquet_original(caminho, *args, **kwargs)

    with tempfile.TemporaryDirectory() as pasta, \
            mock.patch("pandas.read_parquet", side_effect=ler_parquet_vigiado), \
            mock.patch.object(comum, "DIR_TABELAS", Path(pasta)), \
            mock.patch.object(permutation, "SEMENTES", [42]), \
            mock.patch.object(permutation, "N_REPEATS", 1), \
            mock.patch.object(shap_global, "SEMENTES", [42]), \
            mock.patch.object(shap_global, "TAMANHO_AMOSTRA", 200):
        intrinsic.main()
        permutation.main()
        shap_global.main()
        consolidar_gawll.main()

        gerados = {p.name for p in Path(pasta).iterdir()}
        esperados = {
            "importancia_intrinseca.csv", "importancia_permutation.csv",
            "importancia_shap.csv", "importancia_gawll.csv", "interacao_gawll.csv",
            "selecao_gawll.csv", "custo_importancia.csv",
        }
        assert esperados <= gerados, f"faltaram: {esperados - gerados}"

        custo = pd.read_csv(Path(pasta) / "custo_importancia.csv")
        assert set(custo["method"]) == {"intrinseca", "permutation", "shap", "gawll"}


def teste_formato_das_saidas():
    """Cada (método, modelo, semente) tem as 29 features, ranks válidos e nada vazio.

    Usada em: bloco __main__ desta suíte (python tests/test_importancia.py; também em
    python main.py --testes).
    Importância: a Etapa 11 depende desse formato.
    """
    for nome in ["intrinseca", "permutation", "shap", "gawll"]:
        tabela = pd.read_csv(DIR_TABELAS / f"importancia_{nome}.csv")
        assert list(tabela.columns) == comum.COLUNAS_PADRAO, nome
        assert not tabela.isna().any().any(), f"{nome} tem valores vazios"
        for (modelo, semente), grupo in tabela.groupby(["model", "seed"]):
            assert grupo["feature"].nunique() == len(grupo) == 29, (nome, modelo, semente)
            assert grupo["rank"].min() == 1 and grupo["rank"].max() <= 29, (nome, modelo, semente)


def teste_intrinseca_bate_com_baselines():
    """O coeficiente da LogReg aqui é o mesmo salvo pela Etapa 6.

    Usada em: bloco __main__ desta suíte (python tests/test_importancia.py; também em
    python main.py --testes).
    Importância: confirma que os métodos explicam o mesmo modelo.
    """
    intrinseca = pd.read_csv(DIR_TABELAS / "importancia_intrinseca.csv")
    intrinseca = intrinseca[intrinseca["model"] == "logreg"].set_index("feature")
    etapa6 = pd.read_csv(DIR_TABELAS / "coeficientes_logreg.csv").set_index("feature")
    assert np.allclose(intrinseca["importance"], etapa6.loc[intrinseca.index, "importance"])


def teste_consolidacao_gawll():
    """As tabelas reproduzem exatamente o vimp e a vint dos checkpoints.

    Usada em: bloco __main__ desta suíte (python tests/test_importancia.py; também em
    python main.py --testes).
    Importância: nada se perde entre o GA e as tabelas comparadas.
    """
    importancia = pd.read_csv(DIR_TABELAS / "importancia_gawll.csv")
    interacao = pd.read_csv(DIR_TABELAS / "interacao_gawll.csv")

    for arquivo in sorted(consolidar_gawll.DIR_EXECUCOES.glob("*_seed*.npz")):
        execucao = np.load(arquivo)
        modelo, semente = str(execucao["modelo"]), int(execucao["semente"])
        features = [str(f) for f in execucao["features"]]
        n = len(features)

        vimp = (importancia.query("model == @modelo and seed == @semente")
                .set_index("feature").loc[features, "importance"])
        assert np.allclose(vimp, execucao["vimp"]), arquivo.name

        pares = interacao.query("model == @modelo and seed == @semente")
        assert len(pares) == n * (n - 1) // 2, arquivo.name
        posicao = {f: i for i, f in enumerate(features)}
        reconstruida = np.zeros((n, n))
        for a, b, valor in zip(pares["feature_a"], pares["feature_b"], pares["interaction"]):
            reconstruida[posicao[a], posicao[b]] = reconstruida[posicao[b], posicao[a]] = valor
        assert np.allclose(reconstruida, execucao["vint"]), arquivo.name


if __name__ == "__main__":
    testes = [
        teste_regra_de_rank,
        teste_nenhum_metodo_le_o_teste,
        teste_formato_das_saidas,
        teste_intrinseca_bate_com_baselines,
        teste_consolidacao_gawll,
    ]
    for teste in testes:
        logger.info(f"{teste.__name__} ...")
        teste()
        logger.info(f"{teste.__name__} OK")
    logger.info(f"Todos os {len(testes)} testes passaram")
