"""Testes da consolidação final (src/compare/sintese.py).

Script simples com assert (o pytest não está instalado). Rodar da raiz,
depois de sintese.py:
    python tests/test_sintese.py

Conferem a média de concordância num caso montado à mão, que o script só lê
CSV, e que a síntese reproduz exatamente as tabelas de onde sai.
"""

import logging
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

# Raiz do projeto: test_sintese.py -> tests/ -> raiz
RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ))

from src.compare import sintese  # noqa: E402
from src.importance import comum  # noqa: E402

logger = logging.getLogger(__name__)

DIR_TABELAS = RAIZ / "results" / "tables"
ENTRADAS = ["concordancia.csv", "estabilidade.csv", "fidelidade_resumo.csv", "fidelidade.csv",
            "custo_importancia.csv", "baselines.csv", "ranking_consolidado.csv"]


def teste_concordancia_intra():
    """Três métodos num modelo: a média de cada um usa só as duplas dele no mesmo
    modelo; a dupla entre modelos diferentes não entra.

    Usada em: bloco __main__ desta suíte (python tests/test_sintese.py; também em
    python main.py --testes).
    Importância: a concordância do quadro-síntese é sempre intra-modelo (§4.11).
    """
    concordancia = pd.DataFrame([
        {"method_a": "a", "model_a": "m", "method_b": "b", "model_b": "m",
         "tipo_par": "mesmo_modelo", "spearman": 0.9, "rbo": 0.8, "top5": 1.0},
        {"method_a": "a", "model_a": "m", "method_b": "c", "model_b": "m",
         "tipo_par": "mesmo_modelo", "spearman": 0.5, "rbo": 0.4, "top5": 0.6},
        {"method_a": "b", "model_a": "m", "method_b": "c", "model_b": "m",
         "tipo_par": "mesmo_modelo", "spearman": 0.1, "rbo": 0.2, "top5": 0.2},
        {"method_a": "a", "model_a": "m", "method_b": "a", "model_b": "outro",
         "tipo_par": "mesmo_metodo", "spearman": -1.0, "rbo": 0.0, "top5": 0.0},
    ])
    intra = sintese.concordancia_intra(concordancia).set_index("method")
    assert np.isclose(intra.loc["a", "concordancia_spearman"], 0.7)
    assert np.isclose(intra.loc["b", "concordancia_spearman"], 0.5)
    assert np.isclose(intra.loc["c", "concordancia_rbo"], 0.3)
    assert set(intra["model"]) == {"m"}


def teste_script_so_le_csv():
    """O script roda inteiro numa pasta temporária, sem abrir nenhum parquet.

    Usada em: bloco __main__ desta suíte (python tests/test_sintese.py; também em
    python main.py --testes).
    Importância: prova que a síntese não toca dados (nem o teste): só junta CSVs.
    """
    def proibir_parquet(*args, **kwargs):
        """Substituto de pandas.read_parquet (via mock.patch) que falha se for chamado.

        Usada em: teste_script_so_le_csv.
        Importância: qualquer leitura de dados na síntese vira erro.
        """
        raise AssertionError("A síntese não deveria ler parquet")

    with tempfile.TemporaryDirectory() as pasta:
        pasta = Path(pasta)
        for nome in ENTRADAS:
            shutil.copy(DIR_TABELAS / nome, pasta)
        with mock.patch("pandas.read_parquet", side_effect=proibir_parquet), \
                mock.patch.object(comum, "DIR_TABELAS", pasta):
            sintese.main()
        for esperado in ["sintese_metodos.csv", "sintese_modelos.csv", "sintese_top_features.csv"]:
            assert (pasta / esperado).exists(), f"faltou {esperado}"


def teste_coerencia_com_as_fontes():
    """As tabelas reais da síntese repetem exatamente os números das fontes.

    Usada em: bloco __main__ desta suíte (python tests/test_sintese.py; também em
    python main.py --testes).
    Importância: o relatório final cita a síntese; um erro de junção aqui
    contaminaria as respostas às perguntas de pesquisa.
    """
    metodos = pd.read_csv(DIR_TABELAS / "sintese_metodos.csv")
    assert len(metodos) == 11  # 4 métodos × 3 modelos, menos o SHAP da LogReg

    estabilidade = pd.read_csv(DIR_TABELAS / "estabilidade.csv").set_index(["method", "model"])
    resumo = pd.read_csv(DIR_TABELAS / "fidelidade_resumo.csv")
    resumo = resumo[(resumo["conjunto"] == "test") & (resumo["k"] == 5)].set_index(["method", "model"])
    for linha in metodos.itertuples():
        chave = (linha.method, linha.model)
        if linha.method == "intrinseca":
            assert np.isnan(linha.estabilidade_spearman)  # determinística
        else:
            assert np.isclose(linha.estabilidade_spearman, estabilidade.loc[chave, "spearman"], atol=1e-4)
        assert np.isclose(linha.fidelidade_top5, resumo.loc[chave, "pr_auc_frac_todas"], atol=1e-4)

    # O x* do PyGAwLLfs: nº médio de features = média do selecao_gawll.csv
    selecao = pd.read_csv(DIR_TABELAS / "selecao_gawll.csv").groupby("model")["n_features"].mean()
    gawll = metodos[metodos["method"] == "gawll"].set_index("model")
    assert np.allclose(gawll["n_features_x_estrela"], selecao.loc[gawll.index], atol=1e-4)

    # O top-5 listado é o mesmo conjunto que a fidelidade treinou com k = 5
    tops = pd.read_csv(DIR_TABELAS / "sintese_top_features.csv")
    fidelidade = pd.read_csv(DIR_TABELAS / "fidelidade.csv")
    fidelidade = fidelidade[(fidelidade["conjunto"] == "test") & (fidelidade["k"] == 5)]
    for linha in tops.itertuples():
        treinadas = fidelidade[(fidelidade["model"] == linha.model)
                               & (fidelidade["method"] == linha.method)]["features"].iloc[0]
        assert treinadas.split(";") == [linha.top1, linha.top2, linha.top3, linha.top4, linha.top5]

    # Desempenho dos modelos = Etapa 6
    modelos = pd.read_csv(DIR_TABELAS / "sintese_modelos.csv").set_index("model")
    baselines = pd.read_csv(DIR_TABELAS / "baselines.csv").set_index("modelo")
    for modelo in modelos.index:
        assert np.isclose(modelos.loc[modelo, "f1"], baselines.loc[modelo, "f1"], atol=1e-4)


if __name__ == "__main__":
    testes = [
        teste_concordancia_intra,
        teste_script_so_le_csv,
        teste_coerencia_com_as_fontes,
    ]
    for teste in testes:
        logger.info(f"{teste.__name__} ...")
        teste()
        logger.info(f"{teste.__name__} OK")
    logger.info(f"Todos os {len(testes)} testes passaram")
