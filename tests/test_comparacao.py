"""Testes da Etapa 11 — comparação de rankings (src/compare/).

Script simples com assert (o pytest não está instalado). Rodar da raiz,
depois de rankings.py, stability.py e figuras.py:
    python tests/test_comparacao.py

Os quatro primeiros conferem as métricas em casos montados à mão; o quinto
roda os três scripts inteiros numa pasta temporária, sem nenhum parquet
(a Etapa 11 só lê CSVs); o último confere as tabelas reais.
"""

import logging
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

# Raiz do projeto: test_comparacao.py -> tests/ -> raiz
RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ))

from src.compare import figuras, rankings, stability  # noqa: E402
from src.importance import comum  # noqa: E402

logger = logging.getLogger(__name__)

DIR_TABELAS = RAIZ / "results" / "tables"
FEATURES = [f"f{i:02d}" for i in range(29)]


def teste_rbo():
    """Valores conhecidos e as propriedades que justificam usar o RBO."""
    # isclose: a soma da série dá 0,9999999999999998 em ponto flutuante
    assert np.isclose(rankings.rbo(FEATURES, FEATURES), 1.0)

    # Exemplo calculado à mão: X1 = 0, X2 = 2, X3 = 3
    # RBO = 0,9^3 + (0,1/0,9) * (0 + 0,81 + 0,729) = 0,729 + 0,171 = 0,900
    assert abs(rankings.rbo(["a", "b", "c"], ["b", "a", "c"]) - 0.9) < 1e-12

    invertida = FEATURES[::-1]
    assert rankings.rbo(FEATURES, invertida) == rankings.rbo(invertida, FEATURES), "não é simétrico"
    assert rankings.rbo(FEATURES, invertida) < 0.3

    # Pesa mais o topo: trocar as 2 primeiras custa mais que trocar as 2 últimas
    troca_topo = [FEATURES[1], FEATURES[0]] + FEATURES[2:]
    troca_fim = FEATURES[:-2] + [FEATURES[-1], FEATURES[-2]]
    assert rankings.rbo(FEATURES, troca_topo) < rankings.rbo(FEATURES, troca_fim)


def teste_metricas_e_ordenacao():
    """Spearman/Kendall nos extremos, top-k e desempate determinístico."""
    crescente = pd.Series(np.arange(29, 0, -1, dtype=float), index=FEATURES)
    igual = rankings.comparar(crescente, crescente)
    assert np.isclose([igual["spearman"], igual["kendall_tau_b"], igual["rbo"]], 1.0).all()
    assert igual["top5"] == igual["top10"] == igual["top15"] == 1.0

    oposto = rankings.comparar(crescente, -crescente)
    assert abs(oposto["spearman"] + 1) < 1e-12 and abs(oposto["kendall_tau_b"] + 1) < 1e-12
    assert oposto["top5"] == 0.0

    # A ordem das linhas não pode mudar o resultado
    embaralhada = crescente.sample(frac=1, random_state=0)
    assert rankings.comparar(crescente, embaralhada) == igual

    assert rankings.sobreposicao_top_k(["a", "b", "c"], ["c", "x", "a"], 2) == 0.0
    assert rankings.sobreposicao_top_k(["a", "b", "c"], ["c", "x", "a"], 3) == 2 / 3
    assert rankings.ordenar(pd.Series([1.0, 1.0, 2.0], index=["b", "a", "c"])) == ["c", "a", "b"]


def teste_pontuacao_normalizacao_e_tipo_de_par():
    tabela = pd.DataFrame({
        "feature": ["a", "b", "a", "b"],
        "importance": [-0.5, 0.2, -0.5, 0.2],
        "method": ["intrinseca", "intrinseca", "permutation", "permutation"],
        "model": ["logreg", "logreg", "logreg", "logreg"],
    })
    pontos = rankings.pontuacao(tabela)
    assert list(pontos) == [0.5, 0.2, -0.5, 0.2], "só o coeficiente usa o absoluto"

    normalizado = rankings.normalizar_magnitude(pd.Series([-0.3, 0.0, 0.5, 1.0]))
    assert list(normalizado) == [0.0, 0.0, 0.5, 1.0], "negativo vira 0, depois min-max"
    assert (rankings.normalizar_magnitude(pd.Series([0.2, 0.2])) == 0).all()

    assert rankings.tipo_de_par(("shap", "xgboost"), ("gawll", "xgboost")) == "mesmo_modelo"
    assert rankings.tipo_de_par(("gawll", "logreg"), ("gawll", "xgboost")) == "mesmo_metodo"
    assert rankings.tipo_de_par(("shap", "random_forest"), ("gawll", "logreg")) == "diferentes"


def teste_estabilidade_de_sementes_iguais():
    """Cinco sementes idênticas -> concordância 1 e frequência 0 ou 1."""
    linhas = []
    for semente in range(5):
        tabela = comum.tabela_importancia(FEATURES, np.arange(29, 0, -1), "m", "modelo", semente)
        linhas.append(tabela)
    grupo = pd.concat(linhas)
    grupo["pontuacao"] = grupo["importance"]

    resultado = stability.concordancia_entre_sementes(grupo)
    assert resultado["n_duplas"] == 10
    assert np.isclose([resultado["spearman_min"], resultado["rbo_min"]], 1.0).all()

    por_feature = stability.estabilidade_por_feature(grupo)
    assert (por_feature["rank_desvio"] == 0).all()
    assert set(por_feature["freq_top10"]) == {0.0, 1.0}
    assert por_feature.iloc[0]["feature"] == "f00"


def teste_scripts_so_leem_csv():
    """Os três scripts rodam inteiros numa pasta temporária, sem abrir nenhum parquet."""
    def proibir_parquet(*args, **kwargs):
        raise AssertionError("A Etapa 11 não deveria ler parquet")

    with tempfile.TemporaryDirectory() as pasta:
        pasta = Path(pasta)
        for metodo in rankings.METODOS:
            shutil.copy(DIR_TABELAS / f"importancia_{metodo}.csv", pasta)
        with mock.patch("pandas.read_parquet", side_effect=proibir_parquet), \
                mock.patch.object(comum, "DIR_TABELAS", pasta), \
                mock.patch.object(figuras, "DIR_FIGURAS", pasta / "figuras"):
            rankings.main()
            stability.main()
            figuras.main()

        gerados = {p.name for p in pasta.rglob("*") if p.is_file()}
        for esperado in ["ranking_consolidado.csv", "concordancia.csv", "concordancia_resumo.csv",
                         "estabilidade.csv", "estabilidade_features.csv",
                         "concordancia_intra_modelo.png", "concordancia_geral.png",
                         "estabilidade_sementes.png"]:
            assert esperado in gerados, f"faltou {esperado}"


def teste_tabelas_reais():
    """Estrutura e faixas das saídas da Etapa 11."""
    consolidado = pd.read_csv(DIR_TABELAS / "ranking_consolidado.csv")
    assert consolidado.groupby(["method", "model"]).size().eq(29).all()
    assert len(consolidado.groupby(["method", "model"])) == 11
    assert consolidado["importance_norm"].between(0, 1).all()

    concordancia = pd.read_csv(DIR_TABELAS / "concordancia.csv")
    assert len(concordancia) == 55  # 11 rankings, 2 a 2
    assert concordancia["tipo_par"].value_counts().to_dict() == {
        "diferentes": 30, "mesmo_modelo": 15, "mesmo_metodo": 10}
    for coluna in ["spearman", "kendall_tau_b"]:
        assert concordancia[coluna].between(-1, 1).all()
    for coluna in ["rbo", "top5", "top10", "top15"]:
        assert concordancia[coluna].between(0, 1).all()

    estabilidade = pd.read_csv(DIR_TABELAS / "estabilidade.csv")
    assert len(estabilidade) == 8  # (permutation, shap, gawll) x modelos, sem SHAP-LogReg
    assert (estabilidade["n_duplas"] == 10).all()
    assert (estabilidade["spearman_min"] <= estabilidade["spearman"]).all()


if __name__ == "__main__":
    testes = [
        teste_rbo,
        teste_metricas_e_ordenacao,
        teste_pontuacao_normalizacao_e_tipo_de_par,
        teste_estabilidade_de_sementes_iguais,
        teste_scripts_so_leem_csv,
        teste_tabelas_reais,
    ]
    for teste in testes:
        logger.info(f"{teste.__name__} ...")
        teste()
        logger.info(f"{teste.__name__} OK")
    logger.info(f"Todos os {len(testes)} testes passaram")
