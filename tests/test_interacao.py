"""Testes da Etapa 13 — H de Friedman e comparação com o PyGAwLLfs.

Script simples com assert (o pytest não está instalado). Rodar da raiz,
depois de friedman_h.py, compare/interacoes.py e synthetic/aplicar_metodos.py:
    python tests/test_interacao.py

Os quatro primeiros conferem o H e o grafo em casos com resposta conhecida
(inclusive a justificativa de deixar a LogReg de fora); o quinto roda os
scripts inteiros em miniatura com o test.parquet bloqueado (§4.9); o último
confere as tabelas reais.
"""

import logging
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

# Raiz do projeto: test_interacao.py -> tests/ -> raiz
RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ))

from src.compare import interacoes  # noqa: E402
from src.domain import validate_pairs  # noqa: E402
from src.importance import comum  # noqa: E402
from src.interaction import friedman_h, graph  # noqa: E402
from src.models.baselines import carregar  # noqa: E402

logger = logging.getLogger(__name__)

DIR_TABELAS = RAIZ / "results" / "tables"


def teste_h_em_funcoes_conhecidas():
    """x1·x2 + x3: só o par (x1, x2) interage. Função aditiva: nenhum par."""
    rng = np.random.default_rng(0)
    amostra = pd.DataFrame(rng.standard_normal((200, 4)), columns=["x1", "x2", "x3", "x4"])

    com_interacao = friedman_h.calcular_h(
        lambda x: (x["x1"] * x["x2"] + x["x3"]).to_numpy(), amostra, list(amostra.columns))
    par = com_interacao.set_index(["feature_a", "feature_b"])
    assert par.loc[("x1", "x2"), "h_normalizado"] > 0.95
    outros = par.drop(index=("x1", "x2"))
    assert (outros["h_abs"] < 1e-12).all() and (outros["h_normalizado"] < 1e-12).all()

    aditiva = friedman_h.calcular_h(
        lambda x: (2 * x["x1"] - x["x2"] + np.sin(x["x3"])).to_numpy(), amostra,
        list(amostra.columns))
    assert (aditiva["h_abs"] < 1e-12).all()


def teste_dependencia_parcial():
    """PD centrada; numa função linear, PD_j(x) = coeficiente × (x − média)."""
    rng = np.random.default_rng(1)
    amostra = pd.DataFrame(rng.standard_normal((100, 2)), columns=["x1", "x2"])
    pd_x1 = friedman_h.dependencia_parcial(lambda x: (3 * x["x1"] + x["x2"]).to_numpy(),
                                           amostra, ["x1"])
    assert abs(pd_x1.mean()) < 1e-12
    assert np.allclose(pd_x1, 3 * (amostra["x1"] - amostra["x1"].mean()))


def teste_logreg_tem_h_zero():
    """Justifica deixar a LogReg de fora: no logit ela é aditiva, então H = 0."""
    x_val, _ = carregar("val")
    amostra = x_val.sample(n=60, random_state=0)
    prever = friedman_h.prever_logit(comum.carregar_modelo("logreg"))
    h = friedman_h.calcular_h(prever, amostra, ["hora", "dia_ano", "empresa_te", "origem_te"])
    assert h["h_abs"].max() < 1e-6, h["h_abs"].max()


def teste_grafo_e_conjunto_do_h():
    media = graph.interacao_media(pd.read_csv(DIR_TABELAS / "interacao_gawll.csv"))
    assert media.groupby(["method", "model"])["rank"].min().eq(1).all()
    arestas = graph.arestas_reduzidas(media)
    assert arestas.groupby("model").size().eq(round(406 * graph.FRACAO_ARESTAS)).all()

    # §4.7: o conjunto do H contém o grafo do PyGAwLLfs inteiro e o top-15 por consenso
    ranking = pd.read_csv(DIR_TABELAS / "ranking_consolidado.csv")
    for modelo in friedman_h.MODELOS_H:
        conjunto = set(friedman_h.conjunto_de_features(modelo))
        assert graph.features_do_grafo(arestas[arestas["model"] == modelo]) <= conjunto
        consenso = (ranking[ranking["model"] == modelo].groupby("feature")["rank"]
                    .mean().nsmallest(friedman_h.TOP_CONSENSO).index)
        assert set(consenso) <= conjunto


def teste_scripts_nao_leem_o_teste():
    """§4.9: H e comparação rodam inteiros (em miniatura) sem abrir test.parquet."""
    ler_parquet_original = pd.read_parquet

    def ler_parquet_vigiado(caminho, *args, **kwargs):
        if Path(caminho).name.startswith("test"):
            raise AssertionError(f"A Etapa 13 tentou ler o teste: {caminho}")
        return ler_parquet_original(caminho, *args, **kwargs)

    with tempfile.TemporaryDirectory() as pasta:
        pasta = Path(pasta)
        for nome in ["interacao_gawll.csv", "ranking_consolidado.csv", "dominio_pares.csv"]:
            shutil.copy(DIR_TABELAS / nome, pasta)
        with mock.patch("pandas.read_parquet", side_effect=ler_parquet_vigiado), \
                mock.patch.object(comum, "DIR_TABELAS", pasta), \
                mock.patch.object(comum, "SEMENTES", [42, 43]), \
                mock.patch.object(friedman_h, "AMOSTRA", 30), \
                mock.patch.object(validate_pairs, "N_BOOTSTRAP", 5), \
                mock.patch.object(interacoes, "DIR_FIGURAS", pasta / "figuras"):
            friedman_h.main()
            interacoes.main()
        for esperado in ["interacao_h.csv", "interacoes.csv", "interacoes_pares.csv",
                         "interacoes_redundancia.csv", "interacoes_dominio.csv"]:
            assert (pasta / esperado).exists(), esperado
        assert (pasta / "figuras" / "interacao_gawll_x_h.png").exists()


def teste_tabelas_reais():
    h = pd.read_csv(DIR_TABELAS / "interacao_h.csv")
    assert set(h["method"]) == {"friedman_h_abs", "friedman_h"}
    assert set(h["model"]) == set(friedman_h.MODELOS_H)
    for (metodo, modelo, semente), grupo in h.groupby(["method", "model", "seed"]):
        n = len(friedman_h.conjunto_de_features(modelo))
        assert len(grupo) == n * (n - 1) // 2, (metodo, modelo, semente)
        assert (grupo["interaction"] >= 0).all()
    # O H² normalizado NÃO é limitado a 1: quando o efeito conjunto é quase nulo,
    # o denominador some. Aconteceu 1 vez (porte_origem_grande × porte_origem_hub,
    # duas dummies da mesma variável, H_abs minúsculo). Por isso o H_abs é o principal.
    normalizado = h.loc[h["method"] == "friedman_h", "interaction"]
    assert (normalizado > 1).mean() < 0.01

    comparacao = pd.read_csv(DIR_TABELAS / "interacoes.csv")
    assert comparacao["spearman"].between(-1, 1).all()

    # Top-10 de cada método × modelo no teste de domínio, e números iguais aos da Etapa 15
    dominio_tops = pd.read_csv(DIR_TABELAS / "interacoes_dominio.csv")
    assert dominio_tops.groupby(["model", "method"]).size().eq(10).all()
    etapa15 = pd.read_csv(DIR_TABELAS / "dominio_pares.csv").set_index(["feature_a", "feature_b"])
    for linha in dominio_tops.itertuples():
        if (linha.feature_a, linha.feature_b) in etapa15.index:
            assert linha.classificacao == etapa15.loc[(linha.feature_a, linha.feature_b),
                                                      "classificacao"]

    # Diferencial 1: o H acha o par verdadeiro em 1º, e não põe a redundância no topo
    recuperacao = pd.read_csv(DIR_TABELAS / "recuperacao_sintetico_interacao.csv")
    h_sintetico = recuperacao[recuperacao["method"].str.startswith("friedman")]
    assert len(h_sintetico) == 4
    assert (h_sintetico["rank_par_interacao"] == 1).all()


if __name__ == "__main__":
    testes = [
        teste_h_em_funcoes_conhecidas,
        teste_dependencia_parcial,
        teste_logreg_tem_h_zero,
        teste_grafo_e_conjunto_do_h,
        teste_scripts_nao_leem_o_teste,
        teste_tabelas_reais,
    ]
    for teste in testes:
        logger.info(f"{teste.__name__} ...")
        teste()
        logger.info(f"{teste.__name__} OK")
    logger.info(f"Todos os {len(testes)} testes passaram")
