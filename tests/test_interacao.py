"""Testes da Etapa 13 — H de Friedman e comparação com o PyGAwLLfs.

Script simples com assert (o pytest não está instalado). Rodar da raiz,
depois de friedman_h.py, avaliacao/interacoes.py e avaliacao/sintetico_aplicar.py:
    python tests/test_interacao.py

Os primeiros conferem o H, o tipo dos pares e a agregação das sementes em
casos com resposta conhecida (inclusive a justificativa de deixar a LogReg de
fora); o penúltimo roda os scripts inteiros em miniatura com o test.parquet
bloqueado (§4.9); o último confere as tabelas reais.
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

from src.avaliacao import interacoes  # noqa: E402
from src.modelagem import comum  # noqa: E402
from src.avaliacao import graph  # noqa: E402
from src.modelagem import friedman_h  # noqa: E402
from src.modelagem.baselines import carregar  # noqa: E402

logger = logging.getLogger(__name__)

DIR_TABELAS = RAIZ / "results" / "tables"


def teste_h_em_funcoes_conhecidas():
    """x1·x2 + x3: só o par (x1, x2) interage. Função aditiva: nenhum par.

    Usada em: bloco __main__ desta suíte (python tests/test_interacao.py; também em
    python main.py --testes).
    Importância: valida o H em casos de resposta conhecida.
    """
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
    """PD centrada; numa função linear, PD_j(x) = coeficiente × (x − média).

    Usada em: bloco __main__ desta suíte (python tests/test_interacao.py; também em
    python main.py --testes).
    Importância: a PD é o bloco básico do H.
    """
    rng = np.random.default_rng(1)
    amostra = pd.DataFrame(rng.standard_normal((100, 2)), columns=["x1", "x2"])
    pd_x1 = friedman_h.dependencia_parcial(lambda x: (3 * x["x1"] + x["x2"]).to_numpy(),
                                           amostra, ["x1"])
    assert abs(pd_x1.mean()) < 1e-12
    assert np.allclose(pd_x1, 3 * (amostra["x1"] - amostra["x1"].mean()))


def teste_logreg_tem_h_zero():
    """Justifica deixar a LogReg de fora: no logit ela é aditiva, então H = 0.

    Usada em: bloco __main__ desta suíte (python tests/test_interacao.py; também em
    python main.py --testes).
    Importância: sustenta a decisão de calcular o H só em RF e XGBoost.
    """
    x_val, _ = carregar("val")
    amostra = x_val.sample(n=60, random_state=0)
    prever = friedman_h.prever_logit(comum.carregar_modelo("logreg"))
    h = friedman_h.calcular_h(prever, amostra, ["hora", "dia_ano", "empresa_te", "origem_te"])
    assert h["h_abs"].max() < 1e-6, h["h_abs"].max()


def teste_tipo_por_construcao():
    """Casos conhecidos: duas dummies da mesma coluna (e o porte com o target encoding
    do mesmo aeroporto) são estruturais; hora e período do dia são redundantes; hora e
    empresa são operacionais.

    Usada em: bloco __main__ desta suíte (python tests/test_interacao.py; também em
    python main.py --testes).
    Importância: o tipo do par é a régua da hipótese da redundância.
    """
    assert graph.tipo_por_construcao("tipo_linha_C", "tipo_linha_I") == "estrutural"
    assert graph.tipo_por_construcao("porte_origem_hub", "origem_te") == "estrutural"
    assert graph.tipo_por_construcao("porte_origem_hub", "porte_destino_hub") == "operacional"
    assert graph.tipo_por_construcao("hora", "periodo_dia") == "redundante"
    assert graph.tipo_por_construcao("dia_ano", "mes") == "redundante"
    assert graph.tipo_por_construcao("hora", "empresa_te") == "operacional"


def teste_agregacao_das_sementes():
    """Decisão 15: o zero do PyGAwLLfs fica fora da média se o par não foi sondado
    naquela semente e entra se foi (coluna "sondada"); par nunca sondado não tem média
    e par sondado numa semente só tem média mas não tem rank; no H, o zero é medida.
    Sem a coluna "sondada", a agregação recusa a tabela.

    Usada em: bloco __main__ desta suíte (python tests/test_interacao.py; também em
    python main.py --testes).
    Importância: é a regra que decide quais pares entram no ranking, no top-k e no grafo.
    """
    # (valores por semente, sementes em que o par foi sondado)
    casos = {("a", "b"): ([0.2, 0, 0, 0.4, 0], [1, 0, 0, 1, 0]),   # sondado em 2 sementes
             ("a", "c"): ([0.9, 0, 0, 0, 0], [1, 0, 0, 0, 0]),     # sondado em 1 semente
             ("b", "c"): ([0, 0, 0, 0, 0], [0, 0, 0, 0, 0])}       # nunca sondado
    linhas = [{"feature_a": a, "feature_b": b, "interaction": x, "method": "gawll",
               "model": "rf", "seed": semente, "sondada": bool(sondada)}
              for (a, b), (serie, sondadas) in casos.items()
              for semente, (x, sondada) in enumerate(zip(serie, sondadas))]
    linhas += [{"feature_a": a, "feature_b": b, "interaction": x, "method": "friedman_h_abs",
                "model": "rf", "seed": semente, "sondada": True}
               for (a, b), (serie, _) in casos.items() for semente, x in enumerate(serie)]
    # Outro modelo: em (a, b), o zero da semente 0 foi sondado e entra na média
    linhas += [{"feature_a": "a", "feature_b": "b", "interaction": x, "method": "gawll",
                "model": "xgb", "seed": semente, "sondada": bool(sondada)}
               for semente, (x, sondada) in enumerate(zip([0, 0, 0, 0.4, 0], [1, 0, 0, 1, 0]))]
    interacao = pd.DataFrame(linhas)

    media = graph.interacao_media(interacao)
    xgb = media[media["model"] == "xgb"].iloc[0]
    assert np.isclose(xgb["interaction"], 0.2) and xgb["n_sementes"] == 2
    media = media[media["model"] == "rf"].set_index(["method", "feature_a", "feature_b"])
    gawll = media.loc["gawll"]
    assert np.isclose(gawll.loc[("a", "b"), "interaction"], 0.3)
    assert gawll.loc[("a", "b"), "n_sementes"] == 2 and gawll.loc[("a", "b"), "rank"] == 1
    assert np.isclose(gawll.loc[("a", "c"), "interaction"], 0.9)
    assert gawll.loc[("a", "c"), "n_sementes"] == 1 and np.isnan(gawll.loc[("a", "c"), "rank"])
    assert np.isnan(gawll.loc[("b", "c"), "interaction"]) and gawll.loc[("b", "c"), "n_sementes"] == 0
    h = media.loc["friedman_h_abs"]  # no H, todo par é medido nas 5 amostras
    assert h["n_sementes"].eq(5).all() and h.loc[("b", "c"), "interaction"] == 0
    assert h["rank"].notna().all()

    # O grafo só usa pares com rank: (a, c), o mais forte, fica de fora
    arestas = graph.arestas_reduzidas(graph.interacao_media(interacao), fracao=1.0)
    arestas = arestas[(arestas["method"] == "gawll") & (arestas["model"] == "rf")]
    assert set(zip(arestas["feature_a"], arestas["feature_b"])) == {("a", "b")}

    try:
        graph.interacao_media(interacao.drop(columns="sondada"))
        raise AssertionError("a agregação aceitou uma tabela sem 'sondada'")
    except ValueError:
        pass

    # rankings.comparar ignora NaN: correlação nos itens comuns, listas em cada um
    a = pd.Series([3.0, 2.0, np.nan, 1.0], index=list("wxyz"))
    b = pd.Series([1.0, 2.0, 5.0, 3.0], index=list("wxyz"))
    resultado = interacoes.comparar(a, b)
    assert np.isclose(resultado["spearman"], -1.0)
    assert resultado["top5"] == 3 / 5  # w, x, z de um lado; os quatro do outro


def teste_scripts_nao_leem_o_teste():
    """§4.9: H e comparação rodam inteiros (em miniatura) sem abrir test.parquet.

    Usada em: bloco __main__ desta suíte (python tests/test_interacao.py; também em
    python main.py --testes).
    Importância: nenhuma interação é calculada no teste.
    """
    ler_parquet_original = pd.read_parquet

    def ler_parquet_vigiado(caminho, *args, **kwargs):
        """Substituto de pandas.read_parquet (via mock.patch) que falha se o arquivo for
        o conjunto de teste e, senão, lê normalmente.

        Usada em: teste_scripts_nao_leem_o_teste.
        Importância: é o fiscal automático do §4.9 (docs/metodologia.md).
        """
        if Path(caminho).name.startswith("test"):
            raise AssertionError(f"A Etapa 13 tentou ler o teste: {caminho}")
        return ler_parquet_original(caminho, *args, **kwargs)

    with tempfile.TemporaryDirectory() as pasta:
        pasta = Path(pasta)
        shutil.copy(DIR_TABELAS / "interacao_gawll.csv", pasta)
        with mock.patch("pandas.read_parquet", side_effect=ler_parquet_vigiado), \
                mock.patch.object(comum, "DIR_TABELAS", pasta), \
                mock.patch.object(comum, "SEMENTES", [42, 43]), \
                mock.patch.object(friedman_h, "AMOSTRA", 30), \
                mock.patch.object(interacoes, "DIR_FIGURAS", pasta / "figuras"):
            friedman_h.main()
            interacoes.main()
        for esperado in ["interacao_h.csv", "interacoes.csv", "interacoes_pares.csv",
                         "interacoes_redundancia.csv"]:
            assert (pasta / esperado).exists(), esperado
        assert (pasta / "figuras" / "interacao_gawll_x_h.png").exists()


def teste_tabelas_reais():
    """Saídas reais da Etapa 13: os 406 pares por (método, modelo, semente), H ≥ 0, H
    normalizado acima de 1 em menos de 1% dos pares não estruturais, Spearman em
    [-1, 1] e "sondada" preenchida no PyGAwLLfs.

    Usada em: bloco __main__ desta suíte (python tests/test_interacao.py; também em
    python main.py --testes).
    Importância: pega erro nas tabelas reais que os casos artificiais não cobrem.
    """
    h = pd.read_csv(DIR_TABELAS / "interacao_h.csv")
    assert set(h["method"]) == {"friedman_h_abs", "friedman_h"}
    assert set(h["model"]) == set(friedman_h.MODELOS_H)
    for (metodo, modelo, semente), grupo in h.groupby(["method", "model", "seed"]):
        assert len(grupo) == 406, (metodo, modelo, semente)
        assert (grupo["interaction"] >= 0).all() and grupo["sondada"].all()
    # O H² normalizado NÃO é limitado a 1: quando o efeito conjunto é quase nulo,
    # o denominador some. Acontece com duas dummies da mesma variável (pares
    # estruturais), cuja combinação "as duas ligadas" não existe nos dados. Por isso
    # o H_abs é o principal, e a checagem vale só nos pares não estruturais.
    normalizado = h[h["method"] == "friedman_h"]
    estrutural = [graph.tipo_por_construcao(a, b) == "estrutural"
                  for a, b in zip(normalizado["feature_a"], normalizado["feature_b"])]
    assert (normalizado.loc[~np.array(estrutural), "interaction"] > 1).mean() < 0.01

    gawll = pd.read_csv(DIR_TABELAS / "interacao_gawll.csv")
    assert gawll["sondada"].notna().all()

    comparacao = pd.read_csv(DIR_TABELAS / "interacoes.csv")
    assert comparacao["spearman"].between(-1, 1).all()

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
        teste_tipo_por_construcao,
        teste_agregacao_das_sementes,
        teste_scripts_nao_leem_o_teste,
        teste_tabelas_reais,
    ]
    for teste in testes:
        logger.info(f"{teste.__name__} ...")
        teste()
        logger.info(f"{teste.__name__} OK")
    logger.info(f"Todos os {len(testes)} testes passaram")
