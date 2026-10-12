"""Testes da Etapa 12 — fidelidade por retenção (src/compare/fidelity.py).

Script simples com assert (o pytest não está instalado). Rodar da raiz,
depois de fidelity.py:
    python tests/test_fidelidade.py

Conferem as listas de features de cada configuração, o resumo, uma execução
em miniatura e a coerência da tabela real com a Etapa 6.
"""

import logging
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

# Raiz do projeto: test_fidelidade.py -> tests/ -> raiz
RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ))

from src.compare import fidelity  # noqa: E402
from src.compare.rankings import ordenar  # noqa: E402
from src.importance import comum  # noqa: E402
from src.models.baselines import carregar  # noqa: E402

logger = logging.getLogger(__name__)

DIR_TABELAS = RAIZ / "results" / "tables"


def configuracoes_reais(modelo: str) -> list[dict]:
    """Configurações de fidelidade de um modelo, montadas com as tabelas reais.

    Usada em: teste_listas_de_features.
    Importância: testa listas_de_features com os dados de verdade.
    """
    features = list(carregar("val")[0].columns)
    ranking = pd.read_csv(DIR_TABELAS / "ranking_consolidado.csv")
    selecao = pd.read_csv(DIR_TABELAS / "selecao_gawll.csv")
    return fidelity.listas_de_features(modelo, features, ranking, selecao)


def teste_listas_de_features():
    """As configurações batem com o ranking consolidado: "todas" tem as 29 features e o
    top-k de cada método são as k primeiras do ranking.

    Usada em: bloco __main__ desta suíte (python tests/test_fidelidade.py; também em
    python main.py --testes).
    Importância: um top-k errado invalidaria a fidelidade inteira.
    """
    ranking = pd.read_csv(DIR_TABELAS / "ranking_consolidado.csv")
    selecao = pd.read_csv(DIR_TABELAS / "selecao_gawll.csv")
    for modelo in fidelity.MODELOS:
        configs = configuracoes_reais(modelo)
        por_metodo = pd.DataFrame(configs)

        todas = por_metodo[por_metodo["method"] == "todas"].iloc[0]
        assert todas["k"] == 29 and len(todas["features"]) == 29

        # top-k = as k primeiras do ranking consolidado
        for metodo, grupo in ranking[ranking["model"] == modelo].groupby("method"):
            ordem = ordenar(grupo.set_index("feature")["pontuacao"])
            for k in fidelity.TOP_K:
                config = por_metodo[(por_metodo["method"] == metodo) & (por_metodo["k"] == k)]
                assert config.iloc[0]["features"] == ordem[:k], (modelo, metodo, k)

        # x* = exatamente as seleções do PyGAwLLfs
        x_estrela = por_metodo[por_metodo["method"] == "gawll_x*"]
        esperado = selecao[selecao["model"] == modelo]["features"].str.split(";").tolist()
        assert x_estrela["features"].tolist() == esperado

        # sorteios: tamanho k, sem repetição, reprodutíveis e diferentes entre si
        sorteios = por_metodo[por_metodo["method"] == "aleatorio"]
        assert len(sorteios) == len(fidelity.TOP_K) * fidelity.N_SORTEIOS
        for config in sorteios.itertuples():
            assert len(set(config.features)) == len(config.features) == config.k
        assert sorteios["features"].tolist() == [
            c["features"] for c in configuracoes_reais(modelo) if c["method"] == "aleatorio"]
        assert sorteios["features"].map(tuple).nunique() > 1


def teste_resumo():
    """resumir calcula médias, extremos e a fração de "todas" corretamente, num caso
    pequeno feito à mão.

    Usada em: bloco __main__ desta suíte (python tests/test_fidelidade.py; também em
    python main.py --testes).
    Importância: a fração de "todas" é o número que vai para o texto.
    """
    linhas = []
    for metodo, k, f1 in [("todas", 29, 0.40), ("shap", 5, 0.30), ("aleatorio", 5, 0.10),
                          ("aleatorio", 5, 0.20)]:
        linhas.append({"model": "m", "method": metodo, "k": k, "seed": len(linhas),
                       "conjunto": "test", "f1": f1, "pr_auc": f1, "roc_auc": 0.5,
                       "acuracia_balanceada": 0.5})
    resumo = fidelity.resumir(pd.DataFrame(linhas)).set_index("method")
    assert np.isclose(resumo.loc["todas", "f1_frac_todas"], 1.0)
    assert np.isclose(resumo.loc["shap", "f1_frac_todas"], 0.75)
    assert np.isclose(resumo.loc["aleatorio", "f1"], 0.15)
    assert resumo.loc["aleatorio", "f1_min"] == 0.10 and resumo.loc["aleatorio", "f1_max"] == 0.20


def teste_execucao_em_miniatura():
    """O script roda inteiro (LogReg, 3 mil linhas por conjunto, 2 sorteios).

    Usada em: bloco __main__ desta suíte (python tests/test_fidelidade.py; também em
    python main.py --testes).
    Importância: pega erro de integração sem esperar os ~50 min da execução real.
    """
    fatias = {nome: tuple(parte.iloc[:3000] for parte in carregar(nome))
              for nome in ["train", "val", "test"]}
    with tempfile.TemporaryDirectory() as pasta:
        pasta = Path(pasta)
        for nome in ["ranking_consolidado.csv", "selecao_gawll.csv"]:
            shutil.copy(DIR_TABELAS / nome, pasta)
        with mock.patch.object(comum, "DIR_TABELAS", pasta), \
                mock.patch.object(fidelity, "carregar", lambda nome: fatias[nome]), \
                mock.patch.object(fidelity, "MODELOS", ["logreg"]), \
                mock.patch.object(fidelity, "N_SORTEIOS", 2), \
                mock.patch.object(fidelity, "DIR_FIGURAS", pasta / "figuras"):
            fidelity.main()
        tabela = pd.read_csv(pasta / "fidelidade.csv")
        assert set(tabela["conjunto"]) == {"val", "test"}
        assert (pasta / "figuras" / "fidelidade.png").exists()


def teste_tabela_real():
    """fidelidade.csv real: métricas em [0, 1], contagem certa de configurações por
    modelo e "todas" no teste igual a baselines.csv.

    Usada em: bloco __main__ desta suíte (python tests/test_fidelidade.py; também em
    python main.py --testes).
    Importância: confere a coerência entre a Etapa 12 e a Etapa 6.
    """
    tabela = pd.read_csv(DIR_TABELAS / "fidelidade.csv")
    for coluna in fidelity.METRICAS:
        assert tabela[coluna].between(0, 1).all(), coluna

    for (modelo, conjunto), grupo in tabela.groupby(["model", "conjunto"]):
        contagem = grupo["method"].value_counts()
        assert contagem["todas"] == 1 and contagem["gawll_x*"] == 5
        assert contagem["aleatorio"] == len(fidelity.TOP_K) * fidelity.N_SORTEIOS
        n_metodos = 3 if modelo == "logreg" else 4
        assert sum(contagem.get(m, 0) for m in fidelity.ESTILO) == n_metodos * len(fidelity.TOP_K)

    # Coerência com a Etapa 6: todas as features no teste = baselines.csv
    baselines = pd.read_csv(DIR_TABELAS / "baselines.csv").set_index("modelo")
    todas = tabela[(tabela["method"] == "todas") & (tabela["conjunto"] == "test")]
    for linha in todas.itertuples():
        for metrica in ["f1", "pr_auc"]:
            assert abs(getattr(linha, metrica) - baselines.loc[linha.model, metrica]) < 1e-3, \
                (linha.model, metrica)


if __name__ == "__main__":
    testes = [
        teste_listas_de_features,
        teste_resumo,
        teste_execucao_em_miniatura,
        teste_tabela_real,
    ]
    for teste in testes:
        logger.info(f"{teste.__name__} ...")
        teste()
        logger.info(f"{teste.__name__} OK")
    logger.info(f"Todos os {len(testes)} testes passaram")
