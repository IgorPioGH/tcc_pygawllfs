"""Testes da Etapa 15 — Diferencial 4 (src/domain/validate_pairs.py).

Script simples com assert (o pytest não está instalado). Rodar da raiz,
depois de validate_pairs.py:
    python tests/test_dominio.py

Os quatro primeiros conferem as peças do teste descritivo em dados montados
à mão (onde se sabe se há interação ou não); o quinto roda o script inteiro
numa pasta temporária com o test.parquet bloqueado (§4.9); o último confere a
tabela real.
"""

import logging
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

# Raiz do projeto: test_dominio.py -> tests/ -> raiz
RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ))

from src.domain import validate_pairs as v  # noqa: E402
from src.importance import comum  # noqa: E402

logger = logging.getLogger(__name__)

DIR_TABELAS = RAIZ / "results" / "tables"


def tabela_artificial(com_interacao: bool, n_por_celula: int = 20_000, semente: int = 0):
    """Tabela 4 x 6 com logit(p) = efeito(A) + efeito(B) [+ interação, se pedida]."""
    rng = np.random.default_rng(semente)
    efeito_a = np.array([-1.5, -1.0, -0.5, 0.0])
    efeito_b = np.linspace(-0.5, 0.5, 6)
    linhas = []
    for i, ea in enumerate(efeito_a):
        for j, eb in enumerate(efeito_b):
            logit = ea + eb + (0.8 if com_interacao and i == 3 and j >= 3 else 0.0)
            p = 1 / (1 + np.exp(-logit))
            linhas.append({"nivel_a": i, "nivel_b": j, "n": n_por_celula,
                           "k": rng.binomial(n_por_celula, p)})
    return pd.DataFrame(linhas)


def teste_coeficiente_de_incerteza():
    """1 quando uma variável determina a outra; ~0 quando são independentes."""
    rng = np.random.default_rng(0)
    mes = pd.Series(rng.integers(1, 13, 50_000))
    trimestre = (mes - 1) // 3 + 1
    assert np.isclose(v.coeficiente_de_incerteza(mes, trimestre), 1.0)
    independente = pd.Series(rng.integers(0, 5, 50_000))
    assert v.coeficiente_de_incerteza(mes, independente) < 0.01


def teste_ajuste_sem_interacao():
    """Sem interação, o modelo de efeitos isolados reproduz a tabela; com, não."""
    rng = np.random.default_rng(1)
    for com_interacao, esperado in [(False, "dentro do ruído"), (True, "acima do ruído")]:
        tabela = tabela_artificial(com_interacao)
        desvio, ruido, ajustada = v.testar_tabela(tabela, rng, n_bootstrap=50)
        if com_interacao:
            assert desvio > 3 * ruido and desvio > v.LIMIAR_PP, (desvio, ruido)
        else:
            assert desvio < 1.5 * ruido, (desvio, ruido)
        assert ajustada["previsto_sem_interacao"].between(0, 1).all(), esperado


def teste_classificacao():
    passa = {"desvio_2024_pp": 2.0, "ruido_p95_2024_pp": 0.5,
             "desvio_2025_pp": 1.5, "ruido_p95_2025_pp": 0.5, "replicacao_anos": 0.7}
    assert v.classificar("operacional", passa) == "corroborado descritivamente"
    assert v.classificar("operacional", {**passa, "replicacao_anos": 0.2}) == "não corroborado"
    assert v.classificar("operacional", {**passa, "desvio_2025_pp": 0.8}) == "não corroborado"
    assert v.classificar("operacional", {**passa, "ruido_p95_2024_pp": 2.5}) == "não corroborado"
    assert v.classificar("redundante", passa).startswith("redundância")
    assert v.classificar("estrutural", {}).startswith("não testável")


def teste_selecao_e_tipos():
    """Controles não repetem o par de domínio de nenhum candidato."""
    media = v.interacao_media()
    pares = v.selecionar_pares(media)
    candidatos = pares[pares["grupo"] == "candidato"]
    controles = pares[pares["grupo"] == "controle"]
    assert len(controles) == v.N_CONTROLES

    def dominio(linha):
        return frozenset((v.variavel(linha.feature_a), v.variavel(linha.feature_b)))

    chaves_candidatos = {dominio(linha) for linha in candidatos.itertuples()}
    chaves_controles = [dominio(linha) for linha in controles.itertuples()]
    assert not (set(chaves_controles) & chaves_candidatos)
    assert len(set(chaves_controles)) == len(chaves_controles)

    for modelo in v.MODELOS:  # todo par do top-10 de cada modelo é candidato
        top = media[(media["model"] == modelo) & (media["rank"] <= v.TOP_PARES_POR_MODELO)]
        chaves = set(zip(candidatos["feature_a"], candidatos["feature_b"]))
        assert set(zip(top["feature_a"], top["feature_b"])) <= chaves, modelo


def teste_script_nao_le_o_teste():
    """§4.9: o script inteiro roda (bootstrap reduzido) sem abrir test.parquet."""
    ler_parquet_original = pd.read_parquet

    def ler_parquet_vigiado(caminho, *args, **kwargs):
        if Path(caminho).name.startswith("test"):
            raise AssertionError(f"A validação de domínio tentou ler o teste: {caminho}")
        return ler_parquet_original(caminho, *args, **kwargs)

    with tempfile.TemporaryDirectory() as pasta:
        pasta = Path(pasta)
        for nome in ["interacao_gawll.csv", "ranking_consolidado.csv"]:
            shutil.copy(DIR_TABELAS / nome, pasta)
        with mock.patch("pandas.read_parquet", side_effect=ler_parquet_vigiado), \
                mock.patch.object(comum, "DIR_TABELAS", pasta), \
                mock.patch.object(v, "DIR_FIGURAS", pasta / "figuras"), \
                mock.patch.object(v, "N_BOOTSTRAP", 5):
            v.main()
        assert (pasta / "dominio_pares.csv").exists()
        assert (pasta / "figuras" / "grafo_interacao.png").exists()


def teste_tabela_real():
    resultado = pd.read_csv(DIR_TABELAS / "dominio_pares.csv")
    assert resultado["grupo"].value_counts()["controle"] == v.N_CONTROLES
    validas = {"corroborado descritivamente", "não corroborado",
               "redundância (não é interação operacional)", "não testável (mesma variável)"}
    assert set(resultado["classificacao"]) <= validas

    # Todo corroborado cumpre os três critérios, nos dois anos
    for linha in resultado[resultado["classificacao"] == "corroborado descritivamente"].itertuples():
        for ano in v.ANOS:
            desvio = getattr(linha, f"desvio_{ano}_pp")
            assert desvio > getattr(linha, f"ruido_p95_{ano}_pp") and desvio >= v.LIMIAR_PP
        assert linha.replicacao_anos >= v.LIMIAR_REPLICACAO
        assert linha.tipo == "operacional"


if __name__ == "__main__":
    testes = [
        teste_coeficiente_de_incerteza,
        teste_ajuste_sem_interacao,
        teste_classificacao,
        teste_selecao_e_tipos,
        teste_script_nao_le_o_teste,
        teste_tabela_real,
    ]
    for teste in testes:
        logger.info(f"{teste.__name__} ...")
        teste()
        logger.info(f"{teste.__name__} OK")
    logger.info(f"Todos os {len(testes)} testes passaram")
