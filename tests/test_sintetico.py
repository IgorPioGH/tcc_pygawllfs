"""Testes da Etapa 10 — Diferencial 1 (src/synthetic/).

Script simples com assert (o pytest não está instalado). Rodar da raiz:
    python tests/test_sintetico.py

Os três primeiros não dependem de nada rodado antes: conferem que o gerador
produz a estrutura prometida no gabarito e que as métricas de recuperação dão
o valor certo em casos montados à mão. O último confere as tabelas geradas
por aplicar_metodos.py (rodar antes).
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Raiz do projeto: test_sintetico.py -> tests/ -> raiz
RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ))

from src.importance.comum import COLUNAS_PADRAO, tabela_importancia  # noqa: E402
from src.importance.consolidar_gawll import tabela_interacao  # noqa: E402
from src.synthetic import generate, recovery  # noqa: E402

logger = logging.getLogger(__name__)

DIR_TABELAS = RAIZ / "results" / "tables"


def teste_gerador():
    """O dataset tem a estrutura que o gabarito afirma."""
    a = generate.gerar(n=20_000)
    b = generate.gerar(n=20_000)
    assert a.equals(b), "mesma semente deveria gerar o mesmo dataset"
    assert set(generate.GABARITO) == set(generate.FEATURES)

    # Interação pura: x1 e x2 sozinhas não carregam informação sobre y
    for nome in ["x1", "x2"]:
        taxa_positivo = a.loc[a[nome] > 0, "y"].mean()
        taxa_negativo = a.loc[a[nome] <= 0, "y"].mean()
        assert abs(taxa_positivo - taxa_negativo) < 0.03, f"{nome} tem efeito marginal"
    # Juntas, carregam: o sinal do produto muda a taxa de y
    produto_positivo = a.loc[a["x1"] * a["x2"] > 0, "y"].mean()
    produto_negativo = a.loc[a["x1"] * a["x2"] <= 0, "y"].mean()
    assert produto_positivo - produto_negativo > 0.4, "a interação deveria ser forte"

    assert abs(a["x3"].corr(a["x5"]) - generate.RHO_REDUNDANCIA) < 0.02
    assert abs(a["x4"].corr(a["x6"]) - generate.RHO_REDUNDANCIA) < 0.02
    for nome in ["x7", "x8", "x9", "x10"]:
        assert abs(a[nome].corr(a["y"])) < 0.03, f"{nome} deveria ser ruído"
    assert 0.45 < a["y"].mean() < 0.55

    dividido = generate.dividir(generate.gerar())
    contagem = dividido["conjunto"].value_counts()
    assert contagem["treino"] == 3500 and contagem["validacao"] == 1500
    taxas = dividido.groupby("conjunto")["y"].mean()
    assert abs(taxas["treino"] - taxas["validacao"]) < 0.01, "split deveria ser estratificado"


def teste_metricas_de_importancia():
    """Ranking perfeito, ranking invertido e ranking que troca diretas por proxies."""
    ordem_perfeita = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"]
    valores = list(range(10, 0, -1))  # o primeiro da lista é o mais importante

    perfeito = recovery.metricas_importancia(
        tabela_importancia(ordem_perfeita, valores, "m", "modelo", 42))
    assert perfeito["auc_relevante_ruido"] == 1.0
    assert perfeito["precisao_top6"] == 1.0 and perfeito["precisao_top4_diretas"] == 1.0
    assert perfeito["rank_x3x4"] < perfeito["rank_x5x6"]

    invertido = recovery.metricas_importancia(
        tabela_importancia(ordem_perfeita[::-1], valores, "m", "modelo", 42))
    # Invertido, o top-6 é x10..x5: as 4 de ruído e as 2 redundantes
    assert invertido["auc_relevante_ruido"] == 0.0
    assert invertido["precisao_top6"] == 2 / 6 and invertido["precisao_top4_diretas"] == 0.0

    # Proxies à frente das diretas: relevantes continuam no topo, diretas não
    proxies_primeiro = ["x5", "x6", "x1", "x2", "x3", "x4", "x7", "x8", "x9", "x10"]
    trocado = recovery.metricas_importancia(
        tabela_importancia(proxies_primeiro, valores, "m", "modelo", 42))
    assert trocado["auc_relevante_ruido"] == 1.0 and trocado["precisao_top6"] == 1.0
    assert trocado["precisao_top4_diretas"] == 0.5
    assert trocado["rank_x5x6"] < trocado["rank_x3x4"]

    # Coeficiente com sinal: o rank pelo absoluto tem de valer aqui também
    sinal = recovery.metricas_importancia(tabela_importancia(
        ordem_perfeita, [-v for v in valores], "m", "modelo", 42, rank_por_absoluto=True))
    assert sinal["auc_relevante_ruido"] == 1.0


def teste_metricas_de_interacao_e_selecao():
    """(x1, x2) no topo -> rank 1; seleção igual às diretas -> F1 = 1."""
    features = generate.FEATURES
    matriz = np.zeros((10, 10))
    for (a, b), valor in [(("x1", "x2"), 0.9), (("x3", "x5"), 0.5), (("x4", "x6"), 0.4),
                          (("x7", "x8"), 0.1)]:
        i, j = features.index(a), features.index(b)
        matriz[i, j] = matriz[j, i] = valor
    interacao = pd.concat([tabela_interacao(features, matriz, "m", "modelo", s) for s in [42, 43]])

    resultado = recovery.metricas_interacao(interacao)
    assert resultado["rank_par_interacao"] == 1
    assert resultado["rank_pares_redundancia"] == 2.5
    assert resultado["precisao_top3"] == 1.0 and resultado["acerto_por_semente"] == 1.0
    assert resultado["n_pares"] == 45

    assert recovery.metricas_selecao("x1;x2;x3;x4")["f1"] == 1.0
    parcial = recovery.metricas_selecao("x1;x3;x5;x7")
    assert parcial["precisao"] == 0.5 and parcial["recall"] == 0.5
    assert parcial["n_redundantes"] == 1 and parcial["n_ruido"] == 1
    assert recovery.metricas_selecao("")["f1"] == 0.0


def teste_tabelas_geradas():
    """As saídas de aplicar_metodos.py estão completas e no formato padrão."""
    importancia = pd.read_csv(DIR_TABELAS / "sintetico_importancia.csv")
    assert list(importancia.columns) == COLUNAS_PADRAO
    assert not importancia.isna().any().any()
    for chave, grupo in importancia.groupby(["method", "model", "seed"]):
        assert sorted(grupo["feature"]) == sorted(generate.FEATURES), chave

    esperado = {  # (método, modelo) -> nº de sementes
        **{("intrinseca", m): 1 for m in ["logreg", "random_forest", "xgboost"]},
        **{("permutation", m): 5 for m in ["logreg", "random_forest", "xgboost"]},
        **{("shap", m): 1 for m in ["random_forest", "xgboost"]},
        **{("gawll", m): 5 for m in ["logreg", "random_forest", "xgboost"]},
    }
    obtido = importancia.groupby(["method", "model"])["seed"].nunique().to_dict()
    assert obtido == esperado, obtido

    interacao = pd.read_csv(DIR_TABELAS / "sintetico_interacao.csv")
    assert (interacao.groupby(["method", "model", "seed"]).size() == 45).all()


if __name__ == "__main__":
    testes = [
        teste_gerador,
        teste_metricas_de_importancia,
        teste_metricas_de_interacao_e_selecao,
        teste_tabelas_geradas,
    ]
    for teste in testes:
        logger.info(f"{teste.__name__} ...")
        teste()
        logger.info(f"{teste.__name__} OK")
    logger.info(f"Todos os {len(testes)} testes passaram")
