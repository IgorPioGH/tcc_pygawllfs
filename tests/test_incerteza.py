"""Testes da Etapa 12b — incerteza por bootstrap de dias (src/compare/incerteza.py).

Script simples com assert (o pytest não está instalado). Rodar da raiz,
depois de incerteza.py:
    python tests/test_incerteza.py

Conferem que a métrica com peso por dia é a mesma da reamostragem
materializada, o sorteio de dias, o pareamento num caso com resposta conhecida
e a coerência das tabelas reais com as Etapas 6 e 12.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

# Raiz do projeto: test_incerteza.py -> tests/ -> raiz
RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ))

from src.compare import incerteza  # noqa: E402

logger = logging.getLogger(__name__)

DIR_TABELAS = RAIZ / "results" / "tables"


def dados_de_brinquedo(n_voos=3000, n_dias=30, semente=0):
    """Alvo, probabilidade, classe e dia de voos inventados, com taxa de atraso
    diferente por dia (como no teste real).

    Usada em: teste_peso_igual_a_reamostragem e teste_pareamento.
    Importância: casos pequenos, com resposta conferível à mão.
    """
    rng = np.random.default_rng(semente)
    codigos = rng.integers(0, n_dias, size=n_voos)
    taxa_dia = rng.uniform(0.1, 0.6, size=n_dias)
    y = (rng.random(n_voos) < taxa_dia[codigos]).astype(int)
    score = np.clip(0.3 * y + rng.random(n_voos) * 0.7, 0, 1)
    return y, score, (score > 0.5).astype(int), codigos


def teste_peso_igual_a_reamostragem():
    """A métrica com peso (quantas vezes o dia foi sorteado) é igual à métrica na
    reamostragem materializada, com os voos dos dias repetidos copiados.

    Usada em: bloco __main__ desta suíte (python tests/test_incerteza.py; também em
    python main.py --testes).
    Importância: valida o atalho que deixa o bootstrap rápido.
    """
    y, score, classe, codigos = dados_de_brinquedo()
    contagens = incerteza.sortear_dias(30, 5, semente=1)
    funcoes = {"f1": (f1_score, classe), "pr_auc": (average_precision_score, score),
               "roc_auc": (roc_auc_score, score)}
    for contagem in contagens:
        indices = np.concatenate([np.repeat(np.flatnonzero(codigos == dia), vezes)
                                  for dia, vezes in enumerate(contagem)])
        for metrica, (funcao, previsao) in funcoes.items():
            materializada = funcao(y[indices], previsao[indices])
            ponderada = incerteza.metrica_ponderada(metrica, y, previsao, contagem[codigos])
            assert np.isclose(materializada, ponderada, rtol=0, atol=1e-12), \
                (metrica, materializada, ponderada)


def teste_sorteio_de_dias():
    """Cada repetição sorteia exatamente n_dias dias, e a mesma semente dá o mesmo
    sorteio.

    Usada em: bloco __main__ desta suíte.
    Importância: o pareamento e a reprodutibilidade dependem disso.
    """
    contagens = incerteza.sortear_dias(61, 200, semente=42)
    assert contagens.shape == (200, 61)
    assert (contagens.sum(axis=1) == 61).all()
    assert (contagens == incerteza.sortear_dias(61, 200, semente=42)).all()
    # Em média ~63% dos dias entram em cada reamostragem (1 - 1/e)
    assert 0.55 < (contagens > 0).mean() < 0.70


def teste_pareamento():
    """Duas configurações idênticas: a diferença é zero em toda repetição. Uma
    configuração perfeita contra uma ruim: o IC da diferença não contém zero.

    Usada em: bloco __main__ desta suíte.
    Importância: confere que o mesmo sorteio é aplicado a todas as configurações.
    """
    y, score, _, codigos = dados_de_brinquedo()
    contagens = incerteza.sortear_dias(30, 50, semente=3)
    previsoes = np.stack([score, score, y.astype(float)])
    repeticoes = dict(zip(["a", "a_copia", "perfeito"],
                          incerteza.bootstrap("pr_auc", y, previsoes, codigos, contagens)))
    pontual = {nome: valores.mean() for nome, valores in repeticoes.items()}
    iguais, perfeito = incerteza.comparar_pares([("a", "a_copia"), ("perfeito", "a")],
                                                pontual, repeticoes)
    assert iguais["ic_inf"] == iguais["ic_sup"] == 0 and iguais["ic_contem_zero"]
    assert perfeito["ic_inf"] > 0 and not perfeito["ic_contem_zero"]


def teste_tabelas_reais():
    """Tabelas reais: estimativas iguais a baselines.csv e fidelidade_resumo.csv, IC
    ordenado, "todas" com retenção 1 sem incerteza e diferenças de pares com "a" à
    frente.

    Usada em: bloco __main__ desta suíte.
    Importância: confere a coerência da Etapa 12b com as Etapas 6 e 12.
    """
    modelos = pd.read_csv(DIR_TABELAS / "incerteza_modelos.csv")
    baselines = pd.read_csv(DIR_TABELAS / "baselines.csv").set_index("modelo")
    for linha in modelos.itertuples():
        assert abs(linha.valor - baselines.loc[linha.modelo, linha.metrica]) < 1e-4
        assert linha.ic_inf <= linha.ic_sup
    assert len(pd.read_csv(DIR_TABELAS / "incerteza_modelos_pares.csv")) == \
        3 * len(incerteza.PARES_MODELOS)

    fidelidade = pd.read_csv(DIR_TABELAS / "incerteza_fidelidade.csv")
    resumo = pd.read_csv(DIR_TABELAS / "fidelidade_resumo.csv")
    resumo = resumo[resumo["conjunto"] == "test"].set_index(["model", "method", "k"])
    for linha in fidelidade.itertuples():
        assert abs(linha.pr_auc - resumo.loc[(linha.model, linha.method, linha.k), "pr_auc"]) < 1e-4
        assert linha.frac_ic_inf <= linha.frac_ic_sup
    todas = fidelidade[fidelidade["method"] == "todas"]
    assert np.allclose(todas[["pr_auc_frac_todas", "frac_ic_inf", "frac_ic_sup"]], 1.0)

    pares = pd.read_csv(DIR_TABELAS / "incerteza_fidelidade_pares.csv")
    assert (pares["diferenca"] >= 0).all()
    assert (pares["ic_contem_zero"] == ((pares["ic_inf"] <= 0) & (pares["ic_sup"] >= 0))).all()


if __name__ == "__main__":
    testes = [
        teste_peso_igual_a_reamostragem,
        teste_sorteio_de_dias,
        teste_pareamento,
        teste_tabelas_reais,
    ]
    for teste in testes:
        logger.info(f"{teste.__name__} ...")
        teste()
        logger.info(f"{teste.__name__} OK")
    logger.info(f"Todos os {len(testes)} testes passaram")
