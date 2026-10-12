"""Testes da Etapa 7 — PyGAwLLfs corrigido (src/importance/gawll.py).

Script simples com assert (o pytest não está instalado). Rodar da raiz:
    python tests/test_gawll.py

Verifica as duas correções de bug, o isolamento do teste (§4.1, §4.9), o
determinismo do fitness e uma validação no dataset Artificial do artigo.
"""

import logging
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Raiz do projeto: test_gawll.py -> tests/ -> raiz
RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ))
sys.path.append(str(RAIZ / "PyGAwLLfs"))

from src.core.ga_operators import Mutation  # noqa: E402  (PyGAwLLfs)
from src.core.gawll import GAwLL, Individual  # noqa: E402
from src.importance import gawll  # noqa: E402  (projeto)

logger = logging.getLogger(__name__)

DIR_DADOS = RAIZ / "data" / "processed"


def teste_mutacao():
    """Correção 4: a mutação precisa alterar o próprio cromossomo (probabilidade 1
    inverte tudo, 0 não muda nada).

    Usada em: bloco __main__ desta suíte (python tests/test_gawll.py; também em
    python main.py --testes).
    Importância: prova o bug do original e a correção no wrapper.
    """
    original = np.array([True, False, True, False, False])

    # O bug: a Mutation original não altera o array recebido
    c = original.copy()
    Mutation.bit_flip_mutation(c, 1.0)
    assert np.array_equal(c, original), "bug 1 não reproduzido"

    c = original.copy()
    gawll.MutacaoInPlace.bit_flip_mutation(c, 1.0)
    assert np.array_equal(c, ~original), "probabilidade 1 deveria inverter tudo"

    c = original.copy()
    gawll.MutacaoInPlace.bit_flip_mutation(c, 0.0)
    assert np.array_equal(c, original), "probabilidade 0 não deveria mudar nada"
    assert c.dtype == bool


def teste_elitismo():
    """Correção 5: o elite precisa entrar na população NOVA.

    Usada em: bloco __main__ desta suíte (python tests/test_gawll.py; também em
    python main.py --testes).
    Importância: prova o bug do original e a correção no wrapper.
    """

    def criar_ga(classe):
        """GA mínimo (4 bits, população 3) da classe dada, sem avaliador.

        Usada em: teste_elitismo, com o GAwLL original e com o corrigido.
        Importância: isola o elitismo do resto do motor.
        """
        return classe(
            perf_batch=None, chrom_size=4, mutation_probability=0.25,
            max_generations=1, pop_size=3, crossover_rate=0.8, tau_reset=50,
            linkage_learning=True,
        )

    def populacoes():
        """Populações antiga e nova e um elite conhecidos.

        Usada em: teste_elitismo.
        Importância: com valores fixos, dá para ver em qual população o elite foi
        parar.
        """
        antiga = [Individual(np.zeros(4, bool), 0.1, 0.1) for _ in range(3)]
        nova = [
            Individual(np.array([True, False, False, False]), 0.5, 0.5),
            Individual(np.array([False, True, False, False]), 0.2, 0.2),  # pior
            Individual(np.array([False, False, True, False]), 0.4, 0.4),
        ]
        elite = Individual(np.ones(4, bool), 0.9, 0.9)
        return antiga, nova, elite

    # O bug: no original, a população nova continua sem o elite
    ga = criar_ga(GAwLL)
    antiga, nova, elite = populacoes()
    ga.population = antiga
    ga._apply_elitism(nova, elite)
    assert max(ind.fitness for ind in nova) == 0.5, "bug 2 não reproduzido"

    ga = criar_ga(gawll.GAwLLCorrigido)
    antiga, nova, elite = populacoes()
    ga.population = antiga
    ga._apply_elitism(nova, elite)
    assert nova[1].fitness == 0.9 and nova[1].chromosome.all(), "elite não entrou"
    assert nova[1].chromosome is not elite.chromosome, "elite deveria ser uma cópia"
    assert all(ind.fitness == 0.1 for ind in antiga), "população antiga alterada"


def teste_subamostra_sem_teste():
    """A subamostra só pode ter linhas de treino e validação (nenhum índice em comum com
    test.parquet).

    Usada em: bloco __main__ desta suíte (python tests/test_gawll.py; também em
    python main.py --testes).
    Importância: garante que o GA nunca vê o teste (§4.1).
    """
    subamostra = pd.read_parquet(DIR_DADOS / "subamostra.parquet")
    indice_teste = pd.read_parquet(DIR_DADOS / "test.parquet", columns=[]).index

    assert set(subamostra["origem"]) <= {"treino", "validacao"}
    assert subamostra.index.intersection(indice_teste).empty, "linha de teste na subamostra"
    assert subamostra["Partida Prevista"].max() < pd.Timestamp("2025-11-01")


def teste_busca_nao_le_o_teste():
    """§4.1: um GA completo (mini) roda sem nunca abrir test.parquet.

    Usada em: bloco __main__ desta suíte (python tests/test_gawll.py; também em
    python main.py --testes).
    Importância: prova a correção 1 em execução, não só por leitura do código.
    """
    ler_parquet_original = pd.read_parquet

    def ler_parquet_vigiado(caminho, *args, **kwargs):
        """Substituto de pandas.read_parquet (via mock.patch) que falha se o arquivo for
        o conjunto de teste e, senão, lê normalmente.

        Usada em: teste_busca_nao_le_o_teste.
        Importância: é o fiscal automático do §4.9 (docs/metodologia.md).
        """
        if Path(caminho).name.startswith("test"):
            raise AssertionError(f"A busca tentou ler o teste: {caminho}")
        return ler_parquet_original(caminho, *args, **kwargs)

    with mock.patch("pandas.read_parquet", side_effect=ler_parquet_vigiado):
        x_treino, y_treino, x_val, y_val, features = gawll.carregar_subamostra()
        parametros = gawll.ler_configuracoes_escolhidas()["random_forest"]
        dados = (x_treino[:2000], y_treino[:2000], x_val[:1000], y_val[:1000], features)

        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "mini.npz"
            gawll.executar(
                "random_forest", parametros, 42, dados, caminho,
                geracoes=2, populacao=20,
            )
            resultado = np.load(caminho)
            n = len(features)
            assert resultado["vimp"].shape == (n,)
            assert resultado["vint"].shape == (n, n)
            assert np.allclose(resultado["vint"], resultado["vint"].T)
            assert resultado["n_treinos"] <= resultado["n_avaliacoes"]


def teste_determinismo_e_cache():
    """A mesma máscara dá o mesmo F1 nos três modelos, e o cache não retreina.

    Usada em: bloco __main__ desta suíte (python tests/test_gawll.py; também em
    python main.py --testes).
    Importância: o cache só é correto se o treino for determinístico.
    """
    x_treino, y_treino, x_val, y_val, features = gawll.carregar_subamostra()
    x_treino, y_treino = x_treino[:10_000], y_treino[:10_000]
    configuracoes = gawll.ler_configuracoes_escolhidas()

    rng = np.random.default_rng(0)
    mascara = rng.random(len(features)) < 0.5

    for nome in gawll.MODELOS:
        avaliador = gawll.AvaliadorFitness(
            nome, configuracoes[nome], x_treino, y_treino, x_val, y_val
        )
        # Sem cache: dois treinos independentes
        f1_a, f1_b = (
            gawll.treinar_e_medir(
                nome, configuracoes[nome], avaliador.peso_positivo, mascara,
                x_treino, y_treino, x_val, y_val,
            )
            for _ in range(2)
        )
        assert f1_a == f1_b, f"{nome} não é determinístico: {f1_a} != {f1_b}"

        # Com cache: a máscara repetida e a vazia não geram treino
        vazia = np.zeros(len(features), bool)
        resultado = avaliador([mascara, mascara.copy(), vazia])
        assert resultado == [f1_a, f1_a, 0.0]
        assert avaliador.n_treinos == 1 and avaliador.n_avaliacoes == 3
        logger.info(f"  {nome}: F1 {f1_a:.4f}, determinístico")


def teste_artificial():
    """Validação no Artificial do artigo: as informativas precisam aparecer. Com
    shuffle=False, as colunas 0-3 são informativas, 4-5 redundantes (combinações
    lineares das informativas) e 6-9 ruído.

    Usada em: bloco __main__ desta suíte (python tests/test_gawll.py; também em
    python main.py --testes).
    Importância: confere que o GA corrigido recupera a importância num caso
    conhecido.
    """
    x, y = make_classification(
        n_samples=5000, n_features=10, n_informative=4, n_redundant=2,
        shuffle=False, random_state=42,
    )
    # Dado sintético, sem tempo: aqui o split aleatório é legítimo
    x_treino, x_val, y_treino, y_val = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )
    features = [f"x{i}" for i in range(10)]
    dados = (x_treino, y_treino, x_val, y_val, features)

    with tempfile.TemporaryDirectory() as pasta:
        caminho = Path(pasta) / "artificial.npz"
        gawll.executar(
            "random_forest", {"max_depth": 10}, 42, dados, caminho, geracoes=20
        )
        vimp = np.load(caminho)["vimp"]

    informativas, redundantes, ruido = vimp[:4], vimp[4:6], vimp[6:]
    piso = 0.02 / len(features)
    logger.info(f"  vimp informativas {np.round(informativas, 4)}")
    logger.info(f"  vimp redundantes  {np.round(redundantes, 4)}")
    logger.info(f"  vimp ruído        {np.round(ruido, 4)} (piso de parcimônia {piso})")
    assert informativas.mean() > ruido.mean(), "informativas não superaram o ruído"


if __name__ == "__main__":
    testes = [
        teste_mutacao,
        teste_elitismo,
        teste_subamostra_sem_teste,
        teste_busca_nao_le_o_teste,
        teste_determinismo_e_cache,
        teste_artificial,
    ]
    for teste in testes:
        logger.info(f"{teste.__name__} ...")
        teste()
        logger.info(f"{teste.__name__} OK")
    logger.info(f"Todos os {len(testes)} testes passaram")
