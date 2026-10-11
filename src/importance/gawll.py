"""Etapa 7 — PyGAwLLfs corrigido.

Recebe a subamostra estratificada (data/processed/subamostra.parquet, só
treino e validação) e os hiperparâmetros escolhidos na Etapa 6
(results/tables/baselines.csv). Para cada (modelo, semente), roda o GA com
linkage learning e salva um checkpoint .npz com o melhor subconjunto, o vetor
de importância (vimp), a matriz de interação (vint), o tempo e o número de
treinos. Quem consome: Etapa 8 (protocolo completo), comparações a partir da
Etapa 9 e grafo de interação.

Correções em relação ao código original (documentar no Cap. 3):
    1. fitness calculado na VALIDAÇÃO, não no teste (§4.1);
    2. F1 no lugar da acurácia (§4.2);
    3. split temporal da Etapa 4, não train_test_split embaralhado (§4.4);
    4. mutação bit-flip que de fato altera o cromossomo (MutacaoInPlace);
    5. elitismo gravado na população nova (GAwLLCorrigido._apply_elitism).

Nada dentro de PyGAwLLfs/ é editado: o motor do GA é usado por import e
subclasse. Os dois src/ (projeto e PyGAwLLfs) são namespace packages, sem
__init__.py, e se fundem no sys.path. Nunca criar __init__.py em nenhum deles.

Uso (da raiz do repositório):
    python src/importance/gawll.py --modelos random_forest --sementes 42 43
"""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score

# Raiz do projeto: gawll.py -> importance/ -> src/ -> raiz
RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))
sys.path.append(str(RAIZ / "PyGAwLLfs"))

from src.core.evig import eVIG  # noqa: E402  (PyGAwLLfs)
from src.core.ga_operators import Crossover, Mutation, Selection  # noqa: E402
from src.core.gawll import GAwLL, Individual  # noqa: E402
from src.core.importance import Importance  # noqa: E402
from src.core.linkage_learning_mutation import LinkageLearning  # noqa: E402
from src.utils.statistic import PopulationAnalyzer  # noqa: E402
from src.data.clean import assert_sem_vazamento  # noqa: E402  (projeto)
from src.models.baselines import (  # noqa: E402
    NAO_ATRIBUTOS,
    criar_modelo,
    ler_configuracoes_escolhidas,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARQUIVO_SUBAMOSTRA = RAIZ / "data" / "processed" / "subamostra.parquet"
DIR_SAIDA = RAIZ / "experiments" / "runs" / "gawll"

MODELOS = ["logreg", "random_forest", "xgboost"]
SEMENTES = [42, 43, 44, 45, 46]

# Parâmetros do artigo (Guimarães et al., 2026); pm = 1/N é calculado em executar()
GERACOES = 100
POPULACAO = 50
TAXA_CRUZAMENTO = 0.8
TAU_RESET = 50

# Mesma fração de núcleos do pipeline original (hardware_config.SAFE_CORES)
N_PROCESSOS = max(1, int((os.cpu_count() or 1) * 0.8))


def carregar_subamostra() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Lê a subamostra e devolve (x_treino, y_treino, x_val, y_val, features).

    Nunca lê test.parquet: a subamostra só tem linhas de treino e validação.
    """
    df = pd.read_parquet(ARQUIVO_SUBAMOSTRA)

    origens = set(df["origem"].unique())
    if not origens <= {"treino", "validacao"}:
        raise ValueError(f"Subamostra com origem inesperada: {origens}")

    x = df.drop(columns=NAO_ATRIBUTOS + ["origem"])
    assert_sem_vazamento(x)

    eh_treino = (df["origem"] == "treino").to_numpy()
    x_np = x.to_numpy(dtype=float)
    y_np = df["atrasado"].to_numpy(dtype=int)
    return (
        x_np[eh_treino],
        y_np[eh_treino],
        x_np[~eh_treino],
        y_np[~eh_treino],
        x.columns.tolist(),
    )


def treinar_e_medir(
    nome_modelo: str,
    parametros: dict,
    peso_positivo: float,
    mascara: np.ndarray,
    x_treino: np.ndarray,
    y_treino: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Treina o modelo só com as colunas da máscara e devolve o F1 na validação.

    Fica fora da classe para o joblib conseguir mandá-la aos outros processos.
    """
    modelo = criar_modelo(nome_modelo, parametros, peso_positivo, n_jobs=1)
    modelo.fit(x_treino[:, mascara], y_treino)
    y_previsto = modelo.predict(x_val[:, mascara])
    return float(f1_score(y_val, y_previsto, zero_division=0))


class AvaliadorFitness:
    """O perf_batch que o GAwLL recebe: lista de máscaras -> lista de F1.

    Guarda um cache por máscara. Os modelos têm random_state fixo, então a
    mesma máscara sempre dá o mesmo F1, e o cache é exato. Ele importa porque
    o linkage learning chama o perf_batch direto, por fora do cache do GAwLL,
    e o GAwLL reavalia depois os mesmos cromossomos.
    """

    def __init__(self, nome_modelo, parametros, x_treino, y_treino, x_val, y_val):
        self.nome_modelo = nome_modelo
        self.parametros = parametros
        self.x_treino = x_treino
        self.y_treino = y_treino
        self.x_val = x_val
        self.y_val = y_val
        self.peso_positivo = (y_treino == 0).sum() / (y_treino == 1).sum()

        self.cache: dict[bytes, float] = {}
        self.n_treinos = 0  # treinos de modelo realmente feitos (a "moeda" do §7)
        self.n_avaliacoes = 0  # pedidos recebidos do GA, contando acertos de cache

    def __call__(self, mascaras: list[np.ndarray]) -> list[float]:
        self.n_avaliacoes += len(mascaras)
        mascaras = [np.asarray(m, dtype=bool) for m in mascaras]

        # Máscaras ainda não vistas, sem repetir dentro do mesmo lote
        novas: dict[bytes, np.ndarray] = {}
        for mascara in mascaras:
            chave = mascara.tobytes()
            if chave not in self.cache and chave not in novas:
                novas[chave] = mascara

        # Nenhuma feature selecionada: não há modelo para treinar
        for chave, mascara in list(novas.items()):
            if not mascara.any():
                self.cache[chave] = 0.0
                del novas[chave]

        if novas:
            for chave, f1 in zip(novas, self._treinar_lote(list(novas.values()))):
                self.cache[chave] = f1
            self.n_treinos += len(novas)

        return [self.cache[m.tobytes()] for m in mascaras]

    def _treinar_lote(self, mascaras: list[np.ndarray]) -> list[float]:
        argumentos = (self.x_treino, self.y_treino, self.x_val, self.y_val)

        # XGBoost: um por vez na GPU (vários processos não cabem nos 6 GB)
        if self.nome_modelo == "xgboost":
            return [
                treinar_e_medir(
                    self.nome_modelo, self.parametros, self.peso_positivo, m, *argumentos
                )
                for m in mascaras
            ]

        # LogReg e RF: um modelo por processo, cada um com n_jobs=1
        return Parallel(n_jobs=min(N_PROCESSOS, len(mascaras)))(
            delayed(treinar_e_medir)(
                self.nome_modelo, self.parametros, self.peso_positivo, m, *argumentos
            )
            for m in mascaras
        )


class MutacaoInPlace(Mutation):
    """Correção 4: a mutação original devolve um array NOVO, mas o GAwLL ignora
    o retorno, então nenhum bit era invertido (nem no reset por estagnação).
    Aqui o XOR é feito no próprio array, e o gerador aleatório é consumido do
    mesmo jeito que no original.
    """

    @staticmethod
    def bit_flip_mutation(chromosome: np.ndarray, mutation_probability: float) -> np.ndarray:
        mascara = np.random.random(len(chromosome)) < mutation_probability
        chromosome ^= mascara
        return chromosome


class GAwLLCorrigido(GAwLL):
    """GAwLL com as correções 4 e 5. O resto do motor é o original."""

    def _apply_elitism(self, new_population, fittest_individual):
        """Correção 5: o original gravava o elite em self.population (a
        população antiga, descartada logo depois), e o melhor se perdia.
        """
        pior = min(
            range(len(new_population)),
            key=lambda i: (
                new_population[i].fitness
                if new_population[i].fitness is not None
                else -1.0
            ),
        )
        new_population[pior] = Individual(
            fittest_individual.chromosome.copy(),
            fittest_individual.fitness,
            fittest_individual.f_perf,
        )

    def run(self, seed: int) -> None:
        """Cópia do GAwLL.run original, trocando Mutation por MutacaoInPlace
        e print por logger (com uma linha de progresso por geração).
        """
        random.seed(seed)
        np.random.seed(seed)

        self._initialize_population()
        self.evig = eVIG(self.chrom_size)
        self.importance = Importance(self.chrom_size)
        self.population_analyzer = PopulationAnalyzer()
        self.selection = Selection()
        self.crossover = Crossover()
        self.mutation = MutacaoInPlace()
        self.mutation_ll = LinkageLearning(
            self.evig, self.importance, self.perf_batch, self._calculate_fitness_value
        )

        last_improvement_gen = 0
        best_fitness_track = 0.0

        for generation in range(self.max_generations):
            fittest = self.population_analyzer.get_fittest_individual(self.population)
            logger.info(
                f"  geração {generation:3d} | fitness {fittest.fitness:.4f} "
                f"| F1 {fittest.f_perf:.4f} | {int(fittest.chromosome.sum())} features"
            )

            if fittest.fitness > best_fitness_track + self.EPSILON:
                best_fitness_track = fittest.fitness
                last_improvement_gen = generation

            # Tau-reset: sem melhora há tau_reset gerações -> perturba a população
            if generation - last_improvement_gen > self.tau_reset_generations:
                logger.info(f"  [reset] geração {generation}: perturbando a população")
                reset_pop = []
                for _ in range(self.population_size):
                    c = fittest.chromosome.copy()
                    self.mutation.bit_flip_mutation(c, 0.30)
                    reset_pop.append(Individual(c))

                self._evaluate_individuals(reset_pop)
                self.population = reset_pop
                last_improvement_gen = generation

            if self.linkage_learning:
                self.population = self._generation_ll(fittest)
            else:
                self.population = self._generation(fittest)

        self.final_fittest = self.population_analyzer.get_fittest_individual(self.population)


def executar(
    nome_modelo: str,
    parametros: dict,
    semente: int,
    dados: tuple,
    caminho: Path,
    geracoes: int = GERACOES,
    populacao: int = POPULACAO,
) -> None:
    """Roda uma execução (modelo, semente) e salva o checkpoint em caminho."""
    x_treino, y_treino, x_val, y_val, features = dados
    n = len(features)

    avaliador = AvaliadorFitness(nome_modelo, parametros, x_treino, y_treino, x_val, y_val)
    ga = GAwLLCorrigido(
        perf_batch=avaliador,
        chrom_size=n,
        mutation_probability=1.0 / n,
        max_generations=geracoes,
        pop_size=populacao,
        crossover_rate=TAXA_CRUZAMENTO,
        tau_reset=TAU_RESET,
        linkage_learning=True,
    )

    logger.info(f"Início: {nome_modelo} {parametros}, semente {semente}, N={n}")
    inicio = time.perf_counter()
    ga.run(semente)
    tempo = time.perf_counter() - inicio

    cromossomo, fitness, f1_validacao = ga.get_feature_selection_results()
    vimp = ga.importance.export_importance_vector()
    vint = ga.evig.export_interaction_matrix()

    # Fração dos pares (i < j) com alguma interação registrada
    cobertura_pares = float((vint[np.triu_indices(n, k=1)] > 0).mean())

    # Grava num arquivo temporário e só depois renomeia: se a execução morrer
    # no meio da gravação, não sobra um .npz incompleto que a retomada pularia
    temporario = caminho.with_name(caminho.name + ".tmp")
    with open(temporario, "wb") as arquivo:
        np.savez(
            arquivo,
            cromossomo=cromossomo,
            fitness=fitness,
            f1_validacao=f1_validacao,
            vimp=vimp,
            vint=vint,
            features=np.array(features),
            modelo=nome_modelo,
            parametros=str(parametros),
            semente=semente,
            geracoes=geracoes,
            populacao=populacao,
            tempo_s=tempo,
            n_treinos=avaliador.n_treinos,
            n_avaliacoes=avaliador.n_avaliacoes,
            cobertura_pares=cobertura_pares,
        )
    temporario.replace(caminho)

    selecionadas = [f for f, bit in zip(features, cromossomo) if bit]
    logger.info(
        f"Fim: {nome_modelo} semente {semente} | {tempo / 60:.1f} min | "
        f"{avaliador.n_treinos} treinos / {avaliador.n_avaliacoes} avaliações | "
        f"F1 validação {f1_validacao:.4f} | cobertura de pares {cobertura_pares:.2f}"
    )
    logger.info(f"Selecionadas ({len(selecionadas)}): {selecionadas}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PyGAwLLfs corrigido (Etapa 7)")
    parser.add_argument("--modelos", nargs="+", default=MODELOS, choices=MODELOS)
    parser.add_argument("--sementes", nargs="+", type=int, default=SEMENTES)
    parser.add_argument("--geracoes", type=int, default=GERACOES)
    parser.add_argument("--pop", type=int, default=POPULACAO)
    parser.add_argument("--saida", type=Path, default=DIR_SAIDA)
    args = parser.parse_args()

    # Caminho relativo vale a partir da raiz do projeto
    dir_saida = RAIZ / args.saida
    dir_saida.mkdir(parents=True, exist_ok=True)

    # O console já vem do basicConfig; aqui só se acrescenta o arquivo de log
    arquivo_log = logging.FileHandler(dir_saida / "gawll.log", encoding="utf-8")
    arquivo_log.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logging.getLogger().addHandler(arquivo_log)

    dados = carregar_subamostra()
    configuracoes = ler_configuracoes_escolhidas()
    logger.info(
        f"Subamostra: treino {dados[0].shape}, validação {dados[2].shape} | "
        f"{N_PROCESSOS} processos | saída {dir_saida}"
    )

    # Semente por fora: a primeira passada já entrega os três modelos
    for semente in args.sementes:
        for nome_modelo in args.modelos:
            caminho = dir_saida / f"{nome_modelo}_seed{semente}.npz"
            if caminho.exists():
                logger.info(f"Já existe, pulando: {caminho.name}")
                continue
            executar(
                nome_modelo,
                configuracoes[nome_modelo],
                semente,
                dados,
                caminho,
                geracoes=args.geracoes,
                populacao=args.pop,
            )


if __name__ == "__main__":
    main()
