"""Etapa 10 — Diferencial 1: aplica os quatro métodos ao dataset sintético.

Recebe data/processed/sintetico.parquet (generate.py) e os hiperparâmetros
escolhidos na Etapa 6 (baselines.csv). Treina os três modelos com
criar_modelo() — os mesmos do VRA — e roda exatamente o mesmo código das
Etapas 7 e 9 (docs/metodologia.md §5: "usa exatamente o mesmo código de comparação"):

    intrínseca   intrinsic.importancia_intrinseca        determinística, seed 42
    permutation  permutation.calcular_permutation        5 sementes
    shap         shap_global.calcular_shap (RF, XGBoost) a validação inteira
                 (1.500 linhas < 10k) entra toda, então é determinístico: seed 42
    gawll        gawll.executar, 3 modelos x 5 sementes, 100 gerações,
                 checkpoints em experiments/runs/sintetico/ (retomada igual à Etapa 8)

Produz, em results/tables/:
    sintetico_importancia.csv  formato padrão (§7), os quatro métodos
    sintetico_interacao.csv    vint do PyGAwLLfs em formato longo
    sintetico_selecao.csv      o x* de cada execução
    sintetico_custo.csv        tempo e nº de treinos
Quem consome: recovery.py (compara tudo com o gabarito) e, na Etapa 13, o
H de Friedman entra em sintetico_interacao.csv com method="friedman_h".
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))
sys.path.append(str(RAIZ / "PyGAwLLfs"))

from src.importance import gawll  # noqa: E402
from src.importance.comum import SEMENTES, salvar_tabela, tabela_importancia  # noqa: E402
from src.importance.consolidar_gawll import consolidar  # noqa: E402
from src.importance.intrinsic import importancia_intrinseca  # noqa: E402
from src.importance.permutation import calcular_permutation  # noqa: E402
from src.importance.shap_global import calcular_shap  # noqa: E402
from src.models.baselines import SEMENTE, criar_modelo, ler_configuracoes_escolhidas  # noqa: E402
from src.synthetic.generate import ARQUIVO_SINTETICO, FEATURES  # noqa: E402

logger = logging.getLogger(__name__)

DIR_EXECUCOES = RAIZ / "experiments" / "runs" / "sintetico"
MODELOS = ["logreg", "random_forest", "xgboost"]
MODELOS_ARVORE = ["random_forest", "xgboost"]


def carregar_sintetico() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Devolve (x_treino, y_treino, x_val, y_val) do dataset sintético."""
    df = pd.read_parquet(ARQUIVO_SINTETICO)
    treino = df[df["conjunto"] == "treino"]
    validacao = df[df["conjunto"] == "validacao"]
    return treino[FEATURES], treino["y"], validacao[FEATURES], validacao["y"]


def importancias_modelo_fixo(modelos, x_val, y_val) -> tuple[list[pd.DataFrame], list[dict]]:
    """Intrínseca, Permutation e SHAP sobre os modelos já treinados, na validação."""
    tabelas, custos = [], []

    def registrar(features, valores, metodo, nome, semente, inicio, por_absoluto=False):
        tabelas.append(tabela_importancia(
            features, valores, metodo, nome, semente, rank_por_absoluto=por_absoluto
        ))
        custos.append({"method": metodo, "model": nome, "seed": semente,
                       "tempo_s": time.perf_counter() - inicio, "n_treinos": 0})

    for nome, modelo in modelos.items():
        inicio = time.perf_counter()
        features, valores = importancia_intrinseca(nome, modelo)
        registrar(features, valores, "intrinseca", nome, SEMENTE, inicio,
                  por_absoluto=(nome == "logreg"))

        for semente in SEMENTES:
            inicio = time.perf_counter()
            valores = calcular_permutation(modelo, x_val, y_val, semente)
            registrar(FEATURES, valores, "permutation", nome, semente, inicio)

        if nome in MODELOS_ARVORE:
            inicio = time.perf_counter()
            valores = calcular_shap(modelo, x_val)
            registrar(FEATURES, valores, "shap", nome, SEMENTE, inicio)

        logger.info(f"{nome}: intrínseca, permutation e shap prontos")
    return tabelas, custos


def rodar_gawll(x_treino, y_treino, x_val, y_val, configuracoes) -> None:
    """15 execuções do PyGAwLLfs corrigido; pula as que já têm checkpoint."""
    dados = (
        x_treino.to_numpy(dtype=float), y_treino.to_numpy(dtype=int),
        x_val.to_numpy(dtype=float), y_val.to_numpy(dtype=int),
        list(x_treino.columns),
    )
    DIR_EXECUCOES.mkdir(parents=True, exist_ok=True)
    for semente in SEMENTES:
        for nome in MODELOS:
            caminho = DIR_EXECUCOES / f"{nome}_seed{semente}.npz"
            if caminho.exists():
                logger.info(f"Já existe, pulando: {caminho.name}")
                continue
            gawll.executar(nome, configuracoes[nome], semente, dados, caminho)


def main() -> None:
    x_treino, y_treino, x_val, y_val = carregar_sintetico()
    configuracoes = ler_configuracoes_escolhidas()
    peso_positivo = (y_treino == 0).sum() / (y_treino == 1).sum()
    logger.info(f"Sintético: treino {x_treino.shape}, validação {x_val.shape}")

    # 1. Modelos fixos (treinados uma vez) + os três métodos que os explicam
    modelos = {}
    for nome in MODELOS:
        modelos[nome] = criar_modelo(nome, configuracoes[nome], peso_positivo)
        modelos[nome].fit(x_treino, y_treino)
    tabelas, custos = importancias_modelo_fixo(modelos, x_val, y_val)

    # 2. PyGAwLLfs: retreina o modelo para cada subconjunto
    rodar_gawll(x_treino, y_treino, x_val, y_val, configuracoes)
    importancia_ga, interacao_ga, selecao_ga, custos_ga = consolidar(DIR_EXECUCOES)

    salvar_tabela(pd.concat(tabelas + [importancia_ga]), "sintetico_importancia.csv")
    salvar_tabela(interacao_ga, "sintetico_interacao.csv")
    salvar_tabela(selecao_ga, "sintetico_selecao.csv")
    salvar_tabela(pd.DataFrame(custos + custos_ga), "sintetico_custo.csv")
    logger.info("Tabelas do sintético salvas em results/tables/sintetico_*.csv")


if __name__ == "__main__":
    # Log em arquivo, porque as 15 execuções do GA levam alguns minutos
    DIR_EXECUCOES.mkdir(parents=True, exist_ok=True)
    arquivo_log = logging.FileHandler(DIR_EXECUCOES / "sintetico.log", encoding="utf-8")
    arquivo_log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(arquivo_log)
    main()
