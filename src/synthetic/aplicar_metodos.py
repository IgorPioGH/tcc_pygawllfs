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
    H Friedman   interaction/friedman_h.calcular_h (Etapa 13), RF e XGBoost, todos os
                 45 pares, 5 amostras de 500 linhas da validação

Produz, em results/tables/:
    sintetico_importancia.csv  formato padrão (§7), os quatro métodos
    sintetico_interacao.csv    vint do PyGAwLLfs e H de Friedman, em formato longo
    sintetico_selecao.csv      o x* de cada execução
    sintetico_custo.csv        tempo e nº de treinos
Quem consome: recovery.py (compara tudo com o gabarito, inclusive o H:
method = "friedman_h_abs" e "friedman_h").
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
from src.interaction.friedman_h import (  # noqa: E402
    AMOSTRA,
    MODELOS_H,
    calcular_h,
    formato_longo,
    prever_logit,
)
from src.models.baselines import SEMENTE, criar_modelo, ler_configuracoes_escolhidas  # noqa: E402
from src.synthetic.generate import ARQUIVO_SINTETICO, FEATURES  # noqa: E402

logger = logging.getLogger(__name__)

DIR_EXECUCOES = RAIZ / "experiments" / "runs" / "sintetico"
MODELOS = ["logreg", "random_forest", "xgboost"]
MODELOS_ARVORE = ["random_forest", "xgboost"]


def carregar_sintetico() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Devolve (x_treino, y_treino, x_val, y_val) do dataset sintético.

    Usada em: main.
    Importância: mesma interface dos dados reais, para os métodos rodarem sem
    adaptação.
    """
    df = pd.read_parquet(ARQUIVO_SINTETICO)
    treino = df[df["conjunto"] == "treino"]
    validacao = df[df["conjunto"] == "validacao"]
    return treino[FEATURES], treino["y"], validacao[FEATURES], validacao["y"]


def importancias_modelo_fixo(modelos, x_val, y_val) -> tuple[list[pd.DataFrame], list[dict]]:
    """Intrínseca, Permutation e SHAP sobre os modelos já treinados, na validação, com
    as mesmas funções da Etapa 9.

    Usada em: main.
    Importância: o gabarito julga os métodos exatamente como foram usados nos dados
    reais.
    """
    tabelas, custos = [], []

    def registrar(features, valores, metodo, nome, semente, inicio, por_absoluto=False):
        """Acrescenta a tabela no formato padrão e a linha de custo de um (método,
        modelo, semente).

        Usada em: importancias_modelo_fixo.
        Importância: evita repetir o mesmo bloco para cada método.
        """
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


def interacoes_h(modelos, x_val) -> tuple[list[pd.DataFrame], list[dict]]:
    """H de Friedman de todos os pares, no RF e no XGBoost (a LogReg é aditiva no logit:
    H = 0 por construção). Uma amostra de AMOSTRA linhas por semente.

    Usada em: main.
    Importância: permite julgar PyGAwLLfs e H contra o par de interação verdadeiro
    (x1, x2).
    """
    tabelas, custos = [], []
    for nome in MODELOS_H:
        prever = prever_logit(modelos[nome])
        for semente in SEMENTES:
            inicio = time.perf_counter()
            h = calcular_h(prever, x_val.sample(n=AMOSTRA, random_state=semente), FEATURES)
            tabelas.append(formato_longo(h, nome, semente, FEATURES))
            custos.append({"method": "friedman_h", "model": nome, "seed": semente,
                           "tempo_s": time.perf_counter() - inicio, "n_treinos": 0})
        logger.info(f"{nome}: H de Friedman pronto")
    return tabelas, custos


def rodar_gawll(x_treino, y_treino, x_val, y_val, configuracoes) -> None:
    """15 execuções do PyGAwLLfs corrigido (3 modelos × 5 sementes) com gawll.executar;
    pula as que já têm checkpoint.

    Usada em: main.
    Importância: é a parte cara do Diferencial 1 (~30 min).
    """
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
    """Etapa 10b: treina os três modelos no sintético, roda os quatro métodos e o H e
    salva sintetico_{importancia,interacao,selecao,custo}.csv.

    Usada em: python src/synthetic/aplicar_metodos.py; main.py, etapa "sintetico".
    Importância: produz o material do Diferencial 1 (qual método recupera a verdade).
    """
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
    tabelas_h, custos_h = interacoes_h(modelos, x_val)

    # 2. PyGAwLLfs: retreina o modelo para cada subconjunto
    rodar_gawll(x_treino, y_treino, x_val, y_val, configuracoes)
    importancia_ga, interacao_ga, selecao_ga, custos_ga = consolidar(DIR_EXECUCOES)

    salvar_tabela(pd.concat(tabelas + [importancia_ga]), "sintetico_importancia.csv")
    salvar_tabela(pd.concat([interacao_ga] + tabelas_h, ignore_index=True),
                  "sintetico_interacao.csv")
    salvar_tabela(selecao_ga, "sintetico_selecao.csv")
    salvar_tabela(pd.DataFrame(custos + custos_h + custos_ga), "sintetico_custo.csv")
    logger.info("Tabelas do sintético salvas em results/tables/sintetico_*.csv")


if __name__ == "__main__":
    # Log em arquivo, porque as 15 execuções do GA levam alguns minutos
    DIR_EXECUCOES.mkdir(parents=True, exist_ok=True)
    arquivo_log = logging.FileHandler(DIR_EXECUCOES / "sintetico.log", encoding="utf-8")
    arquivo_log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(arquivo_log)
    main()
