"""Consolidação final — quadro-síntese dos métodos e dos modelos.

Recebe as tabelas já produzidas pelas Etapas 6, 9, 11 e 12 (só CSV: nenhum
dado é lido e nenhum modelo é treinado) e junta, lado a lado, os quatro
critérios da pergunta de pesquisa, para cada método e para cada modelo:

    concordância   Spearman e RBO médios com os outros métodos do MESMO modelo
                   (concordancia.csv, pares mesmo_modelo, §4.11)
    estabilidade   Spearman e top-5 médios entre as 10 duplas de sementes
                   (estabilidade.csv; a intrínseca é determinística e fica vazia)
    fidelidade     fração do PR-AUC do modelo completo retida no TESTE pelo
                   top-5 e pelo top-10, e a mesma fração para 5 features sorteadas
                   (fidelidade_resumo.csv e fidelidade.csv)
    custo          tempo e nº de treinos de modelo por execução
                   (custo_importancia.csv)

Produz:
    sintese_metodos.csv       uma linha por (método, modelo)
    sintese_modelos.csv       uma linha por modelo: desempenho no teste (Etapa 6),
                              concordância média entre os métodos e fidelidade
                              média do top-5, contra o acaso
    sintese_top_features.csv  as 5 features mais importantes de cada (método,
                              modelo), na mesma ordem usada na fidelidade
Quem consome: results/relatorio-final.md (quadro-síntese e respostas às
perguntas de pesquisa).
"""

import logging
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.compare.rankings import METODOS, MODELOS, ordenar  # noqa: E402
from src.importance import comum  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

N_TOP = 5  # quantas features listar por (método, modelo)


def concordancia_intra(concordancia: pd.DataFrame) -> pd.DataFrame:
    """Para cada (método, modelo): média de Spearman, RBO e top-5 nas duplas com os
    outros métodos do mesmo modelo.

    Usada em: sintese_metodos; tests/test_sintese.py.
    Importância: resume o critério de concordância por método, sempre dentro do
    mesmo modelo (§4.11), para caber numa linha do quadro-síntese.
    """
    intra = concordancia[concordancia["tipo_par"] == "mesmo_modelo"]
    # Cada dupla conta para os dois lados: (a, b) entra na média de a e na de b
    lado_a = intra.rename(columns={"method_a": "method", "model_a": "model"})
    lado_b = intra.rename(columns={"method_b": "method", "model_b": "model"})
    colunas = ["method", "model", "spearman", "rbo", "top5"]
    duplas = pd.concat([lado_a[colunas], lado_b[colunas]], ignore_index=True)
    return (duplas.groupby(["method", "model"], as_index=False)
            .mean()
            .rename(columns={"spearman": "concordancia_spearman",
                             "rbo": "concordancia_rbo",
                             "top5": "concordancia_top5"}))


def fidelidade_por_metodo(resumo: pd.DataFrame, fidelidade: pd.DataFrame) -> pd.DataFrame:
    """Para cada (método, modelo): fração do PR-AUC retida no teste pelo top-5 e pelo
    top-10; o acaso (5 features sorteadas) do mesmo modelo; e, no PyGAwLLfs, a
    média do x* nas 5 execuções e o nº médio de features dele.

    Usada em: sintese_metodos; tests/test_sintese.py.
    Importância: a fidelidade é o único critério medido no teste; o acaso dá a
    régua para ler a fração retida.
    """
    teste = resumo[resumo["conjunto"] == "test"]
    linhas = []
    for (metodo, modelo), grupo in teste.groupby(["method", "model"]):
        if metodo not in METODOS:
            continue  # "todas", "aleatorio" e "gawll_x*" não são métodos de ranking
        por_k = grupo.set_index("k")["pr_auc_frac_todas"]
        acaso = teste[(teste["model"] == modelo) & (teste["method"] == "aleatorio")
                      & (teste["k"] == 5)]["pr_auc_frac_todas"].iloc[0]
        linhas.append({"method": metodo, "model": modelo,
                       "fidelidade_top5": por_k[5], "fidelidade_top10": por_k[10],
                       "acaso_top5": acaso})
    tabela = pd.DataFrame(linhas)

    # x*: uma linha por execução em fidelidade.csv; a fração é sobre "todas"
    teste_completo = fidelidade[fidelidade["conjunto"] == "test"]
    todas = teste_completo[teste_completo["method"] == "todas"].set_index("model")["pr_auc"]
    x_estrela = teste_completo[teste_completo["method"] == "gawll_x*"].copy()
    x_estrela["fracao"] = x_estrela["pr_auc"] / x_estrela["model"].map(todas)
    x_estrela = (x_estrela.groupby("model", as_index=False)
                 .agg(fidelidade_x_estrela=("fracao", "mean"), n_features_x_estrela=("k", "mean"))
                 .assign(method="gawll"))
    return tabela.merge(x_estrela, on=["method", "model"], how="left")


def custo_por_metodo(custo: pd.DataFrame) -> pd.DataFrame:
    """Para cada (método, modelo): nº de execuções, tempo médio por execução e nº
    médio de treinos de modelo por execução.

    Usada em: sintese_metodos.
    Importância: o nº de treinos é a medida de custo que não depende da máquina (§7).
    """
    return (custo[custo["method"].isin(METODOS)]
            .groupby(["method", "model"], as_index=False)
            .agg(n_execucoes=("seed", "count"),
                 tempo_por_execucao_s=("tempo_s", "mean"),
                 treinos_por_execucao=("n_treinos", "mean")))


def sintese_metodos(concordancia: pd.DataFrame, estabilidade: pd.DataFrame,
                    resumo_fidelidade: pd.DataFrame, fidelidade: pd.DataFrame,
                    custo: pd.DataFrame) -> pd.DataFrame:
    """Junta os quatro critérios numa linha por (método, modelo), na ordem de
    MODELOS e METODOS.

    Usada em: main; tests/test_sintese.py.
    Importância: é o quadro que responde à pergunta de pesquisa de uma vez.
    """
    estab = estabilidade[["method", "model", "spearman", "top5"]].rename(
        columns={"spearman": "estabilidade_spearman", "top5": "estabilidade_top5"})
    tabela = (concordancia_intra(concordancia)
              .merge(estab, on=["method", "model"], how="left")
              .merge(fidelidade_por_metodo(resumo_fidelidade, fidelidade),
                     on=["method", "model"], how="left")
              .merge(custo_por_metodo(custo), on=["method", "model"], how="left"))
    tabela["fidelidade_top5_acima_acaso"] = tabela["fidelidade_top5"] - tabela["acaso_top5"]
    ordem = {(metodo, modelo): i for i, (modelo, metodo) in
             enumerate((m, t) for m in MODELOS for t in METODOS)}
    tabela["ordem"] = [ordem[(m, t)] for m, t in zip(tabela["method"], tabela["model"])]
    return tabela.sort_values("ordem").drop(columns="ordem").reset_index(drop=True)


def sintese_modelos(baselines: pd.DataFrame, concordancia: pd.DataFrame,
                    metodos: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por modelo: métricas no teste (Etapa 6), concordância média entre os
    métodos daquele modelo e fidelidade média do top-5, contra o acaso.

    Usada em: main; tests/test_sintese.py.
    Importância: compara os modelos pelas duas lentes do trabalho: quão bem preveem
    e quão de acordo ficam as explicações deles.
    """
    linhas = []
    desempenho = baselines.set_index("modelo")
    for modelo in MODELOS:
        intra = concordancia[(concordancia["tipo_par"] == "mesmo_modelo")
                             & (concordancia["model_a"] == modelo)]
        do_modelo = metodos[metodos["model"] == modelo]
        linhas.append({
            "model": modelo,
            **{metrica: desempenho.loc[modelo, metrica]
               for metrica in ["f1", "pr_auc", "roc_auc", "acuracia_balanceada"]},
            "n_metodos": len(do_modelo),
            "concordancia_spearman_media": intra["spearman"].mean(),
            "concordancia_spearman_min": intra["spearman"].min(),
            "fidelidade_top5_media": do_modelo["fidelidade_top5"].mean(),
            "acaso_top5": do_modelo["acaso_top5"].iloc[0],
        })
    return pd.DataFrame(linhas)


def top_features(ranking: pd.DataFrame, n: int = N_TOP) -> pd.DataFrame:
    """As n features mais importantes de cada (método, modelo), em colunas top1..topn.

    Usa rankings.ordenar sobre a pontuação consolidada, a mesma ordem que definiu
    os top-k da fidelidade.

    Usada em: main; tests/test_sintese.py.
    Importância: mostra concretamente sobre o que os métodos concordam.
    """
    linhas = []
    for modelo in MODELOS:
        for metodo in METODOS:
            grupo = ranking[(ranking["method"] == metodo) & (ranking["model"] == modelo)]
            if grupo.empty:
                continue  # SHAP não tem LogReg
            ordem = ordenar(grupo.set_index("feature")["pontuacao"])
            linhas.append({"model": modelo, "method": metodo,
                           **{f"top{i + 1}": feature for i, feature in enumerate(ordem[:n])}})
    return pd.DataFrame(linhas)


def main() -> None:
    """Lê as tabelas finais, monta as três sínteses e salva em results/tables/.

    Usada em: python src/compare/sintese.py; main.py, etapa "sintese";
    tests/test_sintese.py.
    Importância: é a última etapa do pipeline; junta tudo o que o relatório final
    precisa sem recalcular nada.
    """
    ler = lambda nome: pd.read_csv(comum.DIR_TABELAS / nome)  # noqa: E731
    concordancia = ler("concordancia.csv")
    metodos = sintese_metodos(concordancia, ler("estabilidade.csv"),
                              ler("fidelidade_resumo.csv"), ler("fidelidade.csv"),
                              ler("custo_importancia.csv"))
    modelos = sintese_modelos(ler("baselines.csv"), concordancia, metodos)
    tops = top_features(ler("ranking_consolidado.csv"))

    comum.salvar_tabela(metodos.round(4), "sintese_metodos.csv")
    comum.salvar_tabela(modelos.round(4), "sintese_modelos.csv")
    comum.salvar_tabela(tops, "sintese_top_features.csv")

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        logger.info("Métodos:\n" + metodos.round(3).to_string(index=False))
        logger.info("Modelos:\n" + modelos.round(3).to_string(index=False))
        logger.info("Top-5:\n" + tops.to_string(index=False))


if __name__ == "__main__":
    main()
