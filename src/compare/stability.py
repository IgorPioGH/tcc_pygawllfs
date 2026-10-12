"""Etapa 11 — estabilidade dos rankings entre sementes.

Recebe as tabelas de importância da Etapa 9 (results/tables/importancia_*.csv)
e, para cada (método, modelo) que tem 5 sementes, mede quanto o ranking muda
de uma semente para outra. A intrínseca fica de fora: é determinística.

O que muda com a semente em cada método (o modelo explicado é sempre o mesmo):
    permutation  a ordem do embaralhamento das colunas
    shap         a amostra de 10 mil linhas da validação
    gawll        a busca inteira (população inicial, seleção, cruzamento, mutação)

Produz:
    estabilidade.csv           por (método, modelo): média e mínimo, nas 10
                               duplas de sementes, de Spearman, Kendall tau-b,
                               RBO e sobreposição top-5/10/15
    estabilidade_features.csv  por feature: rank médio, desvio-padrão do rank e
                               frequência no top-5 e no top-10 entre as sementes
Quem consome: o Cap. 4 (estabilidade é um dos quatro critérios da pergunta de
pesquisa) e a figura de estabilidade em src/compare/figuras.py.
"""

import logging
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.compare.rankings import METODOS, MODELOS, TOP_K, carregar_importancias, comparar  # noqa: E402
from src.importance import comum  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

METRICAS = ["spearman", "kendall_tau_b", "rbo"] + [f"top{k}" for k in TOP_K]


def concordancia_entre_sementes(grupo: pd.DataFrame) -> dict:
    """Um (método, modelo): as métricas de rankings.comparar em todas as duplas de
    sementes, resumidas em média e mínimo.

    Usada em: main; interacoes.comparar_modelo; tests/test_comparacao.py.
    Importância: é a medida do critério de estabilidade; o mínimo mostra o pior caso,
    não só a média.
    """
    por_semente = {
        semente: linhas.set_index("feature")["pontuacao"]
        for semente, linhas in grupo.groupby("seed")
    }
    duplas = [comparar(por_semente[a], por_semente[b])
              for a, b in combinations(sorted(por_semente), 2)]
    duplas = pd.DataFrame(duplas)

    resultado = {"n_sementes": len(por_semente), "n_duplas": len(duplas)}
    for metrica in METRICAS:
        resultado[metrica] = duplas[metrica].mean()
        resultado[f"{metrica}_min"] = duplas[metrica].min()
    return resultado


def estabilidade_por_feature(grupo: pd.DataFrame) -> pd.DataFrame:
    """Um (método, modelo): como o rank de cada feature varia entre as sementes. Usa a
    coluna "rank" de cada semente (rank 1 = mais importante, já com o valor absoluto
    no coeficiente da LogReg).

    Usada em: main; tests/test_comparacao.py.
    Importância: mostra quais features ficam firmes no topo mesmo quando o ranking
    inteiro oscila.
    """
    por_feature = grupo.groupby("feature")["rank"]
    return pd.DataFrame({
        "rank_medio": por_feature.mean(),
        "rank_desvio": por_feature.std(ddof=0),
        "freq_top5": por_feature.apply(lambda r: (r <= 5).mean()),
        "freq_top10": por_feature.apply(lambda r: (r <= 10).mean()),
    }).reset_index().sort_values("rank_medio")


def main() -> None:
    """Etapa 11b: mede a estabilidade de cada (método, modelo) com mais de uma semente e
    salva estabilidade.csv e estabilidade_features.csv.

    Usada em: python src/compare/stability.py; main.py, etapa "rankings";
    tests/test_comparacao.py.
    Importância: responde ao critério de estabilidade; a intrínseca é determinística
    e fica de fora.
    """
    importancias = carregar_importancias()

    resumo, por_feature = [], []
    for modelo in MODELOS:
        for metodo in METODOS:
            grupo = importancias[(importancias["method"] == metodo)
                                 & (importancias["model"] == modelo)]
            if grupo["seed"].nunique() < 2:
                continue  # intrínseca (determinística) ou combinação inexistente
            resumo.append({"method": metodo, "model": modelo,
                           **concordancia_entre_sementes(grupo)})
            por_feature.append(estabilidade_por_feature(grupo)
                               .assign(method=metodo, model=modelo))

    resumo = pd.DataFrame(resumo)
    por_feature = pd.concat(por_feature, ignore_index=True)
    colunas = ["method", "model", "feature", "rank_medio", "rank_desvio", "freq_top5", "freq_top10"]
    por_feature = por_feature[colunas]

    comum.salvar_tabela(resumo, "estabilidade.csv")
    comum.salvar_tabela(por_feature, "estabilidade_features.csv")

    colunas_log = ["method", "model", "spearman", "spearman_min", "rbo", "rbo_min", "top5", "top10"]
    with pd.option_context("display.width", 200):
        logger.info("Concordância entre sementes (média e mínimo nas 10 duplas):\n"
                    + resumo[colunas_log].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
