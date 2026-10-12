"""Etapa 13 — estatística H de Friedman e Popescu (2008) para pares de features.

Recebe o RF e o XGBoost treinados na Etapa 6, a VALIDAÇÃO (nunca o teste,
§4.9), o ranking_consolidado.csv (Etapa 11) e o interacao_gawll.csv (Etapa 9).
Para cada modelo, calcula o H em todos os pares do conjunto do §4.7:
    união de (a) as 15 features de melhor rank médio entre os métodos de
    importância e (b) as features do grafo reduzido do PyGAwLLfs (5% das
    arestas). Só (a) excluiria, por construção, as features que o PyGAwLLfs
    aponta como mais interativas.

Como o H é calculado (dependência parcial, PD):
    PD_j(x)   = média, sobre a amostra, da previsão com a feature j fixada em x
    PD_jk     = idem, com j e k fixadas juntas
    resíduo   = PD_jk − PD_j − PD_k   (o que a soma dos efeitos isolados não explica)
    H²_jk     = soma(resíduo²) / soma(PD_jk²)     (original, normalizado: 0 a 1)
    H_abs_jk  = raiz da média de resíduo²          (não normalizado, em unidades de logit)
Todas as PDs são centradas e avaliadas nos próprios pontos da amostra
(AMOSTRA linhas, que servem de pontos e de fundo). Custo por PD: AMOSTRA²
previsões.

Escolhas (docs/decisoes.md §12):
    - Escala logit (log-odds), a mesma da validação de domínio (Etapa 15):
      "sem interação" = efeitos que se somam no logit.
    - H_abs é a versão principal para ordenar os pares (como em Inglis et al.,
      2022): o H normalizado fica instável quando o efeito conjunto do par é
      pequeno (denominador perto de zero) e pode até passar de 1 — aconteceu
      com duas dummies da mesma variável, cuja combinação "as duas ligadas"
      não existe nos dados. As duas versões são salvas.
    - LogReg fora: o modelo é aditivo no logit, então H = 0 por construção
      (conferido em tests/test_interacao.py).
    - 5 sementes de amostra: o H também varia com a amostra; a média das
      sementes é o valor final, e a variação vira medida de estabilidade.

Produz results/tables/interacao_h.csv (formato longo, como interacao_gawll.csv,
com method = "friedman_h_abs" e "friedman_h"), mais o custo em
custo_importancia.csv. Quem consome: compare/interacoes.py e o Diferencial 1
(src/synthetic/aplicar_metodos.py usa calcular_h).
"""

import logging
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.importance import comum  # noqa: E402
from src.interaction.graph import arestas_reduzidas, features_do_grafo, interacao_media  # noqa: E402
from src.models.baselines import carregar  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELOS_H = ["random_forest", "xgboost"]
AMOSTRA = 500             # pontos (e fundo) das dependências parciais (§4.7)
TOP_CONSENSO = 15         # parte (a) do conjunto do §4.7
LIMITE_PROBABILIDADE = 1e-6


def prever_logit(modelo):
    """Devolve uma função x -> log-odds da classe 1, a partir do predict_proba do modelo
    (probabilidade cortada em LIMITE_PROBABILIDADE para não dar log de 0).

    Usada em: main; aplicar_metodos.interacoes_h; tests/test_interacao.py.
    Importância: no logit a LogReg é exatamente aditiva (H = 0); medir nessa escala
    evita interações criadas só pela sigmoide.
    """
    def prever(x: pd.DataFrame) -> np.ndarray:
        """x -> logit da probabilidade prevista pelo modelo.

        Usada em: dependencia_parcial e calcular_h (é a função que prever_logit
        devolve).
        Importância: é a função que o H decompõe.
        """
        p = np.clip(modelo.predict_proba(x)[:, 1], LIMITE_PROBABILIDADE, 1 - LIMITE_PROBABILIDADE)
        return np.log(p / (1 - p))
    return prever


def dependencia_parcial(prever, amostra: pd.DataFrame, colunas: list[str]) -> np.ndarray:
    """PD centrada das "colunas", avaliada em cada linha da amostra. Monta n blocos de n
    linhas: o bloco i é a amostra inteira com as "colunas" trocadas pelos valores da
    linha i; a média de cada bloco é PD(x_i).

    Usada em: calcular_h; tests/test_interacao.py.
    Importância: é o bloco básico do H; custa n² previsões por conjunto de colunas,
    daí a amostra de 500 linhas.
    """
    n = len(amostra)
    grade = pd.DataFrame(np.tile(amostra.to_numpy(), (n, 1)), columns=amostra.columns)
    for coluna in colunas:
        grade[coluna] = np.repeat(amostra[coluna].to_numpy(), n)
    pd_valores = prever(grade).reshape(n, n).mean(axis=1)
    return pd_valores - pd_valores.mean()


def calcular_h(prever, amostra: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """H de Friedman e Popescu (2008) em todos os pares de "features": resíduo = PD_ab −
    PD_a − PD_b; h_abs = raiz da média do resíduo²; h_normalizado = soma do resíduo²
    / soma de PD_ab².

    Usada em: main; aplicar_metodos.interacoes_h; tests/test_interacao.py.
    Importância: método de referência para interação, comparado ao PyGAwLLfs. O
    normalizado passa de 1 quando PD_ab é quase nulo, por isso o h_abs é o principal.
    """
    individuais = {f: dependencia_parcial(prever, amostra, [f]) for f in features}
    linhas = []
    for a, b in combinations(features, 2):
        conjunta = dependencia_parcial(prever, amostra, [a, b])
        residuo = conjunta - individuais[a] - individuais[b]
        soma_conjunta = np.sum(conjunta ** 2)
        linhas.append({
            "feature_a": a,
            "feature_b": b,
            "h_normalizado": float(np.sum(residuo ** 2) / soma_conjunta) if soma_conjunta > 0 else 0.0,
            "h_abs": float(np.sqrt(np.mean(residuo ** 2))),
        })
    return pd.DataFrame(linhas)


def formato_longo(h: pd.DataFrame, modelo: str, semente: int, ordem: list[str]) -> pd.DataFrame:
    """Duas linhas por par (uma por versão do H), no formato de interacao_gawll.csv. O
    par é escrito na ordem das colunas do dataset, como no PyGAwLLfs.

    Usada em: main; aplicar_metodos.interacoes_h.
    Importância: deixa o H no mesmo formato da interação do PyGAwLLfs, para a Etapa
    13b comparar direto.
    """
    posicao = {f: i for i, f in enumerate(ordem)}
    trocar = h["feature_a"].map(posicao) > h["feature_b"].map(posicao)
    h = h.copy()
    h.loc[trocar, ["feature_a", "feature_b"]] = h.loc[trocar, ["feature_b", "feature_a"]].to_numpy()
    tabelas = []
    for metodo, coluna in [("friedman_h_abs", "h_abs"), ("friedman_h", "h_normalizado")]:
        tabelas.append(pd.DataFrame({
            "feature_a": h["feature_a"], "feature_b": h["feature_b"],
            "interaction": h[coluna], "method": metodo, "model": modelo, "seed": semente,
        }))
    return pd.concat(tabelas, ignore_index=True)


def conjunto_de_features(modelo: str) -> list[str]:
    """União do §4.7: top-15 por consenso entre os métodos + todas as features do grafo
    reduzido do PyGAwLLfs daquele modelo.

    Usada em: main; tests/test_interacao.py.
    Importância: limita o custo O(N²) sem excluir por construção as features que o
    PyGAwLLfs aponta como interativas.
    """
    ranking = pd.read_csv(comum.DIR_TABELAS / "ranking_consolidado.csv")
    consenso = (ranking[ranking["model"] == modelo].groupby("feature")["rank"].mean()
                .nsmallest(TOP_CONSENSO).index)
    gawll = interacao_media(pd.read_csv(comum.DIR_TABELAS / "interacao_gawll.csv"))
    grafo = features_do_grafo(arestas_reduzidas(gawll[gawll["model"] == modelo]))
    return sorted(set(consenso) | grafo)


def main() -> None:
    """Etapa 13a: para RF e XGBoost, calcula o H na união do §4.7 em 5 amostras de 500
    linhas da validação e salva interacao_h.csv e o custo.

    Usada em: python src/interaction/friedman_h.py; main.py, etapa "interacoes";
    tests/test_interacao.py.
    Importância: gera a referência de interação contra a qual o PyGAwLLfs é
    comparado.
    """
    x_val, _ = carregar("val")
    ordem = list(x_val.columns)
    tabelas, custos = [], []
    for modelo_nome in MODELOS_H:
        modelo = comum.carregar_modelo(modelo_nome)
        prever = prever_logit(modelo)
        features = [f for f in ordem if f in conjunto_de_features(modelo_nome)]
        logger.info(f"{modelo_nome}: {len(features)} features, "
                    f"{len(features) * (len(features) - 1) // 2} pares")
        for semente in comum.SEMENTES:
            amostra = x_val.sample(n=AMOSTRA, random_state=semente)
            inicio = time.perf_counter()
            h = calcular_h(prever, amostra, features)
            tempo = time.perf_counter() - inicio
            tabelas.append(formato_longo(h, modelo_nome, semente, ordem))
            custos.append({"method": "friedman_h", "model": modelo_nome, "seed": semente,
                           "tempo_s": tempo, "n_treinos": 0})
            mais_fortes = h.nlargest(3, "h_abs")
            logger.info(f"  semente {semente}: {tempo:.0f}s | maiores H_abs: "
                        + ", ".join(f"{r.feature_a} × {r.feature_b} ({r.h_abs:.3f})"
                                    for r in mais_fortes.itertuples()))

    caminho = comum.salvar_tabela(pd.concat(tabelas, ignore_index=True), "interacao_h.csv")
    comum.salvar_custo(custos)
    logger.info(f"Salvo em {caminho}")


if __name__ == "__main__":
    main()
