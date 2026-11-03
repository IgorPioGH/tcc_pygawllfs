"""Modelagem — estatística H de Friedman e Popescu (2008) para pares de features.

Recebe o RF e o XGBoost treinados na etapa baselines e a VALIDAÇÃO (nunca o
teste). Para cada modelo, calcula o H nos 406 pares das 29 features: o mesmo
universo de pares que o PyGAwLLfs pode sondar.

Como o H é calculado (dependência parcial, PD):
    PD_j(x)   = média, sobre a amostra, da previsão com a feature j fixada em x
    PD_jk     = idem, com j e k fixadas juntas
    resíduo   = PD_jk − PD_j − PD_k   (o que a soma dos efeitos isolados não explica)
    H²_jk     = soma(resíduo²) / soma(PD_jk²)     (original, normalizado: 0 a 1)
    H_abs_jk  = raiz da média de resíduo²          (não normalizado, em unidades de logit)
Todas as PDs são centradas e avaliadas nos próprios pontos da amostra
(AMOSTRA linhas, que servem de pontos e de fundo). Custo por PD: AMOSTRA²
previsões.

Escolhas (docs/decisoes.md, decisão 12):
    - Escala logit (log-odds): "sem interação" = efeitos que se somam no logit.
    - H_abs é a versão principal para ordenar os pares (como em Inglis et al.,
      2022): o H normalizado fica instável quando o efeito conjunto do par é
      pequeno (denominador perto de zero) e pode até passar de 1 — aconteceu
      com duas dummies da mesma variável, cuja combinação "as duas ligadas"
      não existe nos dados (pares estruturais). As duas versões são salvas.
    - LogReg fora: o modelo é aditivo no logit, então H = 0 por construção
      (conferido em tests/test_interacao.py).
    - 5 sementes de amostra: o H também varia com a amostra; a média das
      sementes é o valor final, e a variação vira medida de estabilidade.

Produz results/tables/interacao_h.csv (formato longo, como interacao_gawll.csv,
com method = "friedman_h_abs" e "friedman_h" e sondada = True), mais o custo em
custo_importancia.csv. Quem consome: avaliacao/interacoes.py e a verificação no sintético
(src/avaliacao/sintetico_aplicar.py usa calcular_h).
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

from src.modelagem import comum  # noqa: E402
from src.modelagem.baselines import carregar  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELOS_H = ["random_forest", "xgboost"]
AMOSTRA = 500             # pontos (e fundo) das dependências parciais
LIMITE_PROBABILIDADE = 1e-6


def prever_logit(modelo):
    """Devolve uma função x -> log-odds da classe 1, a partir do predict_proba do modelo
    (probabilidade cortada em LIMITE_PROBABILIDADE para não dar log de 0).

    Usada em: main; sintetico_aplicar.interacoes_h; tests/test_interacao.py.
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

    Usada em: main; sintetico_aplicar.interacoes_h; tests/test_interacao.py.
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
    par é escrito na ordem das colunas do dataset, como no PyGAwLLfs; sondada = True,
    porque o H mede todo par (sem a coluna, a concatenação com o PyGAwLLfs deixaria
    o H com valores vazios).

    Usada em: main; sintetico_aplicar.interacoes_h.
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
            "sondada": True,
        }))
    return pd.concat(tabelas, ignore_index=True)


def main() -> None:
    """Para RF e XGBoost, calcula o H nos 406 pares em 5 amostras de 500
    linhas da validação e salva interacao_h.csv e o custo.

    Usada em: python src/modelagem/friedman_h.py; main.py, etapa "importancias";
    tests/test_interacao.py.
    Importância: gera a referência de interação contra a qual o PyGAwLLfs é
    comparado.
    """
    x_val, _ = carregar("val")
    features = list(x_val.columns)
    tabelas, custos = [], []
    for modelo_nome in MODELOS_H:
        modelo = comum.carregar_modelo(modelo_nome)
        prever = prever_logit(modelo)
        logger.info(f"{modelo_nome}: {len(features)} features, "
                    f"{len(features) * (len(features) - 1) // 2} pares")
        for semente in comum.SEMENTES:
            amostra = x_val.sample(n=AMOSTRA, random_state=semente)
            inicio = time.perf_counter()
            h = calcular_h(prever, amostra, features)
            tempo = time.perf_counter() - inicio
            tabelas.append(formato_longo(h, modelo_nome, semente, features))
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
