"""Etapa 12b — incerteza dos resultados no teste: bootstrap por dias, pareado.

Recebe:
    data/processed/test.parquet                      alvo e data de cada voo do teste
    experiments/runs/baselines/<modelo>.joblib       os três modelos da Etapa 6
    experiments/runs/fidelidade/previsoes_teste.npz  y_score de cada configuração da Etapa 12
    results/tables/baselines.csv, fidelidade_resumo.csv  para conferir as estimativas

O teste é uma amostra (nov–dez/2025): com outra amostra do mesmo período, o
PR-AUC sairia um pouco diferente. O bootstrap estima essa variação: sorteia os
61 dias do teste com reposição, junta os voos dos dias sorteados e recalcula a
métrica; repete N_REPETICOES vezes. O IC de 95% vai do percentil 2,5 ao 97,5.

Por dias, não por voos: voos do mesmo dia não são independentes (um temporal
atrasa dezenas de voos ao mesmo tempo; a taxa de atraso diária no teste vai de
9% a 62%). Sortear voos daria um IC estreito demais.

Pareado: o mesmo sorteio de dias vale para todos os modelos e configurações.
Para dizer se A é melhor que B, o IC é o da diferença A − B, calculada em cada
repetição; se ele não contém zero, a diferença não é efeito da amostra de dias.
Comparar se os dois ICs se sobrepõem seria conservador demais.

Isolamento do teste (§4.9): não é um novo uso do teste. O script só reaproveita
as previsões das Etapas 6 e 12, de configurações já fixadas, e mede a
incerteza delas; nada é treinado nem escolhido a partir daqui.

Produz:
    results/tables/incerteza_modelos.csv           F1, PR-AUC e ROC-AUC de cada modelo, com IC
    results/tables/incerteza_modelos_pares.csv     diferença entre modelos, com IC
    results/tables/incerteza_fidelidade.csv        retenção do PR-AUC (fração de "todas"), com IC
    results/tables/incerteza_fidelidade_pares.csv  diferença de retenção entre métodos no
                                                   mesmo (modelo, k), com IC
Quem consome: o Cap. 4 (desempenho dos modelos e fidelidade) e o README.
"""

import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.compare import fidelity  # noqa: E402
from src.importance import comum  # noqa: E402
from src.models.baselines import DIR_DADOS, carregar  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

N_REPETICOES = 1000
SEMENTE = 42
PERCENTIS = [2.5, 97.5]  # IC de 95%
# Diferenças entre modelos que interessam ao texto: árvores entre si e contra a LogReg
PARES_MODELOS = [("random_forest", "xgboost"), ("random_forest", "logreg"),
                 ("xgboost", "logreg")]
METODOS_FIDELIDADE = ["intrinseca", "permutation", "shap", "gawll", "aleatorio"]
# Tolerância da conferência com fidelidade_resumo.csv: a tabela é arredondada em 4
# casas e as previsões foram salvas em float32
TOLERANCIA_FIDELIDADE = 1e-4


def dias_do_teste() -> np.ndarray:
    """Código do dia (0 a 60) de cada voo do teste, na ordem do test.parquet.

    Usada em: main.
    Importância: o dia é o bloco do bootstrap.
    """
    partida = pd.read_parquet(DIR_DADOS / "test.parquet", columns=["Partida Prevista"])
    codigos, _ = pd.factorize(partida["Partida Prevista"].dt.normalize(), sort=True)
    return codigos


def sortear_dias(n_dias: int, n_repeticoes: int, semente: int) -> np.ndarray:
    """Matriz (repetições × dias): quantas vezes cada dia entrou em cada reamostragem
    (n_dias sorteios com reposição por repetição).

    Usada em: main; tests/test_incerteza.py.
    Importância: um sorteio só, compartilhado por todas as métricas e configurações,
    é o que torna o bootstrap pareado.
    """
    rng = np.random.default_rng(semente)
    sorteios = rng.integers(0, n_dias, size=(n_repeticoes, n_dias))
    return np.stack([np.bincount(linha, minlength=n_dias) for linha in sorteios])


def metrica_ponderada(metrica: str, y: np.ndarray, previsao: np.ndarray,
                      pesos: np.ndarray) -> float:
    """F1 (previsao = classe), PR-AUC ou ROC-AUC (previsao = probabilidade), com o peso
    de cada voo = quantas vezes o dia dele foi sorteado. Voos de dias não sorteados
    (peso 0) ficam de fora.

    Usada em: avaliar_repeticoes; tests/test_incerteza.py.
    Importância: dá o mesmo valor que a métrica na reamostragem materializada (voos
    repetidos), sem copiar o teste a cada repetição.
    """
    usados = pesos > 0
    y, previsao, pesos = y[usados], previsao[usados], pesos[usados]
    if metrica == "f1":
        return f1_score(y, previsao, sample_weight=pesos)
    if metrica == "pr_auc":
        return average_precision_score(y, previsao, sample_weight=pesos)
    if metrica == "roc_auc":
        return roc_auc_score(y, previsao, sample_weight=pesos)
    raise ValueError(f"Métrica desconhecida: '{metrica}'")


def avaliar_repeticoes(metrica: str, y: np.ndarray, previsoes: np.ndarray,
                       codigos: np.ndarray, contagens: np.ndarray) -> np.ndarray:
    """Métrica de cada configuração (linha de previsoes) em cada repetição (linha de
    contagens); devolve (configurações × repetições).

    Usada em: bootstrap.
    Importância: é o laço do bootstrap; roda em paralelo, um bloco de repetições por
    processo.
    """
    resultado = np.empty((len(previsoes), len(contagens)))
    for b, contagem in enumerate(contagens):
        pesos = contagem[codigos]
        for c, previsao in enumerate(previsoes):
            resultado[c, b] = metrica_ponderada(metrica, y, previsao, pesos)
    return resultado


def bootstrap(metrica: str, y: np.ndarray, previsoes: np.ndarray, codigos: np.ndarray,
              contagens: np.ndarray) -> np.ndarray:
    """avaliar_repeticoes dividido em blocos de repetições, um por processo.

    Usada em: incerteza_modelos e incerteza_fidelidade; tests/test_incerteza.py.
    Importância: ~150 mil cálculos de métrica; em paralelo leva minutos.
    """
    blocos = np.array_split(contagens, min(len(contagens), 32))
    partes = Parallel(n_jobs=-1)(
        delayed(avaliar_repeticoes)(metrica, y, previsoes, codigos, bloco) for bloco in blocos)
    return np.concatenate(partes, axis=1)


def intervalo(repeticoes: np.ndarray) -> tuple[float, float]:
    """IC de 95% pelos percentis 2,5 e 97,5 das repetições.

    Usada em: incerteza_modelos, incerteza_fidelidade e comparar_pares.
    Importância: um critério único de IC no projeto.
    """
    inferior, superior = np.percentile(repeticoes, PERCENTIS)
    return float(inferior), float(superior)


def comparar_pares(pares: list[tuple], pontual: dict, repeticoes: dict) -> list[dict]:
    """Para cada par (a, b): diferença pontual a − b e o IC da diferença calculada em
    cada repetição (pareado). ic_contem_zero = True quer dizer que a diferença pode
    ser efeito da amostra de dias.

    Usada em: incerteza_modelos e incerteza_fidelidade.
    Importância: é o teste de "A é melhor que B" do texto.
    """
    linhas = []
    for a, b in pares:
        inferior, superior = intervalo(repeticoes[a] - repeticoes[b])
        linhas.append({"a": a, "b": b, "diferenca": pontual[a] - pontual[b],
                       "ic_inf": inferior, "ic_sup": superior,
                       "ic_contem_zero": inferior <= 0 <= superior})
    return linhas


def incerteza_modelos(y: np.ndarray, codigos: np.ndarray,
                      contagens: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """IC de F1, PR-AUC e ROC-AUC de cada modelo da Etapa 6 e das diferenças entre
    eles. Confere as estimativas pontuais com baselines.csv.

    Usada em: main.
    Importância: diz se as diferenças de desempenho entre os modelos são reais.
    """
    teste_x, _ = carregar("test")
    baselines = pd.read_csv(comum.DIR_TABELAS / "baselines.csv").set_index("modelo")
    classes, scores = [], []
    for nome in comum.MODELOS:
        modelo = comum.carregar_modelo(nome)
        classes.append(modelo.predict(teste_x))
        scores.append(modelo.predict_proba(teste_x)[:, 1])
    classes, scores = np.stack(classes), np.stack(scores)

    linhas, pares = [], []
    uns = np.ones(len(y))
    for metrica in ["f1", "pr_auc", "roc_auc"]:
        previsoes = classes if metrica == "f1" else scores
        repeticoes = dict(zip(comum.MODELOS,
                              bootstrap(metrica, y, previsoes, codigos, contagens)))
        pontual = {nome: metrica_ponderada(metrica, y, previsao, uns)
                   for nome, previsao in zip(comum.MODELOS, previsoes)}
        for nome in comum.MODELOS:
            if not np.isclose(pontual[nome], baselines.loc[nome, metrica], rtol=0, atol=1e-9):
                raise ValueError(f"{nome} {metrica}: {pontual[nome]} no teste, mas "
                                 f"{baselines.loc[nome, metrica]} em baselines.csv")
            linhas.append({"modelo": nome, "metrica": metrica, "valor": pontual[nome],
                           **dict(zip(["ic_inf", "ic_sup"], intervalo(repeticoes[nome])))})
        for par in comparar_pares(PARES_MODELOS, pontual, repeticoes):
            pares.append({"modelo_a": par.pop("a"), "modelo_b": par.pop("b"),
                          "metrica": metrica, **par})
        logger.info(f"Modelos, {metrica}: bootstrap concluído")
    return pd.DataFrame(linhas), pd.DataFrame(pares)


def carregar_previsoes_fidelidade(y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Lê as previsões salvas pela Etapa 12 e confere que estão na ordem do teste.
    Devolve a tabela de configurações (model, method, k, seed) e os scores
    (configurações × voos).

    Usada em: incerteza_fidelidade.
    Importância: liga cada previsão à sua configuração sem retreinar nada.
    """
    arquivo = np.load(fidelity.DIR_PREVISOES / "previsoes_teste.npz")
    if not (arquivo["y"] == y).all():
        raise ValueError("previsoes_teste.npz não está na ordem do test.parquet; "
                         "rode de novo a Etapa 12 (python src/compare/fidelity.py)")
    configuracoes = pd.DataFrame([i.split("|") for i in arquivo["ids"]],
                                 columns=["model", "method", "k", "seed"])
    configuracoes["k"] = configuracoes["k"].astype(int)
    return configuracoes, arquivo["scores"]


def incerteza_fidelidade(y: np.ndarray, codigos: np.ndarray,
                         contagens: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """IC da retenção do PR-AUC (PR-AUC médio do grupo / PR-AUC de "todas", na mesma
    repetição) de cada (modelo, método, k) e IC da diferença de retenção entre os
    métodos de mesmo (modelo, k), incluindo os sorteios aleatórios. Confere as
    estimativas pontuais com fidelidade_resumo.csv.

    Usada em: main.
    Importância: diz quais diferenças de fidelidade entre métodos são reais e se o
    top-k fica de fato abaixo do modelo completo.
    """
    configuracoes, scores = carregar_previsoes_fidelidade(y)
    repeticoes = bootstrap("pr_auc", y, scores, codigos, contagens)
    uns = np.ones(len(y))
    pontual = np.array([metrica_ponderada("pr_auc", y, s, uns) for s in scores])

    # Grupos como em fidelity.resumir: média das execuções de (modelo, método, k)
    grupos = configuracoes.groupby(["model", "method", "k"]).indices
    media_pontual = {g: pontual[i].mean() for g, i in grupos.items()}
    media_rep = {g: repeticoes[i].mean(axis=0) for g, i in grupos.items()}
    todas = {g[0]: g for g in grupos if g[1] == "todas"}  # modelo -> grupo "todas"
    retencao = {g: media_pontual[g] / media_pontual[todas[g[0]]] for g in grupos}
    retencao_rep = {g: media_rep[g] / media_rep[todas[g[0]]] for g in grupos}

    resumo = pd.read_csv(comum.DIR_TABELAS / "fidelidade_resumo.csv")
    resumo = resumo[resumo["conjunto"] == "test"].set_index(["model", "method", "k"])
    linhas = []
    for g, indices in grupos.items():
        if abs(media_pontual[g] - resumo.loc[g, "pr_auc"]) > TOLERANCIA_FIDELIDADE:
            raise ValueError(f"{g}: PR-AUC {media_pontual[g]:.5f} nas previsões salvas, "
                             f"{resumo.loc[g, 'pr_auc']} em fidelidade_resumo.csv")
        linhas.append({
            "model": g[0], "method": g[1], "k": g[2], "n_execucoes": len(indices),
            "pr_auc": media_pontual[g],
            **dict(zip(["pr_auc_ic_inf", "pr_auc_ic_sup"], intervalo(media_rep[g]))),
            "pr_auc_frac_todas": retencao[g],
            **dict(zip(["frac_ic_inf", "frac_ic_sup"], intervalo(retencao_rep[g]))),
        })

    # Pares no mesmo (modelo, k); "a" é sempre o de maior retenção pontual
    pares = []
    for modelo in comum.MODELOS:
        for k in fidelity.TOP_K:
            presentes = [(modelo, m, k) for m in METODOS_FIDELIDADE if (modelo, m, k) in grupos]
            ordenados = sorted(presentes, key=lambda g: -retencao[g])
            for par in comparar_pares(list(combinations(ordenados, 2)), retencao, retencao_rep):
                pares.append({"model": modelo, "k": k, "method_a": par.pop("a")[1],
                              "method_b": par.pop("b")[1], **par})
    logger.info(f"Fidelidade: {len(scores)} configurações, bootstrap concluído")
    return pd.DataFrame(linhas), pd.DataFrame(pares)


def main() -> None:
    """Etapa 12b: sorteia os dias uma vez, roda o bootstrap dos modelos e da
    fidelidade e salva as quatro tabelas de incerteza.

    Usada em: python src/compare/incerteza.py; main.py, etapa "incerteza".
    Importância: separa diferenças reais de efeito da amostra de dias do teste.
    """
    _, y_teste = carregar("test")
    y = y_teste.to_numpy()
    codigos = dias_do_teste()
    contagens = sortear_dias(codigos.max() + 1, N_REPETICOES, SEMENTE)
    logger.info(f"Teste: {len(y)} voos em {codigos.max() + 1} dias; "
                f"{N_REPETICOES} reamostragens de dias")

    modelos, modelos_pares = incerteza_modelos(y, codigos, contagens)
    fidelidade, fidelidade_pares = incerteza_fidelidade(y, codigos, contagens)
    comum.salvar_tabela(modelos.round(4), "incerteza_modelos.csv")
    comum.salvar_tabela(modelos_pares.round(4), "incerteza_modelos_pares.csv")
    comum.salvar_tabela(fidelidade.round(4), "incerteza_fidelidade.csv")
    comum.salvar_tabela(fidelidade_pares.round(4), "incerteza_fidelidade_pares.csv")

    with pd.option_context("display.width", 220, "display.max_rows", 200):
        logger.info("Modelos:\n" + modelos.round(3).to_string(index=False))
        logger.info("Diferenças entre modelos:\n" + modelos_pares.round(3).to_string(index=False))
        cinco = fidelidade_pares[fidelidade_pares["k"] == 5]
        logger.info("Fidelidade, top-5, diferenças de retenção:\n"
                    + cinco.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
