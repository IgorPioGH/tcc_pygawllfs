"""Exemplos trabalhados: como os métodos baseados em desempenho nascem de matrizes de
confusão.

A Permutation, o fitness do PyGAwLLfs e a fidelidade são todos diferenças de
acurácia, ou seja, de diagonais de matrizes de confusão. Dois exemplos com números
reais do projeto:

1. Permutation, no Random Forest e na validação: a matriz de confusão original e a
   matriz depois de embaralhar o atributo mais importante (um embaralhamento
   ilustrativo), com a queda de acurácia. Em seguida, calcular_permutation com 10
   repetições, que tem de reproduzir o valor de importancia_permutation.csv.
2. Linkage learning do PyGAwLLfs, com o melhor subconjunto x* da LogReg na semente
   42 e o par hora × periodo_dia (redundante por construção): as quatro máscaras
   x, x⊕g, x⊕h e x⊕gh, as quatro matrizes de confusão na validação da
   subamostra, as acurácias, os fitness (0,98 · acurácia + 0,02 · parcimônia) e
   δ_g, δ_h, δ_g(x⊕h) e ω = |f(x⊕gh) − f(x⊕g) − f(x⊕h) + f(x)|. Se ω ≈ 0 nesse
   x*, usa o par de maior v_int da mesma execução.

Produz results/tables/exemplo_{permutation,linkage}.csv e
results/figures/exemplo_{permutation,linkage}.{png,pdf}. Lê só a validação e a
subamostra; nunca o teste.
"""

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # só gera arquivo, sem abrir janela
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))
sys.path.append(str(RAIZ / "PyGAwLLfs"))

from src.avaliacao.matriz_confusao import CLASSES, metricas_da_matriz, milhar  # noqa: E402
from src.graficos import (  # noqa: E402
    COR_TEXTO,
    COR_TEXTO_SECUNDARIO,
    ESCALA_RBO,
    LARGURA_CHEIA,
    cor_do_texto,
    decimal,
    salvar,
)
from src.modelagem import comum, gawll  # noqa: E402
from src.modelagem.baselines import SEMENTE, carregar, criar_modelo, ler_configuracoes_escolhidas  # noqa: E402
from src.modelagem.consolidar_gawll import DIR_EXECUCOES  # noqa: E402
from src.modelagem.permutation import calcular_permutation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DIR_FIGURAS = RAIZ / "results" / "figures"
TOLERANCIA = 1e-9
MODELO_PERMUTATION = "random_forest"
MODELO_LINKAGE = "logreg"
PAR_LINKAGE = ("hora", "periodo_dia")
OMEGA_MINIMO = 1e-6


def desenhar_matriz(ax, matriz: np.ndarray, titulo: str) -> None:
    """Matriz de confusão 2×2 com a contagem em cada célula e a cor pela fração da
    classe real.

    Usada em: figura_permutation e figura_linkage.
    Importância: o mesmo desenho da matriz de confusão do desempenho, para a leitura
    ser a mesma.
    """
    fracao = matriz / matriz.sum(axis=1, keepdims=True)
    ax.pcolormesh(fracao, cmap=ESCALA_RBO, vmin=0, vmax=1, edgecolors="white", linewidth=2)
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.5, milhar(matriz[i, j]), ha="center", va="center",
                    fontsize=8, color=cor_do_texto(ESCALA_RBO(fracao[i, j])))
    ax.set_xticks([0.5, 1.5], CLASSES)
    ax.set_yticks([0.5, 1.5], CLASSES)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.tick_params(length=0, colors=COR_TEXTO_SECUNDARIO)
    for lado in ax.spines.values():
        lado.set_visible(False)
    ax.set_title(titulo, color=COR_TEXTO)


# ---------------------------------------------------------------------------
# 1. Permutation
# ---------------------------------------------------------------------------

def exemplo_permutation() -> tuple[pd.DataFrame, dict]:
    """Matriz original e matriz com o atributo mais importante embaralhado (RF,
    validação), e a Permutation de 10 repetições desse atributo, conferida com
    importancia_permutation.csv.

    Usada em: main.
    Importância: mostra que a importância por permutação é uma queda de acurácia, ou
    seja, massa saindo da diagonal da matriz de confusão.
    """
    x_val, y_val = carregar("val")
    modelo = comum.carregar_modelo(MODELO_PERMUTATION)
    tabela = pd.read_csv(comum.DIR_TABELAS / "importancia_permutation.csv")
    tabela = tabela[(tabela["model"] == MODELO_PERMUTATION) & (tabela["seed"] == SEMENTE)]
    atributo = tabela.nlargest(1, "importance")["feature"].iloc[0]

    embaralhado = x_val.copy()
    embaralhado[atributo] = np.random.default_rng(SEMENTE).permutation(embaralhado[atributo].to_numpy())
    matrizes = {"original": confusion_matrix(y_val, modelo.predict(x_val), labels=[0, 1]),
                "embaralhado": confusion_matrix(y_val, modelo.predict(embaralhado), labels=[0, 1])}

    valores = dict(zip(x_val.columns, calcular_permutation(modelo, x_val, y_val, SEMENTE)))
    esperado = tabela.set_index("feature").loc[atributo, "importance"]
    if abs(valores[atributo] - esperado) > TOLERANCIA:
        raise ValueError(f"Permutation de {atributo}: {valores[atributo]} ≠ CSV {esperado}")

    linhas = [{"matriz": nome, "atributo": atributo, "vn": m[0, 0], "fp": m[0, 1], "fn": m[1, 0],
               "vp": m[1, 1], "acuracia": metricas_da_matriz(m)["acuracia"]}
              for nome, m in matrizes.items()]
    resumo = {"atributo": atributo, "permutation_10": valores[atributo], "matrizes": matrizes}
    return pd.DataFrame(linhas), resumo


def figura_permutation(resumo: dict, caminho_base: Path) -> None:
    """As duas matrizes lado a lado, com a acurácia de cada uma e a queda.

    Usada em: main.
    Importância: é o exemplo do texto para a Permutation.
    """
    fig, eixos = plt.subplots(1, 2, figsize=(LARGURA_CHEIA * 0.75, 2.6), layout="constrained")
    acuracias = {}
    for ax, (nome, matriz) in zip(eixos, resumo["matrizes"].items()):
        acuracias[nome] = metricas_da_matriz(matriz)["acuracia"]
        titulo = "original" if nome == "original" else f"{resumo['atributo']} embaralhado"
        desenhar_matriz(ax, matriz, f"{titulo}\nacurácia {decimal(acuracias[nome], 3)}")
        ax.set_xlabel("previsto", color=COR_TEXTO_SECUNDARIO)
    eixos[0].set_ylabel("real", color=COR_TEXTO_SECUNDARIO)
    queda = acuracias["original"] - acuracias["embaralhado"]
    salvar(fig, caminho_base, legenda=(
        f"Permutation no Random Forest, na validação: matriz de confusão original e depois de "
        f"embaralhar uma vez o atributo mais importante ({resumo['atributo']}). A acurácia cai "
        f"{decimal(queda, 3)}: os acertos saem da diagonal. A importância do método é a média "
        f"dessa queda em 10 embaralhamentos ({decimal(resumo['permutation_10'], 4)})."))


# ---------------------------------------------------------------------------
# 2. Linkage learning
# ---------------------------------------------------------------------------

def avaliar_mascara(mascara: np.ndarray, parametros: dict, dados: tuple) -> tuple[np.ndarray, float]:
    """Treina o modelo com as colunas da máscara na subamostra e devolve a matriz de
    confusão na validação da subamostra e o fitness (0,98 · acurácia + 0,02 ·
    parcimônia). Confere a acurácia com gawll.treinar_e_medir.

    Usada em: exemplo_linkage.
    Importância: é exatamente a conta que o PyGAwLLfs faz a cada indivíduo.
    """
    x_treino, y_treino, x_val, y_val, _ = dados
    modelo = criar_modelo(MODELO_LINKAGE, parametros, n_jobs=1)
    modelo.fit(x_treino[:, mascara], y_treino)
    matriz = confusion_matrix(y_val, modelo.predict(x_val[:, mascara]), labels=[0, 1])
    acuracia = metricas_da_matriz(matriz)["acuracia"]
    if abs(acuracia - gawll.treinar_e_medir(MODELO_LINKAGE, parametros, mascara, *dados[:4])) > TOLERANCIA:
        raise ValueError("A acurácia da matriz difere da do fitness do PyGAwLLfs")
    return matriz, 0.98 * acuracia + 0.02 * (1 - mascara.mean())


def exemplo_linkage() -> tuple[pd.DataFrame, dict]:
    """Uma sondagem do linkage learning feita à mão: as quatro máscaras a partir do x*
    da LogReg (semente 42), as quatro matrizes, os fitness, δ e ω.

    Usada em: main.
    Importância: mostra mecanicamente por que redundância vira interação: desligar uma
    das duas features custa pouco, desligar as duas custa muito.
    """
    execucao = np.load(DIR_EXECUCOES / f"{MODELO_LINKAGE}_seed{SEMENTE}.npz")
    features = [str(f) for f in execucao["features"]]
    x = execucao["cromossomo"].astype(bool)
    dados = gawll.carregar_subamostra()
    parametros = ler_configuracoes_escolhidas()[MODELO_LINKAGE]

    def sondar(par: tuple[str, str]) -> dict:
        """As quatro máscaras de um par e seus fitness, δ e ω.

        Usada em: exemplo_linkage.
        Importância: é a mutation_ll do PyGAwLLfs, com g e h escolhidos.
        """
        g, h = (features.index(f) for f in par)
        mascaras = {"x": x.copy(), "x⊕g": x.copy(), "x⊕h": x.copy(), "x⊕gh": x.copy()}
        mascaras["x⊕g"][g] ^= True
        mascaras["x⊕h"][h] ^= True
        mascaras["x⊕gh"][[g, h]] ^= True
        avaliadas = {nome: avaliar_mascara(m, parametros, dados) for nome, m in mascaras.items()}
        f = {nome: fitness for nome, (_, fitness) in avaliadas.items()}
        omega = abs(f["x⊕gh"] - f["x⊕g"] - f["x⊕h"] + f["x"])
        if abs(omega - abs((f["x⊕gh"] - f["x⊕h"]) - (f["x⊕g"] - f["x"]))) > TOLERANCIA:
            raise ValueError("ω não confere com a diferença dos efeitos condicionais")
        return {"par": par, "mascaras": mascaras, "avaliadas": avaliadas, "fitness": f,
                "delta_g": abs(f["x⊕g"] - f["x"]), "delta_h": abs(f["x⊕h"] - f["x"]),
                "delta_g_dado_h": abs(f["x⊕gh"] - f["x⊕h"]), "omega": omega}

    resultado = sondar(PAR_LINKAGE)
    if resultado["omega"] < OMEGA_MINIMO:
        vint = execucao["vint"]
        i, j = np.unravel_index(np.argmax(np.triu(vint, k=1)), vint.shape)
        logger.info(f"ω ≈ 0 em {PAR_LINKAGE}; usando o par de maior v_int: {features[i]} × {features[j]}")
        resultado = sondar((features[i], features[j]))

    linhas = []
    for nome, (matriz, fitness) in resultado["avaliadas"].items():
        linhas.append({"mascara": nome, "n_features": int(resultado["mascaras"][nome].sum()),
                       "vn": matriz[0, 0], "fp": matriz[0, 1], "fn": matriz[1, 0], "vp": matriz[1, 1],
                       "acuracia": metricas_da_matriz(matriz)["acuracia"], "fitness": fitness})
    tabela = pd.DataFrame(linhas).assign(
        feature_g=resultado["par"][0], feature_h=resultado["par"][1],
        delta_g=resultado["delta_g"], delta_h=resultado["delta_h"],
        delta_g_dado_h=resultado["delta_g_dado_h"], omega=resultado["omega"])
    return tabela, resultado


def figura_linkage(resultado: dict, caminho_base: Path) -> None:
    """As quatro matrizes (x, x⊕g, x⊕h, x⊕gh) com acurácia e fitness, numa linha.

    Usada em: main.
    Importância: é o exemplo do texto para δ e ω.
    """
    g, h = resultado["par"]
    fig, eixos = plt.subplots(1, 4, figsize=(LARGURA_CHEIA, 2.4), layout="constrained")
    for ax, (nome, (matriz, fitness)) in zip(eixos, resultado["avaliadas"].items()):
        acuracia = metricas_da_matriz(matriz)["acuracia"]
        desenhar_matriz(ax, matriz, f"{nome}\nacc {decimal(acuracia, 4)}\nf {decimal(fitness, 4)}")
        ax.set_xlabel("previsto", color=COR_TEXTO_SECUNDARIO)
    eixos[0].set_ylabel("real", color=COR_TEXTO_SECUNDARIO)
    salvar(fig, caminho_base, legenda=(
        f"Uma sondagem do linkage learning do PyGAwLLfs, com o melhor subconjunto x* da "
        f"regressão logística (semente 42), g = {g} e h = {h}: matrizes de confusão na "
        "validação da subamostra, acurácia (acc) e fitness (f = 0,98 · acc + 0,02 · "
        f"parcimônia) das quatro máscaras. δ_g = {decimal(resultado['delta_g'], 4)}, "
        f"δ_h = {decimal(resultado['delta_h'], 4)}, δ_g(x⊕h) = "
        f"{decimal(resultado['delta_g_dado_h'], 4)} e ω = |f(x⊕gh) − f(x⊕g) − f(x⊕h) + f(x)| = "
        f"{decimal(resultado['omega'], 4)}."))


def main() -> None:
    """Gera os dois exemplos, com as tabelas e as figuras.

    Usada em: python src/interpretacao/exemplos.py; main.py, etapa "figuras".
    Importância: mostra como cada número dos métodos baseados em desempenho é obtido.
    """
    tabela_perm, resumo_perm = exemplo_permutation()
    tabela_ll, resultado_ll = exemplo_linkage()
    comum.salvar_tabela(tabela_perm.round(6), "exemplo_permutation.csv")
    comum.salvar_tabela(tabela_ll.round(6), "exemplo_linkage.csv")
    figura_permutation(resumo_perm, DIR_FIGURAS / "exemplo_permutation")
    figura_linkage(resultado_ll, DIR_FIGURAS / "exemplo_linkage")
    logger.info("Permutation:\n" + tabela_perm.round(4).to_string(index=False))
    logger.info("Linkage learning:\n" + tabela_ll.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
