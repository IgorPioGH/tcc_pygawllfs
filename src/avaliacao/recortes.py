"""Recortes temporais: as importâncias se mantêm de 2024 para 2025?

Os dois recortes usam o mesmo calendário (treino jan–ago, validação set–out,
features.RECORTES) e não compartilham nenhum voo, então a diferença entre eles
reflete a mudança de um ano para o outro, e não a sazonalidade. Para cada
recorte, a preparação é a mesma do experimento principal (features.montar_conjuntos,
já 50/50) e os modelos usam os hiperparâmetros escolhidos no principal
(baselines.csv), sem novo ajuste. Em cada recorte, na validação (o teste do
recorte nunca é lido):
    intrínseca   importância embutida em cada modelo
    permutation  10 repetições, semente 42
    shap         TreeSHAP em 10 mil linhas (RF e XGBoost)
    gawll        uma execução por modelo (semente 42), na subamostra do recorte,
                 com checkpoint em experiments/runs/recortes/<recorte>/

Comparação: para cada (método, modelo), Spearman e top-5/10 entre o ranking de
2024 e o de 2025 (rankings.comparar), ao lado do Spearman entre sementes do
experimento principal (estabilidade.csv). No PyGAwLLfs cada recorte tem uma
semente só, então a variação entre recortes mistura período e semente: se o
Spearman entre recortes for parecido com o entre sementes, o período não
acrescenta instabilidade.

Produz:
    results/tables/importancia_recortes.csv    formato padrão + coluna "recorte"
    results/tables/recortes_desempenho.csv     acurácia e ROC-AUC na validação de cada recorte
    results/tables/recortes_concordancia.csv   2024 × 2025 por (método, modelo)
    results/figures/recortes.{png,pdf}
    custo_importancia.csv                       métodos com sufixo "_recorte"
"""

import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # só gera arquivo, sem abrir janela
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))
sys.path.append(str(RAIZ / "PyGAwLLfs"))

from src.avaliacao.rankings import METODOS, MODELOS, comparar, pontuacao  # noqa: E402
from src.graficos import (  # noqa: E402
    COR_GRADE,
    COR_TEXTO,
    COR_TEXTO_SECUNDARIO,
    ROTULO_METODO,
    ROTULO_MODELO,
    salvar,
)
from src.modelagem import comum, gawll  # noqa: E402
from src.modelagem.baselines import (  # noqa: E402
    NAO_ATRIBUTOS,
    SEMENTE,
    avaliar_modelo,
    criar_modelo,
    ler_configuracoes_escolhidas,
)
from src.modelagem.consolidar_gawll import consolidar  # noqa: E402
from src.modelagem.intrinsic import importancia_intrinseca  # noqa: E402
from src.modelagem.permutation import calcular_permutation  # noqa: E402
from src.modelagem.shap_global import MODELOS_ARVORE, TAMANHO_AMOSTRA, calcular_shap  # noqa: E402
from src.preparacao.clean import assert_sem_vazamento  # noqa: E402
from src.preparacao.features import (  # noqa: E402
    criar_subamostra,
    features_sem_alvo,
    montar_conjuntos,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARQUIVO_TARGET = RAIZ / "data" / "interim" / "vra_target.parquet"
DIR_EXECUCOES = RAIZ / "experiments" / "runs" / "recortes"
DIR_FIGURAS = RAIZ / "results" / "figures"
RECORTES_COMPARADOS = ["2024", "2025"]
COR_METODO = {"intrinseca": "#eda100", "permutation": "#2a78d6", "shap": "#eb6834",
              "gawll": "#1baf7a"}


def separar(conjunto: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """(x, y) de um conjunto montado por features.montar_conjuntos, com as mesmas colunas
    de baselines.carregar().

    Usada em: importancias_do_recorte.
    Importância: os modelos dos recortes veem exatamente as 29 features do principal.
    """
    x = conjunto.drop(columns=NAO_ATRIBUTOS)
    assert_sem_vazamento(x)
    return x, conjunto["atrasado"]


def importancias_do_recorte(recorte: str, conjuntos: dict, configuracoes: dict
                            ) -> tuple[list[pd.DataFrame], list[dict], list[dict]]:
    """Treina os três modelos no treino do recorte e calcula as quatro importâncias na
    validação do recorte. Devolve (tabelas de importância, desempenho, custos).

    Usada em: main.
    Importância: aplica o mesmo protocolo do principal a um ano só.
    """
    x_treino, y_treino = separar(conjuntos["train"])
    x_val, y_val = separar(conjuntos["val"])
    tabelas, desempenho, custos = [], [], []

    def registrar(tabela: pd.DataFrame, metodo: str, nome: str, semente: int, inicio: float):
        """Guarda a tabela e o custo de um (método, modelo) do recorte.

        Usada em: importancias_do_recorte.
        Importância: evita repetir o mesmo bloco para cada método.
        """
        tabelas.append(tabela)
        custos.append({"method": f"{metodo}_recorte", "model": nome, "seed": semente,
                       "tempo_s": time.perf_counter() - inicio, "n_treinos": 0})

    for nome in MODELOS:
        modelo = criar_modelo(nome, configuracoes[nome])
        modelo.fit(x_treino, y_treino)
        metricas = avaliar_modelo(modelo, x_val, y_val)
        desempenho.append({"recorte": recorte, "model": nome, "n_treino": len(x_treino),
                           "n_validacao": len(x_val), "acuracia": metricas["acuracia"],
                           "roc_auc": metricas["roc_auc"]})

        inicio = time.perf_counter()
        features, valores = importancia_intrinseca(nome, modelo)
        registrar(comum.tabela_importancia(features, valores, "intrinseca", nome, SEMENTE,
                                           rank_por_absoluto=(nome == "logreg")),
                  "intrinseca", nome, SEMENTE, inicio)

        inicio = time.perf_counter()
        valores = calcular_permutation(modelo, x_val, y_val, SEMENTE)
        registrar(comum.tabela_importancia(list(x_val.columns), valores, "permutation", nome,
                                           SEMENTE), "permutation", nome, SEMENTE, inicio)

        if nome in MODELOS_ARVORE:
            inicio = time.perf_counter()
            amostra = x_val.sample(n=min(TAMANHO_AMOSTRA, len(x_val)), random_state=SEMENTE)
            registrar(comum.tabela_importancia(list(x_val.columns), calcular_shap(modelo, amostra),
                                               "shap", nome, SEMENTE), "shap", nome, SEMENTE, inicio)
        logger.info(f"{recorte} {nome}: acurácia na validação {metricas['acuracia']:.3f}")

    # PyGAwLLfs: subamostra do recorte, uma semente, retomada pelo checkpoint
    dados = gawll.separar_subamostra(criar_subamostra(conjuntos["train"], conjuntos["val"]))
    dir_recorte = DIR_EXECUCOES / recorte
    dir_recorte.mkdir(parents=True, exist_ok=True)
    for nome in MODELOS:
        caminho = dir_recorte / f"{nome}_seed{SEMENTE}.npz"
        if caminho.exists():
            logger.info(f"Já existe, pulando: {recorte}/{caminho.name}")
            continue
        gawll.executar(nome, configuracoes[nome], SEMENTE, dados, caminho,
                       geracoes=gawll.GERACOES)
    importancia_ga, _, _, custos_ga = consolidar(dir_recorte)
    tabelas.append(importancia_ga)
    custos.extend({**custo, "method": "gawll_recorte"} for custo in custos_ga)

    tabelas = [tabela.assign(recorte=recorte) for tabela in tabelas]
    return tabelas, desempenho, custos


def concordancia_entre_recortes(importancias: pd.DataFrame,
                                estabilidade: pd.DataFrame) -> pd.DataFrame:
    """Para cada (método, modelo): Spearman e top-5/10 entre o ranking de 2024 e o de
    2025, e o Spearman entre sementes do experimento principal (vazio na intrínseca,
    que é determinística).

    Usada em: main; tests/test_recortes.py.
    Importância: é a medida de estabilidade entre recortes temporais, lida contra a
    estabilidade entre sementes.
    """
    entre_sementes = estabilidade.set_index(["method", "model"])["spearman"]
    linhas = []
    for (metodo, modelo), grupo in importancias.groupby(["method", "model"]):
        a, b = (grupo[grupo["recorte"] == r] for r in RECORTES_COMPARADOS)
        resultado = comparar(pontuacao(a).set_axis(a["feature"]),
                             pontuacao(b).set_axis(b["feature"]))
        linhas.append({"method": metodo, "model": modelo,
                       "spearman": resultado["spearman"], "top5": resultado["top5"],
                       "top10": resultado["top10"],
                       "spearman_entre_sementes": entre_sementes.get((metodo, modelo))})
    ordem = {(m, t): i for i, (t, m) in enumerate((t, m) for t in MODELOS for m in METODOS)}
    tabela = pd.DataFrame(linhas)
    tabela["ordem"] = [ordem[(m, t)] for m, t in zip(tabela["method"], tabela["model"])]
    return tabela.sort_values("ordem").drop(columns="ordem").reset_index(drop=True)


def figura_recortes(concordancia: pd.DataFrame, caminho_base: Path) -> None:
    """Spearman 2024 × 2025 de cada (método, modelo), com um marcador vazado no Spearman
    entre sementes do experimento principal.

    Usada em: main.
    Importância: é a figura da estabilidade entre recortes temporais.
    """
    y = list(range(len(concordancia)))[::-1]
    fig, ax = plt.subplots(figsize=(6.3, 3.6), layout="constrained")
    for posicao, linha in zip(y, concordancia.itertuples()):
        cor = COR_METODO[linha.method]
        ax.scatter(linha.spearman, posicao, s=45, color=cor, zorder=3)
        if pd.notna(linha.spearman_entre_sementes):
            ax.scatter(linha.spearman_entre_sementes, posicao, s=45, facecolors="white",
                       edgecolors=cor, linewidths=1.5, zorder=3)
    ax.set_yticks(y, [f"{ROTULO_METODO[m]} · {ROTULO_MODELO[t]}"
                      for m, t in zip(concordancia["method"], concordancia["model"])])
    ax.set_xlim(min(0, concordancia["spearman"].min() - 0.05), 1.02)
    ax.set_xlabel("Spearman", color=COR_TEXTO_SECUNDARIO)
    ax.grid(axis="x", color=COR_GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ["top", "right", "left"]:
        ax.spines[lado].set_visible(False)
    ax.tick_params(length=0, colors=COR_TEXTO_SECUNDARIO, labelsize=8)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
    ax.scatter([], [], s=45, color=COR_TEXTO, label="2024 × 2025")
    ax.scatter([], [], s=45, facecolors="white", edgecolors=COR_TEXTO, linewidths=1.5,
               label="entre sementes (principal)")
    ax.legend(frameon=False, loc="lower left")
    salvar(fig, caminho_base, legenda=(
        "Estabilidade entre recortes temporais: correlação de Spearman entre o ranking de "
        "importância de 2024 e o de 2025 (mesmo calendário: treino jan–ago, validação set–out), "
        "por método e modelo. O marcador vazado é o Spearman entre sementes do experimento "
        "principal; a intrínseca é determinística e não tem esse marcador."))


def main() -> None:
    """Monta os dois recortes, calcula as importâncias de cada um, compara 2024 × 2025 e
    salva as tabelas, o custo e a figura.

    Usada em: python src/avaliacao/recortes.py; main.py, etapa "recortes";
    tests/test_recortes.py.
    Importância: responde ao objetivo "estabilidade das estimativas em diferentes
    recortes temporais".
    """
    df = features_sem_alvo(pd.read_parquet(ARQUIVO_TARGET))
    configuracoes = ler_configuracoes_escolhidas()
    tabelas, desempenho, custos = [], [], []
    for recorte in RECORTES_COMPARADOS:
        conjuntos = montar_conjuntos(df, recorte)
        tabelas_r, desempenho_r, custos_r = importancias_do_recorte(recorte, conjuntos, configuracoes)
        tabelas += tabelas_r
        desempenho += desempenho_r
        custos += custos_r

    importancias = pd.concat(tabelas, ignore_index=True)[comum.COLUNAS_PADRAO + ["recorte"]]
    concordancia = concordancia_entre_recortes(
        importancias, pd.read_csv(comum.DIR_TABELAS / "estabilidade.csv"))
    comum.salvar_tabela(importancias, "importancia_recortes.csv")
    comum.salvar_tabela(pd.DataFrame(desempenho).round(4), "recortes_desempenho.csv")
    comum.salvar_tabela(concordancia.round(4), "recortes_concordancia.csv")
    comum.salvar_custo(custos)  # uma chamada só: salvar_custo substitui por método
    figura_recortes(concordancia, DIR_FIGURAS / "recortes")

    logger.info("Desempenho na validação:\n" + pd.DataFrame(desempenho).round(3).to_string(index=False))
    logger.info("2024 × 2025:\n" + concordancia.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
