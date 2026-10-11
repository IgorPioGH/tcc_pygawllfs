"""Etapa 6 — modelos de referência.

Recebe os três conjuntos congelados de data/processed/, treina um baseline
trivial, Regressão Logística, Random Forest e XGBoost, escolhe a melhor
configuração de cada modelo na VALIDAÇÃO e avalia a escolhida UMA única vez
no TESTE.

Saídas:
    results/tables/baselines_tuning.csv     todas as configurações, na validação
    results/tables/baselines.csv            melhor de cada modelo, no teste
    results/tables/coeficientes_logreg.csv  importância intrínseca da LogReg
    experiments/runs/baselines/<modelo>.joblib

Também é a fonte única dos modelos do PyGAwLLfs (src/importance/gawll.py):
criar_modelo() monta o estimador e ler_configuracoes_escolhidas() devolve os
hiperparâmetros escolhidos aqui, para que o GA explique o mesmo modelo (§4.11).
"""

import ast
import logging
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Raiz do projeto: baselines.py -> models/ -> src/ -> raiz
RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.data.clean import assert_sem_vazamento
from src.models.evaluate import avaliar

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DIR_DADOS = RAIZ / "data" / "processed"
DIR_TABELAS = RAIZ / "results" / "tables"
DIR_MODELOS = RAIZ / "experiments" / "runs" / "baselines"

SEMENTE = 42
NAO_ATRIBUTOS = ["Partida Prevista", "Chegada Prevista", "atrasado", "atraso_min"]

# Métrica que escolhe a melhor configuração na validação. F1, por coerência
# com o fitness corrigido do PyGAwLLfs (docs/metodologia.md §4.2).
METRICA_SELECAO = "f1"

# Gatilhos de vazamento (docs/metodologia.md §9), checados na validação ANTES do teste.
LIMITE_ROC_AUC = 0.90
LIMITE_F1 = 0.85
LIMITE_PR_AUC_RAZAO = 3.0  # PR-AUC acima de 3x a taxa base de atraso

# Grade de hiperparâmetros: para cada modelo, a lista de configurações testadas.
# O RF é fixo, igual ao RF dentro do PyGAwLLfs (n_estimators=100,
# max_depth=10), para que a comparação de importâncias do §4.11 seja feita
# sobre o mesmo modelo. No tuning, max_depth=12 ganhou por só 0,006 de F1.
GRADE = {
    "dummy": [{"strategy": "most_frequent"}],
    "logreg": [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
    "random_forest": [{"max_depth": 10}],
    "xgboost": [{"max_depth": 4}, {"max_depth": 6}, {"max_depth": 8}],
}


def carregar(nome: str) -> tuple[pd.DataFrame, pd.Series]:
    """Lê um conjunto congelado e separa atributos (X) do alvo (y)."""
    df = pd.read_parquet(DIR_DADOS / f"{nome}.parquet")
    x = df.drop(columns=NAO_ATRIBUTOS)
    y = df["atrasado"]
    assert_sem_vazamento(x)
    return x, y


def criar_modelo(nome: str, parametros: dict, peso_positivo: float, n_jobs: int = -1):
    """Monta um estimador novo (não treinado) a partir do nome e dos parâmetros.

    peso_positivo = negativos / positivos no treino; é o equivalente do
    class_weight="balanced" no XGBoost.
    n_jobs: o PyGAwLLfs passa 1, porque já paraleliza entre os indivíduos.
    """
    if nome == "dummy":
        return DummyClassifier(**parametros)
    if nome == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                **parametros,
                class_weight="balanced",
                max_iter=1000,
                random_state=SEMENTE,
            ),
        )
    if nome == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            **parametros,
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=SEMENTE,
        )
    if nome == "xgboost":
        return XGBClassifier(
            n_estimators=300,
            **parametros,
            learning_rate=0.1,
            scale_pos_weight=peso_positivo,
            tree_method="hist",
            device="cuda",
            random_state=SEMENTE,
        )
    raise ValueError(f"Modelo desconhecido: '{nome}'")


def montar_configuracoes(y_treino: pd.Series) -> list[tuple[str, dict, object]]:
    """Uma tupla (nome do modelo, parâmetros, estimador) por item da GRADE."""
    peso_positivo = (y_treino == 0).sum() / (y_treino == 1).sum()
    configs = []
    for nome, lista_parametros in GRADE.items():
        for parametros in lista_parametros:
            configs.append(
                (nome, parametros, criar_modelo(nome, parametros, peso_positivo))
            )
    return configs


def ler_configuracoes_escolhidas() -> dict[str, dict]:
    """Lê baselines.csv e devolve {modelo: parâmetros escolhidos na validação}.

    A coluna 'parametros' foi salva como texto ("{'C': 1.0}");
    ast.literal_eval converte de volta para dicionário sem executar código.
    """
    tabela = pd.read_csv(DIR_TABELAS / "baselines.csv")
    return {
        linha["modelo"]: ast.literal_eval(linha["parametros"])
        for _, linha in tabela.iterrows()
    }


def avaliar_modelo(modelo, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Gera previsões de classe e de probabilidade e passa para avaliar()."""
    return avaliar(y, modelo.predict(x), modelo.predict_proba(x)[:, 1])


def auditar_vazamento(nome: str, metricas: dict[str, float], taxa_base: float) -> None:
    """Interrompe a execução se alguma métrica for boa demais (docs/metodologia.md §9)."""
    gatilhos = []
    if metricas["roc_auc"] > LIMITE_ROC_AUC:
        gatilhos.append(f"roc_auc={metricas['roc_auc']:.3f}")
    if metricas["f1"] > LIMITE_F1:
        gatilhos.append(f"f1={metricas['f1']:.3f}")
    if metricas["pr_auc"] > LIMITE_PR_AUC_RAZAO * taxa_base:
        gatilhos.append(f"pr_auc={metricas['pr_auc']:.3f} (taxa base {taxa_base:.3f})")
    if gatilhos:
        raise ValueError(
            f"Possível vazamento em '{nome}': {gatilhos}. "
            "Audite as features contra o §4.3 antes de tocar o teste."
        )


def salvar_coeficientes_logreg(pipeline, colunas: pd.Index, caminho: Path) -> None:
    """Salva os coeficientes padronizados no formato padrão de importância (§7).

    O sinal é mantido (§4.8); o rank é feito pelo valor absoluto.
    """
    coeficientes = pipeline.named_steps["logisticregression"].coef_[0]
    tabela = pd.DataFrame({"feature": colunas, "importance": coeficientes})
    tabela["rank"] = (
        tabela["importance"].abs().rank(ascending=False, method="min").astype(int)
    )
    tabela["method"] = "coeficiente_padronizado"
    tabela["model"] = "logreg"
    tabela["seed"] = SEMENTE
    tabela.sort_values("rank").to_csv(caminho, index=False)


def main() -> None:
    treino_x, treino_y = carregar("train")
    val_x, val_y = carregar("val")
    teste_x, teste_y = carregar("test")
    logger.info(
        f"treino {treino_x.shape} | validação {val_x.shape} | teste {teste_x.shape}"
    )

    # 1. Tuning: toda configuração treina no treino e é medida na validação
    resultados = []
    melhores = {}
    for nome, parametros, modelo in montar_configuracoes(treino_y):
        inicio = time.perf_counter()
        modelo.fit(treino_x, treino_y)
        tempo = time.perf_counter() - inicio

        metricas = avaliar_modelo(modelo, val_x, val_y)
        resultados.append(
            {
                "modelo": nome,
                "parametros": str(parametros),
                "tempo_treino_s": round(tempo, 1),
                **metricas,
            }
        )
        logger.info(
            f"{nome} {parametros} | {tempo:.1f}s | f1={metricas['f1']:.3f} "
            f"roc_auc={metricas['roc_auc']:.3f} pr_auc={metricas['pr_auc']:.3f}"
        )

        melhor_atual = melhores.get(nome)
        if (
            melhor_atual is None
            or metricas[METRICA_SELECAO] > melhor_atual["metricas"][METRICA_SELECAO]
        ):
            melhores[nome] = {
                "modelo": modelo,
                "parametros": str(parametros),
                "metricas": metricas,
            }

    DIR_TABELAS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(resultados).to_csv(DIR_TABELAS / "baselines_tuning.csv", index=False)
    logger.info(f"{len(resultados)} treinos de modelo realizados no tuning")

    # 2. Auditoria de vazamento, na validação, antes de tocar o teste
    for nome, melhor in melhores.items():
        auditar_vazamento(nome, melhor["metricas"], val_y.mean())

    importancias_rf = pd.Series(
        melhores["random_forest"]["modelo"].feature_importances_, index=treino_x.columns
    ).sort_values(ascending=False)
    logger.info(
        "RF — 5 maiores MDI (suspeite se uma dominar):\n"
        + importancias_rf.head(5).round(3).to_string()
    )

    # 3. Teste: uma única vez, só com a melhor configuração de cada modelo.
    # O modelo avaliado é o mesmo treinado no treino; é ele que os métodos
    # de importância vão explicar na Etapa 9.
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    finais = []
    for nome, melhor in melhores.items():
        metricas_teste = avaliar_modelo(melhor["modelo"], teste_x, teste_y)
        finais.append(
            {"modelo": nome, "parametros": melhor["parametros"], **metricas_teste}
        )
        joblib.dump(melhor["modelo"], DIR_MODELOS / f"{nome}.joblib")

    tabela_final = pd.DataFrame(finais)
    tabela_final.to_csv(DIR_TABELAS / "baselines.csv", index=False)
    logger.info("Resultado no teste:\n" + tabela_final.round(3).to_string())

    salvar_coeficientes_logreg(
        melhores["logreg"]["modelo"],
        treino_x.columns,
        DIR_TABELAS / "coeficientes_logreg.csv",
    )


if __name__ == "__main__":
    main()
