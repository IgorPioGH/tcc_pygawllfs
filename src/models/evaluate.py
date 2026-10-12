"""Métricas de classificação usadas em todos os modelos do projeto."""

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def avaliar(y_verdade, y_previsto, y_score) -> dict[str, float]:
    """Calcula as sete métricas de um classificador binário: acurácia, acurácia
    balanceada, precisão, recall, F1, ROC-AUC e PR-AUC.

    Usada em: baselines.avaliar_modelo (e, por ela, na Etapa 6, na fidelidade e na
    auditoria do congestionamento).
    Importância: fonte única das métricas do projeto; F1 e PR-AUC são as principais
    por causa do desbalanceamento (§4.2).

    Args:
        y_verdade: rótulos reais (0/1).
        y_previsto: classes previstas (0/1). Usadas por acurácia, precisão,
            recall e F1, que dependem do corte em 0,5.
        y_score: probabilidade prevista da classe 1. Usada por ROC-AUC e
            PR-AUC, que medem a ordenação e não dependem do corte.

    Returns:
        dict com as sete métricas (as chaves viram colunas do CSV).
    """
    return {
        "acuracia": accuracy_score(y_verdade, y_previsto),
        "acuracia_balanceada": balanced_accuracy_score(y_verdade, y_previsto),
        # zero_division=0: um modelo que nunca prevê 1 (o baseline trivial)
        # deixa a precisão indefinida; registramos 0 em vez de aviso
        "precisao": precision_score(y_verdade, y_previsto, zero_division=0),
        "recall": recall_score(y_verdade, y_previsto),
        "f1": f1_score(y_verdade, y_previsto, zero_division=0),
        "roc_auc": roc_auc_score(y_verdade, y_score),
        "pr_auc": average_precision_score(y_verdade, y_score),
    }
