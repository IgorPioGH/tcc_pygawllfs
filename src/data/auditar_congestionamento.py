"""Auditoria do congestionamento programado (§4.3) — limitação documentada.

O features.py conta os voos por (aeroporto, hora) sobre vra_target.parquet,
que só tem voos REALIZADOS: os cancelados saem no clean.py. A contagem
correta pelo §4.3 ("derivado só da malha programada") inclui os cancelados,
porque o cancelamento só é conhecido depois. Ou seja, congestion_origem e
congestion_destino carregam, em tese, informação pós-hoc.

Este script mede o tamanho do problema sem descongelar o dataset:
    1. em quantas linhas a contagem realizada difere da programada;
    2. a taxa de atraso em slots com e sem cancelamento;
    3. o F1 / ROC-AUC / PR-AUC de validação do RF e do XGBoost treinados com
       a versão atual (realizada) e com a correta (programada).
Usa só treino e validação (§4.9: o teste fica de fora).

Recebe data/interim/vra_bruto.parquet, data/processed/{train,val}.parquet e
os hiperparâmetros de baselines.csv. Produz
results/tables/auditoria_congestionamento.csv. Quem consome: o Cap. 3
(limitação) e o docs/dicionario-features.md.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.importance.comum import salvar_tabela  # noqa: E402
from src.models.baselines import (  # noqa: E402
    avaliar_modelo,
    carregar,
    criar_modelo,
    ler_configuracoes_escolhidas,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARQUIVO_BRUTO = RAIZ / "data" / "interim" / "vra_bruto.parquet"
DATA_INICIO, DATA_FIM = "2024-01-01", "2026-01-01"  # mesmo recorte do target.py


def contagens_programadas() -> pd.DataFrame:
    """Recalcula sobre o bruto (realizados E cancelados) os voos programados no slot de
    partida da origem e no de chegada do destino, e os cancelados no slot de partida.
    O índice é o mesmo dos parquets processados (o pipeline preserva o índice).

    Usada em: main.
    Importância: é a contagem "correta" do congestionamento programado, contra a qual
    se mede a limitação do features.py (que conta sem os cancelados).
    """
    bruto = pd.read_parquet(ARQUIVO_BRUTO, columns=[
        "ICAO Aeródromo Origem", "ICAO Aeródromo Destino",
        "Partida Prevista", "Chegada Prevista", "Situação Voo",
    ])
    bruto = bruto[(bruto["Partida Prevista"] >= DATA_INICIO)
                  & (bruto["Partida Prevista"] < DATA_FIM)]

    slot_partida = [bruto["ICAO Aeródromo Origem"], bruto["Partida Prevista"].dt.floor("h")]
    slot_chegada = [bruto["ICAO Aeródromo Destino"], bruto["Chegada Prevista"].dt.floor("h")]
    cancelado = (bruto["Situação Voo"] != "REALIZADO").astype(int)

    return pd.DataFrame({
        "programado_origem": bruto.groupby(slot_partida)["Partida Prevista"].transform("size"),
        "programado_destino": bruto.groupby(slot_chegada)["Chegada Prevista"].transform("size"),
        "cancelados_no_slot": cancelado.groupby(slot_partida).transform("sum"),
    })


def descrever(nome: str, x: pd.DataFrame, y: pd.Series, programadas: pd.DataFrame) -> dict:
    """Num conjunto: quanto a contagem realizada difere da programada, e o atraso por
    slot.

    Usada em: main.
    Importância: descreve o tamanho da limitação antes de medir o efeito dela no
    modelo.
    """
    p = programadas.loc[x.index]
    com_cancelamento = p["cancelados_no_slot"] > 0
    return {
        "conjunto": nome,
        "linhas": len(x),
        "fracao_linhas_diferentes": (x["congestion_origem"] != p["programado_origem"]).mean(),
        "fracao_slots_com_cancelamento": com_cancelamento.mean(),
        "taxa_atraso_slot_com_cancelamento": y[com_cancelamento.to_numpy()].mean(),
        "taxa_atraso_slot_sem_cancelamento": y[~com_cancelamento.to_numpy()].mean(),
    }


def comparar_modelos(x_treino, y_treino, x_val, y_val, programadas) -> list[dict]:
    """Treina RF e XGBoost com o congestionamento atual e com o programado e mede os
    dois na validação.

    Usada em: main.
    Importância: o efeito no F1 de validação foi pequeno, o que justificou manter o
    dataset congelado e só documentar a limitação.
    """
    configuracoes = ler_configuracoes_escolhidas()
    peso_positivo = (y_treino == 0).sum() / (y_treino == 1).sum()

    def versao_programada(x):
        """Cópia de x com as duas colunas de congestionamento trocadas pelas contagens
        programadas.

        Usada em: comparar_modelos.
        Importância: o resto das features fica idêntico, então a diferença medida vem
        só do congestionamento.
        """
        x = x.copy()
        x["congestion_origem"] = programadas.loc[x.index, "programado_origem"].to_numpy()
        x["congestion_destino"] = programadas.loc[x.index, "programado_destino"].to_numpy()
        return x

    linhas = []
    for nome in ["random_forest", "xgboost"]:
        for versao, transformar in [("realizada (atual)", lambda x: x),
                                    ("programada (correta)", versao_programada)]:
            modelo = criar_modelo(nome, configuracoes[nome], peso_positivo)
            modelo.fit(transformar(x_treino), y_treino)
            metricas = avaliar_modelo(modelo, transformar(x_val), y_val)
            linhas.append({"modelo": nome, "versao": versao, **{
                k: metricas[k] for k in ["f1", "roc_auc", "pr_auc"]}})
            logger.info(f"{nome} {versao}: F1 {metricas['f1']:.4f}")
    return linhas


def main() -> None:
    """Roda a auditoria (descrição e comparação de modelos) e salva
    auditoria_congestionamento.csv. Não toca o teste.

    Usada em: python src/data/auditar_congestionamento.py; main.py, etapa "extras".
    Importância: dá o número que sustenta a limitação do congestionamento no Cap. 3 e
    em docs/dicionario-features.md.
    """
    programadas = contagens_programadas()
    x_treino, y_treino = carregar("train")
    x_val, y_val = carregar("val")

    descricao = pd.DataFrame([
        descrever("treino", x_treino, y_treino, programadas),
        descrever("validacao", x_val, y_val, programadas),
    ])
    logger.info("Diferença entre contagem realizada e programada:\n"
                + descricao.round(4).to_string(index=False))

    modelos = pd.DataFrame(comparar_modelos(x_treino, y_treino, x_val, y_val, programadas))
    logger.info("Validação, versão atual × correta:\n" + modelos.round(4).to_string(index=False))

    # Uma tabela só: a descrição por conjunto e a comparação por modelo
    tabela = pd.concat([descricao.assign(parte="descricao"),
                        modelos.assign(parte="modelos")], ignore_index=True)
    caminho = salvar_tabela(tabela, "auditoria_congestionamento.csv")
    logger.info(f"Salvo em {caminho}")


if __name__ == "__main__":
    main()
