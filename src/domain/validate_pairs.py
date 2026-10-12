"""Etapa 15 — Diferencial 4: validação de domínio das interações do PyGAwLLfs.

Pergunta: os pares que o PyGAwLLfs aponta como interativos correspondem a
padrões reais de atraso no VRA? Para cada par, a taxa de atraso no cruzamento
das duas variáveis difere do que os efeitos isolados de cada uma preveem — e
esse padrão se repete de um ano para o outro?

Recebe:
    results/tables/interacao_gawll.csv   vint de cada execução (Etapa 9)
    results/tables/ranking_consolidado.csv  vimp médio (tamanho dos nós no grafo)
    data/processed/{train,val}.parquet   atributos e alvo (o teste fica de fora, §4.9)
    data/interim/vra_target.parquet      empresa e aeroportos em texto (para ler os pares)

Passos:
1. Seleção dos pares: vint médio das 5 sementes por modelo; candidatos = união
   do top-10 de cada modelo. Controles = os 10 pares com pior rank médio nos
   três modelos (mostram se o teste descritivo distingue alguma coisa).
2. Cada feature volta à variável de domínio que codifica (empresa_te →
   empresa, as dummies de porte → porte, dia_ano → semana do ano...).
   Tipo do par, antes de qualquer teste:
     estrutural   as duas features codificam a mesma variável: nada a testar
     redundante   uma variável quase determina a outra (coeficiente de
                  incerteza >= LIMIAR_REDUNDANCIA; ex.: mês e trimestre) — o
                  fenômeno do Diferencial 1: o PyGAwLLfs lê redundância como interação
     operacional  os demais: candidatos a interação real
3. Teste descritivo, separado em 2024 e em 2025: tabela cruzada (n voos e k
   atrasados por célula) ajustada por uma regressão logística SÓ com os efeitos
   isolados de cada variável. Desvio = média, ponderada por n, de |taxa
   observada − taxa prevista sem interação|, em pontos percentuais. Um
   bootstrap paramétrico dá o desvio que o puro acaso produziria (p95 = limiar
   de ruído). Replicação = correlação, entre os dois anos, do resíduo de cada célula.
   Por que por ano: em cruzamentos finos de calendário, cada célula vira quase
   uma data, e choques de um dia (tempo, panes) aparecem como "interação". Uma
   interação operacional real se repete nos dois anos; um choque de um dia, não.
4. Classificação de um par operacional como "corroborado descritivamente" se,
   NOS DOIS ANOS, o desvio passa o p95 do ruído e vale ao menos LIMIAR_PP, E a
   correlação dos resíduos entre anos é ao menos LIMIAR_REPLICACAO. Nunca
   "confirmado": é evidência observacional, e um confundidor omitido poderia
   produzir o mesmo padrão (docs/metodologia.md, Diferencial 4).

Produz:
    results/tables/dominio_pares.csv       um par por linha: ranks no PyGAwLLfs,
                                           tipo, desvios por ano, ruído, replicação
    results/tables/dominio_celulas.csv     células dos pares das figuras (anos juntos)
    results/figures/grafo_interacao.{png,pdf}        grafo reduzido (5% das arestas)
    results/figures/dominio_<a>_x_<b>.{png,pdf}      heatmaps dos N_HEATMAPS pares
                                                     corroborados mais fortes no PyGAwLLfs
Quem consome: o Cap. 4 (seção de validação de domínio) e docs/validacao-dominio.md.
"""

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # só gera arquivo, sem abrir janela
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

from src.compare.figuras import COR_TEXTO, COR_TEXTO_SECUNDARIO, ROTULO_MODELO  # noqa: E402
from src.compare.rankings import MODELOS  # noqa: E402
from src.importance import comum  # noqa: E402
from src.interaction import graph  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DIR_DADOS = RAIZ / "data" / "processed"
ARQUIVO_TARGET = RAIZ / "data" / "interim" / "vra_target.parquet"
DIR_FIGURAS = RAIZ / "results" / "figures"

TOP_PARES_POR_MODELO = 10
N_CONTROLES = 10
FRACAO_GRAFO = graph.FRACAO_ARESTAS  # grafo reduzido: 5% das arestas mais fortes, como no artigo
ANOS = [2024, 2025]          # 2025 vai só até outubro (o teste fica de fora)
N_BOOTSTRAP = 200
LIMIAR_PP = 1.0              # relevância prática mínima do desvio, em pontos percentuais
LIMIAR_REPLICACAO = 0.5      # correlação mínima dos resíduos entre os dois anos ("forte")
# Coeficiente de incerteza a partir do qual o par é redundante. Nos 35 pares
# analisados há um vão entre 0,40 e 0,85: os derivados do calendário ficam
# acima (dia_mes × dia_ano dá 0,85, e não 1, porque 2024 é bissexto) e os
# demais abaixo. Qualquer limiar dentro do vão dá a mesma classificação.
LIMIAR_REDUNDANCIA = 0.8
N_HEATMAPS = 2
CELULA_MINIMA = 100          # células menores ficam fora da replicação e dos heatmaps
SEMENTE = 42

# Para ler os pares, cada feature volta à variável de domínio que ela codifica
TOP_CATEGORIAS = {"empresa": 10, "origem": 15, "destino": 15}
VARIAVEL_DE_DOMINIO = {
    "empresa_te": "empresa", "origem_te": "origem", "destino_te": "destino",
    **{f"porte_origem_{p}": "porte_origem" for p in ["pequeno", "medio", "grande", "hub"]},
    **{f"porte_destino_{p}": "porte_destino" for p in ["pequeno", "medio", "grande", "hub"]},
    **{f"tipo_linha_{t}": "tipo_linha" for t in ["C", "G", "I", "N"]},
    "minuto": "minuto_quarto",
    "dia_ano": "semana",  # sazonalidade, não data: cada célula junta 7 dias
}
EM_QUINTIS = ["duracao_prevista_min", "congestion_origem", "congestion_destino"]

COR_TIPO = {"operacional": "#2a78d6", "redundante": "#eb6834", "estrutural": "#8c8b85"}
ESCALA_TAXA = LinearSegmentedColormap.from_list(
    "taxa", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
ESCALA_DESVIO = LinearSegmentedColormap.from_list("desvio", ["#256abf", "#f0efec", "#e34948"])


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------

def carregar_base() -> pd.DataFrame:
    """Treino + validação, com uma coluna por variável de domínio, o ano e o alvo."""
    partes = [pd.read_parquet(DIR_DADOS / f"{nome}.parquet") for nome in ["train", "val"]]
    base = pd.concat(partes)
    texto = pd.read_parquet(ARQUIVO_TARGET, columns=[
        "ICAO Empresa Aérea", "ICAO Aeródromo Origem", "ICAO Aeródromo Destino"])
    texto = texto.loc[base.index]  # o pipeline preserva o índice da base limpa

    dominio = pd.DataFrame(index=base.index)
    for nome, coluna in [("empresa", "ICAO Empresa Aérea"),
                         ("origem", "ICAO Aeródromo Origem"),
                         ("destino", "ICAO Aeródromo Destino")]:
        principais = texto[coluna].value_counts().index[:TOP_CATEGORIAS[nome]]
        dominio[nome] = texto[coluna].where(texto[coluna].isin(principais))

    for prefixo in ["porte_origem", "porte_destino", "tipo_linha"]:
        dummies = base.filter(like=f"{prefixo}_")
        dominio[prefixo] = dummies.idxmax(axis=1).str.replace(f"{prefixo}_", "")

    for coluna in ["hora", "dia_sem", "mes", "dia_mes", "trimestre",
                   "periodo_dia", "fim_semana", "feriado", "vespera_feriado"]:
        dominio[coluna] = base[coluna]
    dominio["semana"] = (base["dia_ano"] - 1) // 7 + 1
    dominio["dia_ano"] = base["dia_ano"]  # diário, só para medir redundância (tipo_do_par)
    dominio["minuto_quarto"] = base["minuto"] // 15 * 15
    for coluna in EM_QUINTIS:
        dominio[coluna] = pd.qcut(base[coluna], 5, duplicates="drop").astype(str)

    dominio["ano"] = base["Partida Prevista"].dt.year
    dominio["atrasado"] = base["atrasado"].to_numpy()
    return dominio


def variavel(feature: str) -> str:
    return VARIAVEL_DE_DOMINIO.get(feature, feature)


# ---------------------------------------------------------------------------
# Seleção e tipo dos pares
# ---------------------------------------------------------------------------

def interacao_media() -> pd.DataFrame:
    """vint médio das 5 sementes, com o rank de cada par em cada modelo."""
    interacao = pd.read_csv(comum.DIR_TABELAS / "interacao_gawll.csv")
    return graph.interacao_media(interacao)


def selecionar_pares(media: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por par: rank em cada modelo, e se é candidato ou controle."""
    largo = media.pivot_table(index=["feature_a", "feature_b"], columns="model",
                              values=["rank", "interaction"])
    largo.columns = [f"{metrica}_{modelo}" for metrica, modelo in largo.columns]
    largo = largo.reset_index()
    ranks = largo[[f"rank_{m}" for m in MODELOS]]
    largo["rank_medio"] = ranks.mean(axis=1)
    largo["modelos_top10"] = ranks.apply(
        lambda linha: ";".join(m for m in MODELOS if linha[f"rank_{m}"] <= TOP_PARES_POR_MODELO),
        axis=1)

    candidatos = largo[largo["modelos_top10"] != ""].assign(grupo="candidato")

    # Controles: do fundo do ranking, mas sem repetir o par de domínio de um
    # candidato nem de outro controle. Duas dummies da mesma variável viram a
    # mesma tabela cruzada (ex.: tipo_linha_C × origem e tipo_linha_I × origem
    # são ambas tipo_linha × origem) — um controle assim não é independente.
    def par_de_dominio(linha):
        return frozenset((variavel(linha["feature_a"]), variavel(linha["feature_b"])))

    usados = {par_de_dominio(linha) for _, linha in candidatos.iterrows()}
    escolhidos = []
    for _, linha in largo.sort_values("rank_medio", ascending=False).iterrows():
        chave = par_de_dominio(linha)
        if len(chave) == 2 and chave not in usados:  # len 1 = mesma variável (estrutural)
            escolhidos.append(linha)
            usados.add(chave)
        if len(escolhidos) == N_CONTROLES:
            break
    controles = pd.DataFrame(escolhidos).assign(grupo="controle")
    return pd.concat([candidatos, controles], ignore_index=True)


def coeficiente_de_incerteza(x: pd.Series, y: pd.Series) -> float:
    """U(y|x) = 1 − H(y|x)/H(y): quanto saber x elimina da incerteza sobre y
    (1 = x determina y; 0 = independentes)."""
    contagens = pd.crosstab(x, y).to_numpy(dtype=float)
    conjunta = contagens / contagens.sum()
    p_x, p_y = conjunta.sum(axis=1), conjunta.sum(axis=0)
    h_y = -np.sum(p_y[p_y > 0] * np.log(p_y[p_y > 0]))
    com_massa = conjunta > 0
    h_y_dado_x = -np.sum(conjunta[com_massa]
                         * np.log((conjunta / p_x[:, None])[com_massa]))
    return 1.0 - h_y_dado_x / h_y if h_y > 0 else 1.0


def tipo_do_par(feature_a: str, feature_b: str, dominio: pd.DataFrame) -> tuple[str, float]:
    """(tipo, coeficiente de incerteza máximo entre os dois sentidos).

    A redundância é medida nas features como o modelo as vê (dia_ano diário,
    não a semana), porque é essa a redundância que o PyGAwLLfs enxerga."""
    var_a, var_b = variavel(feature_a), variavel(feature_b)
    if var_a == var_b:
        return "estrutural", 1.0
    # A feature como está no modelo, quando existe no domínio (ex.: dia_ano diário)
    coluna_a = feature_a if feature_a in dominio.columns else var_a
    coluna_b = feature_b if feature_b in dominio.columns else var_b
    linhas = dominio[[coluna_a, coluna_b]].dropna()
    incerteza = max(coeficiente_de_incerteza(linhas[coluna_a], linhas[coluna_b]),
                    coeficiente_de_incerteza(linhas[coluna_b], linhas[coluna_a]))
    return ("redundante" if incerteza >= LIMIAR_REDUNDANCIA else "operacional"), incerteza


# ---------------------------------------------------------------------------
# Teste descritivo
# ---------------------------------------------------------------------------

def tabela_cruzada(dominio: pd.DataFrame, var_a: str, var_b: str) -> pd.DataFrame:
    """Uma linha por célula (nível de A, nível de B): n voos, k atrasados."""
    linhas = dominio[[var_a, var_b, "atrasado"]].dropna()
    tabela = (linhas.groupby([var_a, var_b], observed=True)["atrasado"]
              .agg(n="size", k="sum").reset_index())
    tabela.columns = ["nivel_a", "nivel_b", "n", "k"]
    return tabela


def matriz_sem_interacao(tabela: pd.DataFrame) -> np.ndarray:
    """Uma coluna por nível de A e de B (menos um de cada): só efeitos isolados."""
    a = pd.get_dummies(tabela["nivel_a"].astype(str), prefix="a", drop_first=True)
    b = pd.get_dummies(tabela["nivel_b"].astype(str), prefix="b", drop_first=True)
    return pd.concat([a, b], axis=1).to_numpy(dtype=float)


def ajustar_sem_interacao(matriz: np.ndarray, n: np.ndarray, k: np.ndarray,
                          modelo: LogisticRegression) -> np.ndarray:
    """Taxa prevista em cada célula por logit(p) = constante + efeito(A) + efeito(B).

    Cada célula entra duas vezes — como "atrasou" com peso k e como "não
    atrasou" com peso n − k —, o que equivale a ajustar voo a voo.
    Pesos divididos pelo total: o resultado não muda, e o otimizador converge melhor.
    """
    x = np.vstack([matriz, matriz])
    y = np.concatenate([np.ones(len(n)), np.zeros(len(n))])
    pesos = np.concatenate([k, n - k]) / n.sum()
    modelo.fit(x, y, sample_weight=pesos)
    return modelo.predict_proba(matriz)[:, 1]


def desvio_pp(n: np.ndarray, k: np.ndarray, previsto: np.ndarray) -> float:
    """Média ponderada por n de |taxa observada − taxa prevista|, em pontos percentuais."""
    return float(np.sum(n * np.abs(k / n - previsto)) / n.sum() * 100)


def testar_tabela(tabela: pd.DataFrame, rng: np.random.Generator,
                  n_bootstrap: int | None = None) -> tuple[float, float, pd.DataFrame]:
    """(desvio, p95 do ruído, tabela com a previsão sem interação) de uma tabela cruzada.

    n_bootstrap=None usa N_BOOTSTRAP lido na hora da chamada (um valor padrão
    escrito na assinatura seria fixado na importação e ignoraria mudanças)."""
    if n_bootstrap is None:
        n_bootstrap = N_BOOTSTRAP
    n, k = tabela["n"].to_numpy(float), tabela["k"].to_numpy(float)
    matriz = matriz_sem_interacao(tabela)
    # warm_start: cada reajuste do bootstrap parte da solução anterior (mais rápido)
    modelo = LogisticRegression(C=np.inf, max_iter=5000, warm_start=True)
    previsto = ajustar_sem_interacao(matriz, n, k, modelo)

    # Bootstrap paramétrico: se NÃO houvesse interação, quanto desvio o acaso daria?
    ruido = []
    for _ in range(n_bootstrap):
        k_simulado = rng.binomial(n.astype(int), previsto).astype(float)
        previsto_simulado = ajustar_sem_interacao(matriz, n, k_simulado, modelo)
        ruido.append(desvio_pp(n, k_simulado, previsto_simulado))

    tabela = tabela.assign(taxa=k / n, previsto_sem_interacao=previsto)
    ruido_p95 = float(np.percentile(ruido, 95)) if ruido else np.nan  # sem bootstrap: sem limiar
    return desvio_pp(n, k, previsto), ruido_p95, tabela


def testar_par(dominio: pd.DataFrame, var_a: str, var_b: str,
               rng: np.random.Generator) -> tuple[dict, pd.DataFrame]:
    """Teste em cada ano, replicação entre anos e a tabela dos anos juntos (para figura)."""
    resultado, residuos = {}, {}
    for ano in ANOS:
        tabela = tabela_cruzada(dominio[dominio["ano"] == ano], var_a, var_b)
        desvio, ruido, tabela = testar_tabela(tabela, rng)
        resultado[f"desvio_{ano}_pp"] = desvio
        resultado[f"ruido_p95_{ano}_pp"] = ruido
        grandes = tabela[tabela["n"] >= CELULA_MINIMA]
        residuos[ano] = (grandes.set_index(["nivel_a", "nivel_b"])["taxa"]
                         - grandes.set_index(["nivel_a", "nivel_b"])["previsto_sem_interacao"])

    # Replicação: o mesmo desvio aparece nas mesmas células nos dois anos?
    comuns = residuos[ANOS[0]].index.intersection(residuos[ANOS[1]].index)
    resultado["celulas_comuns"] = len(comuns)
    resultado["replicacao_anos"] = (
        float(np.corrcoef(residuos[ANOS[0]].loc[comuns], residuos[ANOS[1]].loc[comuns])[0, 1])
        if len(comuns) >= 3 else np.nan)

    # Anos juntos, só para a figura e para a cobertura (sem bootstrap)
    juntos = tabela_cruzada(dominio, var_a, var_b)
    _, _, juntos = testar_tabela(juntos, rng, n_bootstrap=0)
    resultado["n_celulas"] = len(juntos)
    resultado["cobertura"] = juntos["n"].sum() / len(dominio)
    return resultado, juntos


def classificar(tipo: str, resultado: dict) -> str:
    if tipo == "estrutural":
        return "não testável (mesma variável)"
    if tipo == "redundante":
        return "redundância (não é interação operacional)"
    passa_nos_dois_anos = all(
        resultado[f"desvio_{ano}_pp"] > resultado[f"ruido_p95_{ano}_pp"]
        and resultado[f"desvio_{ano}_pp"] >= LIMIAR_PP
        for ano in ANOS)
    replica = resultado["replicacao_anos"] >= LIMIAR_REPLICACAO
    return "corroborado descritivamente" if passa_nos_dois_anos and replica else "não corroborado"


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def afastar_nos(posicao: dict, distancia_minima: float = 0.15, iteracoes: int = 100) -> dict:
    """Empurra para longe os pares de nós mais próximos que distancia_minima
    (em fração do tamanho do desenho), para os rótulos não se sobreporem.
    O resto do layout continua como o kamada_kawai deixou."""
    nos = list(posicao)
    pontos = np.array([posicao[no] for no in nos], dtype=float)
    minimo = distancia_minima * np.ptp(pontos, axis=0).max()
    for _ in range(iteracoes):
        moveu = False
        for i in range(len(nos)):
            for j in range(i + 1, len(nos)):
                diferenca = pontos[i] - pontos[j]
                distancia = np.linalg.norm(diferenca)
                if distancia < minimo:
                    direcao = diferenca / distancia if distancia > 0 else np.array([1.0, 0.0])
                    passo = (minimo - distancia) / 2
                    pontos[i] += direcao * passo
                    pontos[j] -= direcao * passo
                    moveu = True
        if not moveu:
            break
    return dict(zip(nos, pontos))


def figura_grafo(media: pd.DataFrame, tipos: dict, importancia: pd.DataFrame,
                 caminho_base: Path) -> None:
    """Grafo reduzido de cada modelo: 5% das arestas mais fortes."""
    fig, eixos = plt.subplots(1, 3, figsize=(18, 7), layout="constrained")
    for ax, modelo in zip(eixos, MODELOS):
        pares = media[media["model"] == modelo]
        n_arestas = int(round(len(pares) * FRACAO_GRAFO))
        fortes = pares.nlargest(n_arestas, "interaction")

        grafo = nx.Graph()
        for linha in fortes.itertuples():
            grafo.add_edge(linha.feature_a, linha.feature_b, peso=linha.interaction,
                           tipo=tipos[(linha.feature_a, linha.feature_b)])
        posicao = afastar_nos(nx.kamada_kawai_layout(grafo))

        pesos = np.array([d["peso"] for *_, d in grafo.edges(data=True)])
        larguras = 1 + 5 * (pesos - pesos.min()) / (np.ptp(pesos) or 1)
        cores = [COR_TIPO[d["tipo"]] for *_, d in grafo.edges(data=True)]
        nx.draw_networkx_edges(grafo, posicao, ax=ax, width=larguras, edge_color=cores, alpha=0.85)

        imp = (importancia[importancia["model"] == modelo]
               .set_index("feature")["importance_norm"])
        tamanhos = [150 + 900 * imp.get(no, 0) for no in grafo.nodes]
        nx.draw_networkx_nodes(grafo, posicao, ax=ax, node_size=tamanhos,
                               node_color="#f0efec", edgecolors=COR_TEXTO_SECUNDARIO, linewidths=1)
        # Rótulos longos em duas linhas (quebra no último "_"), com fundo branco:
        # ficam legíveis mesmo com nós vizinhos ou uma aresta passando por trás
        rotulos = {no: (no[:no.rindex("_")] + "\n" + no[no.rindex("_") + 1:]
                        if len(no) > 14 and "_" in no else no) for no in grafo.nodes}
        nx.draw_networkx_labels(grafo, posicao, labels=rotulos, ax=ax, font_size=7.5,
                                font_color=COR_TEXTO,
                                bbox={"facecolor": "white", "edgecolor": "none",
                                      "alpha": 0.75, "pad": 0.5})
        ax.set_title(f"{ROTULO_MODELO[modelo]} — {n_arestas} arestas", loc="left",
                     color=COR_TEXTO, fontsize=11)
        ax.margins(0.12)  # espaço para os rótulos dos nós da borda
        ax.axis("off")

    legenda = [Line2D([0], [0], color=cor, linewidth=3, label=tipo)
               for tipo, cor in COR_TIPO.items()]
    fig.legend(handles=legenda, loc="outside upper right", ncol=3, frameon=False,
               fontsize=9, title="tipo do par", title_fontsize=9)
    fig.suptitle("Grafo de interação reduzido do PyGAwLLfs (5% das arestas mais fortes, "
                 "média das 5 sementes)", x=0.01, ha="left", color=COR_TEXTO, fontsize=12)
    fig.supxlabel("Espessura = força da interação (vint); tamanho do nó = importância (vimp, "
                  "normalizada). Estrutural = mesma variável; redundante = uma variável quase "
                  "determina a outra.", x=0.01, ha="left", color=COR_TEXTO_SECUNDARIO, fontsize=8.5)
    salvar(fig, caminho_base)


def figura_heatmap(tabela: pd.DataFrame, var_a: str, var_b: str, titulo: str,
                   caminho_base: Path) -> None:
    """Esquerda: taxa observada. Direita: observado − previsto sem interação."""
    tabela = tabela[tabela["n"] >= CELULA_MINIMA].copy()
    tabela["desvio"] = (tabela["taxa"] - tabela["previsto_sem_interacao"]) * 100
    taxa = tabela.pivot(index="nivel_a", columns="nivel_b", values="taxa") * 100
    desvio = tabela.pivot(index="nivel_a", columns="nivel_b", values="desvio")
    # Limites robustos (percentil 98): uma célula extrema não apaga o resto;
    # o que passa do limite fica com a cor da ponta (a barra mostra a seta)
    limite = np.nanpercentile(np.abs(desvio.to_numpy()), 98)
    taxa_maxima = np.nanpercentile(taxa.to_numpy(), 98)

    fig, eixos = plt.subplots(1, 2, figsize=(15, 0.42 * len(taxa) + 2.2), layout="constrained")
    for ax, matriz, escala, vmin, vmax, rotulo in [
            (eixos[0], taxa, ESCALA_TAXA, 0, taxa_maxima, "taxa de atraso (%)"),
            (eixos[1], desvio, ESCALA_DESVIO, -limite, limite, "observado − sem interação (p.p.)")]:
        ax.set_facecolor("white")  # célula sem dados suficientes fica em branco
        imagem = ax.pcolormesh(np.ma.masked_invalid(matriz.to_numpy()), cmap=escala,
                               vmin=vmin, vmax=vmax, edgecolors="white", linewidth=1)
        ax.set_xticks(np.arange(matriz.shape[1]) + 0.5, matriz.columns, fontsize=7.5,
                      rotation=90 if matriz.shape[1] > 12 else 0)
        ax.set_yticks(np.arange(matriz.shape[0]) + 0.5, matriz.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(var_b, color=COR_TEXTO_SECUNDARIO, fontsize=9)
        ax.set_ylabel(var_a, color=COR_TEXTO_SECUNDARIO, fontsize=9)
        ax.tick_params(length=0, colors=COR_TEXTO_SECUNDARIO)
        for lado in ax.spines.values():
            lado.set_visible(False)
        barra = fig.colorbar(imagem, ax=ax, shrink=0.85,
                             extend="max" if vmin == 0 else "both")
        barra.set_label(rotulo, color=COR_TEXTO_SECUNDARIO, fontsize=9)
        barra.outline.set_visible(False)
        barra.ax.tick_params(labelsize=8, colors=COR_TEXTO_SECUNDARIO)
    fig.suptitle(titulo, x=0.01, ha="left", color=COR_TEXTO, fontsize=12)
    fig.supxlabel(f"Treino + validação (jan/2024–out/2025). Células com menos de {CELULA_MINIMA} "
                  "voos em branco. À direita, vermelho = mais atraso do que os efeitos isolados "
                  "preveem; azul = menos. Escalas cortadas no percentil 98 (setas nas barras).",
                  x=0.01, ha="left", color=COR_TEXTO_SECUNDARIO, fontsize=8.5)
    salvar(fig, caminho_base)


def salvar(fig, caminho_base: Path) -> None:
    caminho_base.parent.mkdir(parents=True, exist_ok=True)
    for extensao in ["png", "pdf"]:
        fig.savefig(caminho_base.with_suffix(f".{extensao}"), dpi=200,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Salvo: {caminho_base.name}.png e .pdf")


# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(SEMENTE)
    dominio = carregar_base()
    media = interacao_media()
    pares = selecionar_pares(media)
    logger.info(f"Base: {len(dominio)} voos (treino + validação) | "
                f"{(pares['grupo'] == 'candidato').sum()} candidatos, {N_CONTROLES} controles")

    linhas, tabelas = [], {}
    for par in pares.itertuples():
        tipo, incerteza = tipo_do_par(par.feature_a, par.feature_b, dominio)
        linha = {
            "grupo": par.grupo, "feature_a": par.feature_a, "feature_b": par.feature_b,
            "variavel_a": variavel(par.feature_a), "variavel_b": variavel(par.feature_b),
            **{f"rank_{m}": getattr(par, f"rank_{m}") for m in MODELOS},
            "rank_medio": par.rank_medio, "modelos_top10": par.modelos_top10,
            "tipo": tipo, "coef_incerteza": incerteza,
        }
        if tipo != "estrutural":
            resultado, tabela = testar_par(dominio, linha["variavel_a"], linha["variavel_b"], rng)
            linha.update(resultado)
            tabelas[(par.feature_a, par.feature_b)] = tabela
        linha["classificacao"] = classificar(tipo, linha)
        linhas.append(linha)
        logger.info(f"{par.grupo:9s} {par.feature_a} x {par.feature_b}: {tipo} → "
                    f"{linha['classificacao']}")

    resultado = pd.DataFrame(linhas).sort_values(["grupo", "rank_medio"])
    comum.salvar_tabela(resultado.round(4), "dominio_pares.csv")

    # Grafo reduzido: precisa do tipo de todas as arestas que entram nele
    tipos = {}
    for modelo in MODELOS:
        pares_modelo = media[media["model"] == modelo]
        for linha in pares_modelo.nlargest(int(round(len(pares_modelo) * FRACAO_GRAFO)),
                                           "interaction").itertuples():
            chave = (linha.feature_a, linha.feature_b)
            if chave not in tipos:
                tipos[chave] = tipo_do_par(*chave, dominio)[0]
    importancia = pd.read_csv(comum.DIR_TABELAS / "ranking_consolidado.csv")
    importancia = importancia[importancia["method"] == "gawll"]
    figura_grafo(media, tipos, importancia, DIR_FIGURAS / "grafo_interacao")

    # Heatmaps: os N_HEATMAPS pares corroborados mais fortes no PyGAwLLfs
    corroborados = resultado[(resultado["grupo"] == "candidato")
                             & (resultado["classificacao"] == "corroborado descritivamente")]
    celulas = []
    for par in corroborados.nsmallest(N_HEATMAPS, "rank_medio").itertuples():
        tabela = tabelas[(par.feature_a, par.feature_b)]
        celulas.append(tabela.assign(feature_a=par.feature_a, feature_b=par.feature_b))
        replicacao = f"{par.replicacao_anos:.2f}".replace(".", ",")
        titulo = (f"{par.variavel_a} × {par.variavel_b}: taxa de atraso e desvio do esperado "
                  f"sem interação (replicação entre 2024 e 2025: r = {replicacao})")
        figura_heatmap(tabela, par.variavel_a, par.variavel_b, titulo,
                       DIR_FIGURAS / f"dominio_{par.variavel_a}_x_{par.variavel_b}")
    if celulas:
        comum.salvar_tabela(pd.concat(celulas).round(5), "dominio_celulas.csv")

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        colunas = ["grupo", "feature_a", "feature_b", "tipo", "desvio_2024_pp",
                   "ruido_p95_2024_pp", "desvio_2025_pp", "ruido_p95_2025_pp",
                   "replicacao_anos", "classificacao"]
        logger.info("Resultado:\n" + resultado[colunas].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
