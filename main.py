"""Roda o pipeline do TCC de ponta a ponta, na ordem certa.

Cada etapa é um ou mais scripts de src/, executados como processos separados
(python src/.../arquivo.py) a partir da raiz do repositório — exatamente como
rodar cada um à mão. As etapas e as pastas de src/ seguem as fases do CRISP-DM:
dados, preparação, modelagem, avaliação e interpretação. Este arquivo só cuida da ordem, do tempo de cada etapa e
de parar no primeiro erro, dizendo de onde retomar.

Uso (da raiz, com o ambiente ativado: conda activate tcc_pygawllfs):
    python main.py                       # o pipeline inteiro
    python main.py --listar              # as etapas por fase, o que fazem e o tempo
    python main.py --de importancias     # a partir de uma etapa
    python main.py --ate baselines       # até uma etapa
    python main.py --so rankings sintese # só as etapas citadas (na ordem do pipeline)
    python main.py --testes              # roda também as suítes de teste no fim
    python main.py --simular             # mostra o que rodaria, sem rodar

As etapas longas retomam de onde pararam: o PyGAwLLfs e o dataset sintético
pulam as execuções que já têm checkpoint em experiments/runs/. Para rodar
tudo sem o notebook suspender: systemd-inhibit --what=idle:sleep python main.py
"""

import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
ARQUIVO_LOG = RAIZ / "experiments" / "runs" / "pipeline.log"

logger = logging.getLogger("pipeline")


# Fases do CRISP-DM (Chapman et al., 2000), na ordem em que o trabalho as percorre.
# O entendimento do problema (fase 1) não tem código: está em docs/01-problema.md.
FASES = ["Dados", "Preparação", "Modelagem", "Avaliação", "Interpretação"]


@dataclass
class Etapa:
    """Uma etapa do pipeline: nome, fase do CRISP-DM, scripts em ordem, descrição e
    tempo esperado.

    Usada em: lista ETAPAS, escolher_etapas e main deste arquivo.
    Importância: a ordem de ETAPAS é a ordem de dependência dos dados; é o mapa do
    pipeline num lugar só. A fase diz a que parte do CRISP-DM cada etapa pertence (a
    mesma pasta de src/ onde estão os seus scripts).
    """
    nome: str
    fase: str
    scripts: list[str]
    descricao: str
    tempo_esperado: str


# Ordem obrigatória: cada etapa lê o que as anteriores produziram.
ETAPAS = [
    Etapa("download", "Dados", ["src/dados/download.py"],
          "baixa os 24 CSVs mensais do VRA (ANAC) para data/raw/; pula os já baixados",
          "~5 min (depende da rede)"),
    Etapa("dados", "Dados", ["src/dados/ler_dados.py"],
          "consolida os 24 CSVs numa base bruta única",
          "< 1 min"),
    Etapa("preparacao", "Preparação", ["src/preparacao/clean.py", "src/preparacao/target.py",
                                       "src/preparacao/features.py"],
          "limpa, cria o alvo, as 29 features, a divisão temporal e a subamostra",
          "< 1 min"),
    Etapa("eda", "Dados", ["src/dados/eda.py"],
          "qualidade da base bruta, funil de descartes, taxas de atraso por atributo, "
          "sazonalidade e correlação entre atributos (só até out/2025)",
          "~1 min"),
    Etapa("checagens", "Preparação", ["src/preparacao/checagens.py"],
          "o balanceamento preserva as relações, o target encoding generaliza e a "
          "divisão temporal tem as contagens dos parquets",
          "< 1 min"),
    Etapa("baselines", "Modelagem", ["src/modelagem/baselines.py"],
          "LogReg, RF e XGBoost: ajuste na validação e avaliação única no teste",
          "~1 min"),
    Etapa("pygawllfs", "Modelagem", ["src/modelagem/gawll.py", "src/modelagem/consolidar_gawll.py"],
          "15 execuções do PyGAwLLfs corrigido (3 modelos x 5 sementes) e suas tabelas",
          "~5 h (retoma do checkpoint)"),
    Etapa("importancias", "Modelagem", ["src/modelagem/intrinsic.py", "src/modelagem/permutation.py",
                                        "src/modelagem/shap_global.py",
                                        "src/modelagem/friedman_h.py"],
          "importância intrínseca, Permutation e TreeSHAP; H de Friedman nos 406 pares; "
          "tudo na validação",
          "~25 min"),
    Etapa("desempenho", "Avaliação", ["src/avaliacao/matriz_confusao.py"],
          "matrizes de confusão dos modelos no teste, conferidas com baselines.csv",
          "< 1 min"),
    Etapa("rankings", "Avaliação", ["src/avaliacao/rankings.py", "src/avaliacao/stability.py",
                                    "src/interpretacao/figuras_concordancia.py",
                                    "src/avaliacao/checagem_subamostra.py"],
          "concordância entre rankings, estabilidade entre sementes, figuras e a checagem "
          "da subamostra do PyGAwLLfs",
          "~2 min"),
    Etapa("sintetico", "Avaliação", ["src/avaliacao/sintetico_gerar.py",
                                     "src/avaliacao/sintetico_aplicar.py",
                                     "src/avaliacao/sintetico_recuperar.py"],
          "dataset com gabarito, todos os métodos e a recuperação da verdade",
          "~30 min (~3 min com os checkpoints do GA)"),
    Etapa("interacoes", "Avaliação", ["src/avaliacao/interacoes.py"],
          "interações do PyGAwLLfs × H de Friedman, com o tipo de cada par",
          "< 1 min"),
    Etapa("fidelidade", "Avaliação", ["src/avaliacao/fidelity.py"],
          "retenção top-k, subconjuntos do GA e sorteios; avalia no teste",
          "~13 min"),
    Etapa("recortes", "Avaliação", ["src/avaliacao/recortes.py"],
          "recortes 2024 x 2025 (mesmo calendário): as quatro importâncias em cada ano e a "
          "concordância entre os anos",
          "~4 h (retoma do checkpoint)"),
    Etapa("sintese", "Avaliação", ["src/avaliacao/sintese.py"],
          "quadro-síntese: os quatro critérios por método e por modelo (só lê CSV)",
          "< 1 min"),
    Etapa("figuras", "Interpretação", ["src/interpretacao/figura_grafo.py"],
          "figuras que explicam os resultados: grafo de interação",
          "< 1 min"),
]
assert all(etapa.fase in FASES for etapa in ETAPAS)

TESTES = [
    "tests/test_gawll.py", "tests/test_importancia.py", "tests/test_sintetico.py",
    "tests/test_comparacao.py", "tests/test_interacao.py", "tests/test_fidelidade.py",
    "tests/test_sintese.py", "tests/test_balanceamento.py", "tests/test_recortes.py",
]


def configurar_log() -> None:
    """Manda o log para o console e para experiments/runs/pipeline.log.

    Usada em: main.
    Importância: execuções longas (o PyGAwLLfs leva ~5,4 h) precisam de registro em
    arquivo, que sobrevive ao terminal.
    """
    ARQUIVO_LOG.parent.mkdir(parents=True, exist_ok=True)
    formato = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    logger.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(), logging.FileHandler(ARQUIVO_LOG, encoding="utf-8")):
        handler.setFormatter(formato)
        logger.addHandler(handler)


def escolher_etapas(de: str | None, ate: str | None, so: list[str] | None) -> list[Etapa]:
    """Recorta a lista de etapas (--de, --ate, --so) sem mudar a ordem do pipeline.

    Usada em: main.
    Importância: permite retomar de onde parou sem refazer etapas caras.
    """
    nomes = [etapa.nome for etapa in ETAPAS]
    for nome in [de, ate, *(so or [])]:
        if nome is not None and nome not in nomes:
            raise SystemExit(f"Etapa desconhecida: '{nome}'. Opções: {', '.join(nomes)}")
    if so:
        return [etapa for etapa in ETAPAS if etapa.nome in so]
    inicio = nomes.index(de) if de else 0
    fim = nomes.index(ate) + 1 if ate else len(ETAPAS)
    return ETAPAS[inicio:fim]


def rodar_script(script: str) -> float:
    """Roda um script como subprocesso, com o mesmo Python deste processo e a partir da
    raiz; devolve a duração e levanta RuntimeError se ele falhar.

    Usada em: main.
    Importância: cada script continua rodável sozinho; o main só os encadeia, e para
    no primeiro erro para não rodar etapas sobre saídas quebradas.
    """
    inicio = time.perf_counter()
    resultado = subprocess.run([sys.executable, script], cwd=RAIZ)
    duracao = time.perf_counter() - inicio
    if resultado.returncode != 0:
        raise RuntimeError(f"{script} terminou com código {resultado.returncode}")
    return duracao


def formatar(segundos: float) -> str:
    """Segundos -> texto legível ("42 s", "3.5 min", "5.4 h").

    Usada em: main (log da duração de cada etapa).
    Importância: só apresentação.
    """
    if segundos < 60:
        return f"{segundos:.0f} s"
    if segundos < 3600:
        return f"{segundos / 60:.1f} min"
    return f"{segundos / 3600:.1f} h"


def main() -> None:
    """Ponto de entrada do pipeline: lê os argumentos (--listar, --de, --ate, --so,
    --testes, --simular), roda as etapas em ordem e, se pedido, as suítes de teste.

    Usada em: python main.py.
    Importância: reproduz o projeto inteiro com um comando, na ordem certa, e
    registra quanto tempo cada etapa levou.
    """
    parser = argparse.ArgumentParser(description="Pipeline do TCC (PyGAwLLfs e atrasos de voos)")
    parser.add_argument("--listar", action="store_true", help="lista as etapas e sai")
    parser.add_argument("--de", help="começa nesta etapa")
    parser.add_argument("--ate", help="termina nesta etapa")
    parser.add_argument("--so", nargs="+", help="roda só estas etapas")
    parser.add_argument("--testes", action="store_true", help="roda as suítes de teste no fim")
    parser.add_argument("--simular", action="store_true", help="mostra o que rodaria, sem rodar")
    args = parser.parse_args()

    if args.listar:
        for fase in FASES:
            print(f"\n[{fase}]")
            for etapa in (e for e in ETAPAS if e.fase == fase):
                print(f"  {etapa.nome:13s} {etapa.tempo_esperado:30s} {etapa.descricao}")
        print(f"\n  {'(--testes)':13s} {'~6 min':30s} as {len(TESTES)} suítes de teste")
        return

    etapas = escolher_etapas(args.de, args.ate, args.so)
    if args.simular:
        for etapa in etapas:
            print(f"[{etapa.nome}] {etapa.fase} ({etapa.tempo_esperado})")
            for script in etapa.scripts:
                print(f"    {Path(sys.executable).name} {script}")
        if args.testes:
            print("[testes]\n" + "\n".join(f"    python {t}" for t in TESTES))
        return

    configurar_log()
    logger.info(f"Etapas: {', '.join(e.nome for e in etapas)} | Python: {sys.executable}")

    duracoes = []
    for etapa in etapas:
        logger.info(f"=== {etapa.nome} ({etapa.fase}): {etapa.descricao} "
                    f"(esperado: {etapa.tempo_esperado})")
        inicio = time.perf_counter()
        try:
            for script in etapa.scripts:
                logger.info(f"--- {script}")
                rodar_script(script)
        except RuntimeError as erro:
            logger.error(f"{erro}. Corrija e retome com: python main.py --de {etapa.nome}")
            sys.exit(1)
        duracoes.append((etapa.nome, time.perf_counter() - inicio))
        logger.info(f"=== {etapa.nome} concluída em {formatar(duracoes[-1][1])}")

    if args.testes:
        logger.info("=== testes")
        inicio = time.perf_counter()
        try:
            for teste in TESTES:
                logger.info(f"--- {teste}")
                rodar_script(teste)
        except RuntimeError as erro:
            logger.error(f"Teste falhou: {erro}")
            sys.exit(1)
        duracoes.append(("testes", time.perf_counter() - inicio))

    logger.info("Resumo:\n" + "\n".join(f"  {nome:13s} {formatar(segundos)}"
                                       for nome, segundos in duracoes)
                + f"\n  {'total':13s} {formatar(sum(s for _, s in duracoes))}")


if __name__ == "__main__":
    main()
