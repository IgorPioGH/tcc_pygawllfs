"""Roda o pipeline do TCC de ponta a ponta, na ordem certa.

Cada etapa é um ou mais scripts de src/, executados como processos separados
(python src/.../arquivo.py) a partir da raiz do repositório — exatamente como
rodar cada um à mão. Este arquivo só cuida da ordem, do tempo de cada etapa e
de parar no primeiro erro, dizendo de onde retomar.

Uso (da raiz, com o ambiente ativado: conda activate tcc_pygawllfs):
    python main.py                       # o pipeline inteiro, do download à fidelidade
    python main.py --listar              # as etapas, o que fazem e o tempo esperado
    python main.py --de importancias     # a partir de uma etapa
    python main.py --ate baselines       # até uma etapa
    python main.py --so dominio rankings # só as etapas citadas (na ordem do pipeline)
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


@dataclass
class Etapa:
    nome: str
    scripts: list[str]
    descricao: str
    tempo_esperado: str


# Ordem obrigatória: cada etapa lê o que as anteriores produziram.
ETAPAS = [
    Etapa("download", ["src/data/download.py"],
          "baixa os 24 CSVs mensais do VRA (ANAC) para data/raw/; pula os já baixados",
          "~5 min (depende da rede)"),
    Etapa("dados", ["src/data/ler_dados.py", "src/data/clean.py",
                    "src/data/target.py", "src/data/features.py"],
          "consolida, limpa, cria o alvo, as 29 features, o split temporal e a subamostra",
          "~3 min"),
    Etapa("baselines", ["src/models/baselines.py"],
          "LogReg, RF e XGBoost: tuning na validação e avaliação única no teste",
          "~3 min"),
    Etapa("pygawllfs", ["src/importance/gawll.py", "src/importance/consolidar_gawll.py"],
          "15 execuções do PyGAwLLfs corrigido (3 modelos x 5 sementes) e suas tabelas",
          "~5,4 h (retoma do checkpoint)"),
    Etapa("importancias", ["src/importance/intrinsic.py", "src/importance/permutation.py",
                           "src/importance/shap_global.py"],
          "importância intrínseca, Permutation e TreeSHAP, na validação",
          "~18 min"),
    Etapa("rankings", ["src/compare/rankings.py", "src/compare/stability.py",
                       "src/compare/figuras.py"],
          "concordância entre rankings, estabilidade entre sementes e figuras",
          "< 1 min"),
    Etapa("sintetico", ["src/synthetic/generate.py", "src/synthetic/aplicar_metodos.py",
                        "src/synthetic/recovery.py"],
          "Diferencial 1: dataset com gabarito, todos os métodos e a recuperação da verdade",
          "~30 min (~3 min com os checkpoints do GA)"),
    Etapa("dominio", ["src/domain/validate_pairs.py"],
          "Diferencial 4: validação dos pares de interação do PyGAwLLfs nos dados reais",
          "~1 min"),
    Etapa("interacoes", ["src/interaction/friedman_h.py", "src/compare/interacoes.py"],
          "H de Friedman e comparação com as interações do PyGAwLLfs",
          "~9 min"),
    Etapa("fidelidade", ["src/compare/fidelity.py"],
          "retenção top-k, subconjuntos do GA e sorteios; avalia no teste",
          "~50 min"),
    Etapa("extras", ["src/data/figuras.py", "src/data/auditar_congestionamento.py"],
          "histograma do atraso e auditoria do congestionamento (limitação documentada)",
          "~4 min"),
]

TESTES = [
    "tests/test_gawll.py", "tests/test_importancia.py", "tests/test_sintetico.py",
    "tests/test_comparacao.py", "tests/test_dominio.py", "tests/test_interacao.py",
    "tests/test_fidelidade.py",
]


def configurar_log() -> None:
    ARQUIVO_LOG.parent.mkdir(parents=True, exist_ok=True)
    formato = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    logger.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(), logging.FileHandler(ARQUIVO_LOG, encoding="utf-8")):
        handler.setFormatter(formato)
        logger.addHandler(handler)


def escolher_etapas(de: str | None, ate: str | None, so: list[str] | None) -> list[Etapa]:
    """Recorta a lista de etapas sem mudar a ordem do pipeline."""
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
    """Roda um script com o mesmo Python deste processo; para tudo se ele falhar."""
    inicio = time.perf_counter()
    resultado = subprocess.run([sys.executable, script], cwd=RAIZ)
    duracao = time.perf_counter() - inicio
    if resultado.returncode != 0:
        raise RuntimeError(f"{script} terminou com código {resultado.returncode}")
    return duracao


def formatar(segundos: float) -> str:
    if segundos < 60:
        return f"{segundos:.0f} s"
    if segundos < 3600:
        return f"{segundos / 60:.1f} min"
    return f"{segundos / 3600:.1f} h"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline do TCC (PyGAwLLfs e atrasos de voos)")
    parser.add_argument("--listar", action="store_true", help="lista as etapas e sai")
    parser.add_argument("--de", help="começa nesta etapa")
    parser.add_argument("--ate", help="termina nesta etapa")
    parser.add_argument("--so", nargs="+", help="roda só estas etapas")
    parser.add_argument("--testes", action="store_true", help="roda as suítes de teste no fim")
    parser.add_argument("--simular", action="store_true", help="mostra o que rodaria, sem rodar")
    args = parser.parse_args()

    if args.listar:
        for etapa in ETAPAS:
            print(f"{etapa.nome:13s} {etapa.tempo_esperado:42s} {etapa.descricao}")
        print(f"{'(--testes)':13s} {'~6 min':42s} as {len(TESTES)} suítes de teste")
        return

    etapas = escolher_etapas(args.de, args.ate, args.so)
    if args.simular:
        for etapa in etapas:
            print(f"[{etapa.nome}] ({etapa.tempo_esperado})")
            for script in etapa.scripts:
                print(f"    {Path(sys.executable).name} {script}")
        if args.testes:
            print("[testes]\n" + "\n".join(f"    python {t}" for t in TESTES))
        return

    configurar_log()
    # O download.py grava em data/raw/ mas não cria a pasta: num clone novo, ela não existe
    (RAIZ / "data" / "raw").mkdir(parents=True, exist_ok=True)
    logger.info(f"Etapas: {', '.join(e.nome for e in etapas)} | Python: {sys.executable}")

    duracoes = []
    for etapa in etapas:
        logger.info(f"=== {etapa.nome}: {etapa.descricao} (esperado: {etapa.tempo_esperado})")
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
