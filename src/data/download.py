import logging
import time
from pathlib import Path
from urllib.parse import quote

import requests

BASE_URL = "https://sistemas.anac.gov.br/dadosabertos"
CATEGORIA = "Voos e operações aéreas"
SUBPASTA = "Voo Regular Ativo (VRA)"

MESES_PT = {
    1: "Janeiro",
    2: "Fevereiro",
    3: "Março",
    4: "Abril",
    5: "Maio",
    6: "Junho",
    7: "Julho",
    8: "Agosto",
    9: "Setembro",
    10: "Outubro",
    11: "Novembro",
    12: "Dezembro",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def montar_url_vra(ano: int, mes: int) -> str:
    """
    Função que recebe dois inteiros, um ano e um mes e monta a url correta para fazer
    o acesso aos dados de VRA da ANAC
    """
    pasta_mes = f"{mes:02d} - {MESES_PT[mes]}"  # Zero a esquerda
    nome_arquivo = f"VRA_{ano}{mes}.csv"  # Sem zero a esquerda
    caminho = f"{CATEGORIA}/{SUBPASTA}/{ano}/{pasta_mes}/{nome_arquivo}"
    return f"{BASE_URL}/{quote(caminho)}"


def baixar_arquivo(url: str, destino: Path) -> None:
    """Função que faz downloads dos csv das url geradas por montar_url_vra(...)
    Faz validação se o arquivo já existe

    Args:
        url (str): URL retornada pela função montar_url_vra(...)
        destino (Path): Caminho do csv onde serão escritas os dados
    """
    if destino.exists():
        logger.info(f"Já existe, pulando: {destino.name}")
        return

    resposta = requests.get(url=url, stream=True, timeout=40)
    resposta.raise_for_status()

    with open(destino, "wb") as f:
        f.writelines(resposta.iter_content(chunk_size=8192))

    logger.info(f"Baixado: {destino.name}")


# Diretório onde serão salvos os CSV
DATA_RAW = Path("data/raw")
# Loop para fazer o download
for ano in [2024, 2025]:
    for mes in MESES_PT:
        url = montar_url_vra(ano=ano, mes=mes)
        destino = DATA_RAW / f"VRA_{ano}_{mes:02d}.csv"
        try:
            baixar_arquivo(url=url, destino=destino)
        except requests.exceptions.RequestException as e:
            logger.error(f"Falhou {ano}-{mes:02d}: {e}")
        time.sleep(1)
