# Avaliação comparativa de métodos de importância e interação de atributos em modelos de aprendizado de máquina aplicados à previsão de atrasos de voos.
**Pergunta central:** Como diferentes métodos de estimação de importância e interação de atributos se comparam em termos de concordância, estabilidade, fidelidade e custo computacional no problema de previsão de atrasos de voos?

**Autor:** Igor Pio — Bacharelado em Ciência da Computação, FFCLRP/USP, Ribeirão Preto
**Orientador:** Prof. Dr. Renato Tinós

**Status Atual:** Em desenvolvimento

## Configuração de Ambiente
- Ambiente Virtual criado e configurado utilizando conda e pip.
```bash
conda create -n tcc_pygawllfs python=3.13
conda activate tcc_pygawllfs
pip install -r requirements.txt
```
- Todas as bibliotecas estão no arquivo requirements.txt
- Os dados brutos estão no `.gitignore`, dessa forma é necessário fazer o download dos mesmos com a função do arquivo `src/data/download.py`