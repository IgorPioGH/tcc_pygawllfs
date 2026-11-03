# Reprodutibilidade

Como reproduzir os resultados a partir de um clone limpo, e o que fixa cada número.

## Versões

| Item | Versão |
|---|---|
| Dados | CSVs do VRA (ANAC) com a marca "Atualizado em: 2026-10-07", baixados em 23/10/2026; a base bruta tem 1.992.831 registros |
| PyGAwLLfs | https://github.com/dgabrl/PyGAwLLfs, commit `3be28a694164e9b2cb99fc9d1fb9482d6a5af454` (18/02/2026) |
| Python | 3.13, em ambiente conda |
| Bibliotecas | `requirements.txt` (entre elas pandas 3.0.5, scikit-learn 1.9.0, xgboost 3.4.1, shap 0.52.0, matplotlib 3.11.1, holidays 0.103) |

A ANAC atualiza a série histórica. Se `download.py` baixar outra versão, a etapa
`eda` para, porque o número de registros muda; para reproduzir exatamente, é
preciso usar os arquivos da versão acima.

## Hardware

Notebook com Intel Core i7-13620H (16 núcleos lógicos), 14 GB de RAM e NVIDIA RTX
4050 (6 GB). A GPU só acelera o XGBoost (`device="cuda"`); o resto roda na CPU. Sem
GPU NVIDIA, troque `device="cuda"` por `device="cpu"` em `criar_modelo`
(`src/modelagem/baselines.py`).

## Sementes

| O quê | Semente |
|---|---|
| *target encoding* (dobras), balanceamento, subamostra, modelos | 42 |
| PyGAwLLfs, Permutation, SHAP (amostra), H de Friedman (amostra) | 42, 43, 44, 45, 46 |
| sorteios de features da fidelidade | 42 |
| recortes temporais (PyGAwLLfs, Permutation, SHAP) | 42 |
| dataset sintético | 42 |

## Instalação

```bash
git clone https://github.com/IgorPioGH/tcc_pygawllfs.git && cd tcc_pygawllfs
conda create -n tcc_pygawllfs python=3.13
conda activate tcc_pygawllfs
pip install -r requirements.txt
git clone https://github.com/dgabrl/PyGAwLLfs.git PyGAwLLfs
git -C PyGAwLLfs checkout 3be28a694164e9b2cb99fc9d1fb9482d6a5af454
```

O projeto e o clone do PyGAwLLfs têm, cada um, uma pasta `src/` sem `__init__.py`;
as duas se combinam no `sys.path`. Nenhuma das duas deve ganhar `__init__.py`.

## Execução

Sempre da raiz do repositório, com o ambiente ativado:

```bash
python main.py --listar        # as etapas, por fase do CRISP-DM, e o tempo esperado
python main.py --testes        # o pipeline inteiro e, no fim, as suítes de teste
python main.py --de pygawllfs  # retoma a partir de uma etapa
python main.py --so eda        # só algumas etapas
python main.py --simular       # mostra os comandos, sem rodar
```

As etapas longas retomam de onde pararam: o PyGAwLLfs, o sintético e os recortes
pulam as execuções que já têm checkpoint em `experiments/runs/`. Para o notebook não
suspender: `systemd-inhibit --what=idle:sleep python main.py --testes`. O log fica
em `experiments/runs/pipeline.log`, e cada script também roda sozinho
(`python src/<pasta>/<arquivo>.py`).

**Ensaio rápido:** `TCC_GERACOES=5 python main.py --testes` roda tudo com 5
gerações do PyGAwLLfs, para conferir a integração em cerca de uma hora. Os
checkpoints de um ensaio têm de ser apagados antes da execução completa
(`rm -rf experiments/runs`): a consolidação do PyGAwLLfs recusa checkpoints com
número de gerações diferente de `TCC_GERACOES`.

## Etapas

| Fase | Etapa | Scripts | Tempo aproximado |
|---|---|---|---|
| Dados | `download` | `src/dados/download.py` | ~5 min (rede) |
| Dados | `dados` | `src/dados/ler_dados.py` | < 1 min |
| Preparação | `preparacao` | `src/preparacao/{clean,target,features}.py` | < 1 min |
| Dados | `eda` | `src/dados/eda.py` | ~1 min |
| Preparação | `checagens` | `src/preparacao/checagens.py` | < 1 min |
| Modelagem | `baselines` | `src/modelagem/baselines.py` | ~1 min |
| Modelagem | `pygawllfs` | `src/modelagem/{gawll,consolidar_gawll}.py` | ~10 h |
| Modelagem | `importancias` | `src/modelagem/{intrinsic,permutation,shap_global,friedman_h}.py` | ~25 min |
| Avaliação | `desempenho` | `src/avaliacao/{matriz_confusao,erros}.py` | < 1 min |
| Avaliação | `rankings` | `src/avaliacao/{rankings,stability,checagem_subamostra}.py` | ~2 min |
| Avaliação | `sintetico` | `src/avaliacao/sintetico_{gerar,aplicar,recuperar}.py` | ~1 h |
| Avaliação | `interacoes` | `src/avaliacao/interacoes.py` | < 1 min |
| Avaliação | `fidelidade` | `src/avaliacao/fidelity.py` | ~15 min |
| Avaliação | `recortes` | `src/avaliacao/recortes.py` | ~4 h |
| Avaliação | `sintese` | `src/avaliacao/sintese.py` | < 1 min |
| Interpretação | `figuras` | `src/interpretacao/*.py` | ~2 min |

## Testes

São scripts com `assert` (sem pytest), um por parte do pipeline, rodados depois
dele (leem os resultados gerados):

| Suíte | O que garante |
|---|---|
| `test_balanceamento.py` | 50/50 nos três conjuntos, nenhum atrasado perdido, divisão temporal, congestionamento calculado antes do descarte, subamostra sem teste |
| `test_gawll.py` | os dois erros do código original existem e foram corrigidos; a busca nunca abre o teste; o fitness é determinístico; o checkpoint tem 3 sondagens por geração e um melhor fitness que nunca cai |
| `test_importancia.py` | nenhum método de importância abre o teste; formato das tabelas; a intrínseca bate com o modelo salvo; a consolidação reproduz v_imp, v_int e as sondagens |
| `test_sintetico.py` | o gerador e as métricas de recuperação em casos conhecidos; tabelas completas |
| `test_comparacao.py` | métricas de ranking em casos conhecidos; as figuras e tabelas da comparação só leem CSV |
| `test_interacao.py` | o H acerta funções com resposta conhecida; a regressão logística tem H = 0; a regra das sondagens e o tipo dos pares; nada lê o teste; 406 pares por execução |
| `test_fidelidade.py` | listas de features, resumo e o modelo com todas as features igual ao `baselines.csv` |
| `test_sintese.py` | a síntese repete os números das tabelas de origem |
| `test_recortes.py` | conjuntos dos recortes (datas, 50/50, colunas), o script não acessa o teste de nenhum recorte, 638 linhas de importância |
