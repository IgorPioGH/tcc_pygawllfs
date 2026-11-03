# Registro de decisões

As escolhas de método do trabalho, cada uma com o motivo e a alternativa que ficou
de fora. A descrição completa de cada etapa está nos documentos por fase
([`01-problema.md`](01-problema.md) a [`05-avaliacao.md`](05-avaliacao.md)).

| # | Decisão | Fase |
|---|---|---|
| 1 | Divisão temporal, com o teste em nov–dez/2025 | Preparação |
| 2 | Balanceamento 50/50 global em cada conjunto | Preparação |
| 3 | 29 atributos, com *target encoding* e porte no lugar de *one-hot* | Preparação |
| 4 | Ano fora dos atributos | Preparação |
| 5 | Acurácia como métrica central | Modelagem |
| 6 | Random Forest fixo em 100 árvores e profundidade 10 | Modelagem |
| 7 | Fitness do PyGAwLLfs na validação | Modelagem |
| 8 | Erros de implementação do PyGAwLLfs corrigidos | Modelagem |
| 9 | 200 gerações, com as taxas do código | Modelagem |
| 10 | Subamostra de 50 mil voos para o PyGAwLLfs | Modelagem |
| 11 | Sem SHAP na regressão logística | Modelagem |
| 12 | H de Friedman nos 406 pares, em logit, com H_abs como versão principal | Modelagem |
| 13 | Média das sementes da interação só onde o par foi sondado | Avaliação |
| 14 | Tipo do par pela definição dos atributos | Avaliação |
| 15 | Recortes temporais com o mesmo calendário | Avaliação |
| 16 | Incerteza pelo erro-padrão entre dias | Avaliação |
| 17 | Desenho da fidelidade | Avaliação |
| 18 | Parâmetros do dataset sintético | Avaliação |

## 1. Divisão temporal, com o teste em nov–dez/2025

- **Decisão:** treino até agosto de 2025, validação em setembro e outubro, teste em
  novembro e dezembro, por data absoluta.
- **Motivo:** o modelo prevê o futuro a partir do passado; um sorteio misturaria
  voos do mesmo dia entre treino e teste. O teste cai no pico sazonal (24,3% de
  atrasos, contra 17,9% até outubro): é o período mais recente disponível.
- **Alternativa:** divisão aleatória 70/30, como no artigo do PyGAwLLfs.

## 2. Balanceamento 50/50 global em cada conjunto

- **Decisão:** em treino, validação e teste, todos os atrasados e o mesmo número de
  pontuais, sorteados no conjunto inteiro, depois do porte e do encoding.
- **Motivo:** com 50/50, a acurácia é legível (o trivial acerta 0,5) e é a mesma
  métrica do artigo e do código do PyGAwLLfs. Sortear no conjunto inteiro
  multiplica as *odds* de todos os grupos pelo mesmo fator e preserva a ordem entre
  eles (conferido: Spearman de 0,997 na taxa por hora).
- **Alternativa:** manter a proporção original (cerca de 18%) e usar F1 ou PR-AUC
  com pesos de classe; ou balancear por grupo, o que apagaria a informação da
  variável.

## 3. 29 atributos, com *target encoding* e porte no lugar de *one-hot*

- **Decisão:** empresa e aeroportos entram como taxa de atraso suavizada
  (*out-of-fold* no treino) e como faixa de porte; o tipo de linha entra como
  *one-hot* (4 valores).
- **Motivo:** o PyGAwLLfs trabalha com um bit por atributo, e a matriz de interação
  e o H de Friedman crescem com o quadrado do número de atributos; *one-hot* de
  centenas de aeroportos tornaria os dois inviáveis.
- **Alternativa:** *one-hot* completo, ou incluir regiões geográficas (exigiria uma
  tabela externa).

## 4. Ano fora dos atributos

- **Decisão:** o ano não é atributo.
- **Motivo:** validação e teste são 100% de 2025; o modelo só aprenderia a diferença
  entre os anos no treino, e ela nunca seria testada. Um atributo assim, com
  importância alta, distorceria os rankings que o trabalho compara.

## 5. Acurácia como métrica central

- **Decisão:** a acurácia escolhe os hiperparâmetros na validação, é o desempenho no
  fitness do PyGAwLLfs, a métrica da Permutation e a da fidelidade. As outras
  métricas (precisão, recall, F1, ROC-AUC, PR-AUC, acurácia balanceada) continuam
  calculadas.
- **Motivo:** com os conjuntos 50/50 ela é informativa, e é a métrica do artigo, do
  código original e do tutorial de importância de atributos usado como referência.
  Usar a mesma métrica em todos os métodos baseados em desempenho torna os seus
  números comparáveis.

## 6. Random Forest fixo em 100 árvores e profundidade 10

- **Decisão:** não ajustar a profundidade do Random Forest.
- **Motivo:** é o Random Forest do código do PyGAwLLfs; assim o GA e os demais
  métodos explicam o mesmo modelo.

## 7. Fitness do PyGAwLLfs na validação

- **Decisão:** o GA treina na parte de treino da subamostra e mede o fitness na
  parte de validação; o teste nunca entra na busca.
- **Motivo:** o algoritmo do artigo mede o fitness no conjunto de teste, e a busca
  otimizaria sobre o próprio conjunto de avaliação. A acurácia de validação do
  melhor indivíduo é otimista (ele foi escolhido olhando esse número) e não é
  reportada como desempenho; o desempenho do subconjunto escolhido sai da
  fidelidade, no teste.

## 8. Erros de implementação do PyGAwLLfs corrigidos

- **Decisão:** usar a mutação e o elitismo corrigidos, por subclasse, sem editar o
  repositório original (Adendo A de [`04-modelagem.md`](04-modelagem.md)).
- **Motivo:** o código original não executa o que o artigo descreve; os testes
  reproduzem os dois erros. Rodar com eles seria comparar métodos usando um GA sem
  mutação.
- **Alternativa:** rodar também a versão original, para medir o efeito dos erros.

## 9. 200 gerações, com as taxas do código

- **Decisão:** população 50, 200 gerações, taxa de cruzamento 0,8 e mutação 1/N.
- **Motivo:** com 100 gerações (as do artigo), cada execução sonda cerca de metade
  dos 406 pares, e a matriz de interação fica pouco confiável. Com 200, são 600
  sondagens e cerca de 77% dos pares sondados. As taxas são as do código, que é o
  que roda (Adendo B de [`04-modelagem.md`](04-modelagem.md)).
- **Alternativa:** mais sementes com 100 gerações, ou as taxas reportadas no artigo.

## 10. Subamostra de 50 mil voos para o PyGAwLLfs

- **Decisão:** o GA usa 50 mil voos de treino e validação, estratificados por
  período e alvo; todo o resto usa os conjuntos completos.
- **Motivo:** o GA retreina o modelo milhares de vezes por execução; com o treino
  completo, as 15 execuções seriam inviáveis. A checagem da subamostra mede se isso
  muda o que é importante.

## 11. Sem SHAP na regressão logística

- **Decisão:** TreeSHAP só no Random Forest e no XGBoost.
- **Motivo:** o TreeSHAP é exato e rápido para árvores; na regressão logística, os
  coeficientes padronizados cumprem o papel da importância baseada na função do
  modelo. KernelSHAP seria aproximado e caro nesta escala.

## 12. H de Friedman nos 406 pares, em logit, com H_abs como versão principal

- **Decisão:** calcular o H em todos os pares, na escala logit, com 500 voos da
  validação e 5 sementes de amostra; ordenar os pares pelo H não normalizado.
- **Motivo:** cobrir o mesmo universo de pares que o PyGAwLLfs pode sondar, sem
  escolher pares a partir de outros resultados. No logit, a regressão logística é
  exatamente aditiva (H = 0), então ela fica de fora. O H normalizado é instável
  quando o efeito conjunto do par é pequeno e passa de 1 em pares de *dummies* da
  mesma variável; as duas versões são salvas.

## 13. Média das sementes da interação só onde o par foi sondado

- **Decisão:** na média das 5 execuções do PyGAwLLfs, cada par usa só as execuções
  em que foi sondado (pelo contador de sondagens), e só pares sondados em pelo menos
  2 execuções entram no ranking, no top-k e no grafo.
- **Motivo:** a matriz de interação de uma execução começa em zero e só registra o
  par quando uma sondagem o sorteia; contar o par não sondado como "interação zero"
  faria o ranking acompanhar quantas vezes o par foi sorteado, e não a força da
  interação. Uma sondagem única guarda o resultado de um retreino só.

## 14. Tipo do par pela definição dos atributos

- **Decisão:** classificar cada par como estrutural, redundante ou operacional pela
  forma como `features.py` cria os atributos ([`05-avaliacao.md`](05-avaliacao.md)).
- **Motivo:** o tipo é conhecido por construção (`periodo_dia = hora // 6`, por
  exemplo) e não precisa ser estimado; assim a classificação não depende de nenhum
  teste nos dados.

## 15. Recortes temporais com o mesmo calendário

- **Decisão:** comparar 2024 e 2025, cada um com treino de janeiro a agosto e
  validação em setembro e outubro, com os hiperparâmetros do experimento principal e
  uma execução do PyGAwLLfs por modelo e recorte.
- **Motivo:** com o mesmo calendário, a diferença entre os recortes reflete a
  mudança de um ano para o outro, e não a sazonalidade. A estabilidade entre
  sementes do experimento principal dá a régua para ler a diferença.

## 16. Incerteza pelo erro-padrão entre dias

- **Decisão:** EP = desvio-padrão das acurácias diárias no teste / √61; diferença
  menor que 2·EP é empate.
- **Motivo:** uma medida simples, tirada dos próprios dados, que reconhece que os
  voos do mesmo dia não são independentes. Ela é otimista, porque dias vizinhos
  também se parecem; isso fica registrado como limitação.
- **Alternativa:** *bootstrap* por blocos de dias, mais completo e mais caro de
  explicar.

## 17. Desenho da fidelidade

- **Decisão:** retreinar cada modelo no treino completo só com as top-k features de
  cada método (k = 5, 10, 15), contra 10 sorteios de k features, o modelo com todas e
  o acaso; avaliar no teste, por último.
- **Motivo:** mede se as features que um método aponta carregam a informação, com
  uma régua de acaso. Os mesmos hiperparâmetros do modelo original garantem que a
  diferença vem só das features.

## 18. Parâmetros do dataset sintético

- **Decisão:** z = 2 · x1 · x2 + x3 + x4 + 0,5 · ruído; x5 e x6 com correlação 0,9
  com x3 e x4; 5.000 linhas.
- **Motivo:** com esses valores, a interação vale para as árvores (a regressão
  logística, linear, não consegue usá-la) e tirar x3 custa pouco quando x5 está
  presente, que é a redundância que se quer observar.
