# Machine Learning e Visão Computacional \[T1\] 

# Mini-Projeto Avaliativo \- Módulo 2 \- Semana 07

**SUMÁRIO**

[1 CONTEXTUALIZAÇÃO	1](#1-contextualizaÇÃo)

[2 DESAFIO	1](#2-desafio)

[3 RESULTADOS ESPERADOS (ENTREGA)	2](#3-resultados-esperados-\(entrega\))

[4 REQUISITOS DAS TAREFAS	2](#4-requisitos-das-tarefas)

[4.1 GRAVAÇÃO DE VÍDEO	3](#4.1-gravaÇÃo-de-vÍdeo)

[5 CRITÉRIOS DE AVALIAÇÃO	3](#5-critÉrios-de-avaliaÇÃo)

# **1 CONTEXTUALIZAÇÃO**

A Indústria 4.0 transformou a maneira como as fábricas operam, integrando tecnologias inteligentes para otimizar a produção e garantir o controle de qualidade. No mercado de trabalho atual, a automação da inspeção visual exige profissionais capazes de transitar entre duas abordagens: a Visão Clássica e a Inteligência Artificial.

Muitas vezes, antes de treinarmos uma rede neural complexa, precisamos usar ferramentas de processamento clássico (como a biblioteca OpenCV) para realizar uma análise exploratória, limpando ruídos, destacando bordas e compreendendo as características físicas do defeito que a máquina precisará procurar. Somente após essa compreensão, aplicamos Redes Neurais Convolucionais (CNNs) para automatizar a classificação em larga escala, substituindo inspeções manuais demoradas e sujeitas a erros humanos.

Neste projeto, utilizaremos o dataset público *Casting Product Image Data for Quality Inspection* (disponível no Drive: [https://drive.google.com/file/d/1NZOjCHDRrpn7PmbFKVqegUP5arfdXHKK/view?usp=sharing](https://drive.google.com/file/d/1NZOjCHDRrpn7PmbFKVqegUP5arfdXHKK/view?usp=sharing) ), que contém imagens reais de peças de fundição metálica com e sem defeitos estruturais.

# **2 DESAFIO**

Você foi contratado como Analista de Inteligência Artificial Júnior por uma indústria metalúrgica. Seu desafio técnico é desenvolver um pipeline (script) em Python que una processamento clássico e aprendizado profundo.

O objetivo do seu código é duplo:

1. **Análise Exploratória (OpenCV):** Extrair uma amostra de imagens do dataset e aplicar um pipeline clássico de tratamento (escala de cinza, blur, limiarização, detecção de bordas e morfologia) para provar visualmente que é possível destacar a ranhura ou trinca da peça.  
2. **Classificação Automatizada (TensorFlow/Keras):** Ingerir o lote completo de imagens, aplicar *Data Augmentation* dinâmico para simular variações de esteira e treinar uma Rede Neural Convolucional (CNN) básica capaz de classificar a peça como "OK" ou "Defeituosa". Ao final, gerar o gráfico de *Loss* e *Acurácia* do modelo.

# **3 RESULTADOS ESPERADOS (ENTREGA)**

O arquivo `readme.md` com a solução do mini-projeto e o vídeo deverão ser inseridos no Google Drive em modo leitor para qualquer pessoa com o link, e o planejamento e organização ficará a critério de cada estudante. Os links deverão ser submetidos na tarefa **Módulo 2 \- Mini-Projeto Avaliativo**, presente no AVA, até o dia **14/09/2026 às 22h.** 

# **4 REQUISITOS DAS TAREFAS**

Para garantir a organização e o sucesso do projeto, divida seu desenvolvimento nas seguintes Sprints industriais:

* **Sprint 1 \- Configuração e Versionamento:** Inicializar um repositório Git, configurar o ambiente de notebooks e fazer o download do dataset.  
* **Sprint 2 \- Análise Exploratória Clássica (OpenCV):** Selecionar imagens de amostra e aplicar técnicas de conversão para escala de cinza (*Grayscale*) e suavização de ruídos industriais (ex: *Gaussian Blur* ou *Median Blur*).  
* **Sprint 3 \- Destaque de Características:** Ainda na amostra, aplicar limiarização (*Thresholding*) e algoritmos de detecção de bordas (*Canny* ou *Sobel*), além de operações morfológicas (Erosão/Dilatação) para isolar visualmente os defeitos da peça.  
* **Sprint 4 \- Ingestão de Dados e Augmentation (Keras):** Utilizar image\_dataset\_from\_directory para carregar o dataset completo dividindo em Treino e Validação. Implementar *Data Augmentation* (giros, zoom, brilho) para imunizar a IA.  
* **Sprint 5 \- A Arquitetura CNN e Treinamento:** Construir um modelo Sequencial com camadas Conv2D e MaxPooling2D, finalizando com o achatamento (Flatten) e a camada de classificação binária (Dense). Executar o treinamento com o otimizador *Adam*.  
* **Sprint 6 \- Auditoria e Gravação:** Gerar o gráfico analítico do treinamento (Curvas de *Loss* e *Val\_Loss*), redigir o readme.md finalizando a branch no GitHub e gravar o vídeo técnico de apresentação.




## **4.1 GRAVAÇÃO DE VÍDEO**

Além do desenvolvimento do código, você deverá gravar um vídeo, com tempo máximo de 5 minutos, abordando os seguintes questionamentos:

1. Qual o objetivo do sistema e demonstração de funcionamento no notebook?  
2. O que as técnicas clássicas de OpenCV (Canny, Blur) revelaram sobre os defeitos da peça na sua análise exploratória?  
3. Como você estruturou a sua CNN e o *Data Augmentation* para aprender esses padrões em larga escala?  
4. Analisando o seu gráfico de *Loss* gerado no final, ocorreu *Overfitting* ou o modelo aprendeu de forma saudável?

Você poderá gravar na vertical ou na horizontal. O arquivo de vídeo (ou link) deve estar acessível via Google Drive.

# **5 CRITÉRIOS DE AVALIAÇÃO**

A tabela abaixo apresenta os critérios que serão avaliados durante a correção do projeto, variando de 0 a 10 pontos. Projetos que apresentarem plágio de soluções encontradas na internet ou de outros colegas receberão nota 0 (zero). O uso de IA generativa para consulta é permitido, desde que o código final seja compreendido e defendido pelo aluno no vídeo.

| Apresentação do Projeto (Nota máxima no bloco: 2,0) |  |  |  |  |
| :---: | ----- | ----- | ----- | ----- |

| Nº | Critério de Avaliação | 0 | 2,0 |  |
| :---: | :---: | :---: | :---: | ----- |
| **1** | **Gravação de vídeo** | Não foi realizada a gravação do vídeo.  | Gravou o vídeo e abordou todos os tópicos listados no item 4.1 |  |

| Uso adequado do GitHub e [Readme.md](http://Readme.md) (Nota máxima no bloco: 2,0) |  |  |  |  |
| :---: | ----- | ----- | ----- | ----- |

| Nº | Critério de Avaliação | 0 | 0,5 | 1.0 |
| :---: | ----- | :---: | :---: | :---: |
| **2** | **Versionamento com branches e commits** | O repositório do projeto não apresenta branches e commits. | O repositório do projeto apresenta parte das branches e commits distintos e nomeadas padronizadamente para cada funcionalidade desenvolvida. | O repositório do projeto apresenta branches e commits distintos e nomeadas padronizadamente para cada funcionalidade desenvolvida. |
| **3** | **Organização dos arquivos no repositório** | O repositório do projeto não apresenta os arquivos estruturados conforme as instruções.  | O repositório do projeto apresenta parte dos arquivos estruturados conforme as instruções.  | O repositório do projeto apresenta os arquivos estruturados conforme as instruções com Readme.md estruturado  |

| Desenvolvimento do Projeto |  |  |  |  |
| :---: | ----- | ----- | ----- | ----- |

| Nº | Critério de Avaliação | 0 | 0,50 | 1,00 |
| :---: | ----- | ----- | ----- | ----- |
| **4** | **OpenCV: Filtros Básicos**  | Não aplicou os filtros básicos de tratamento de imagem. | Aplicou apenas uma das etapas (somente conversão de cor ou somente o filtro de suavização) na amostra de análise. |  Aplicação correta de conversão de cores (Grayscale) e filtros de suavização (Blur) na amostra de análise. |
| **5** | **OpenCV: Bordas e Morfologia**  | Não aplicou técnicas de detecção de bordas ou morfologia. | Aplicou as técnicas de forma parcial ou desorganizada, sem destacar o defeito da peça. | Executou o pré-processamento de imagens aplicando Limiarização, detecção de bordas (Canny/Sobel) e operações morfológicas para evidenciar falhas no componente.  |
| **6** | **Ingestão e Augmentation**  | Lê apenas imagens fixas ou não realizou leitura automatizada do dataset. | Lê pastas com falhas de carregamento ou não aplicou o Data Augmentation adequadamente em ambas as formas (geométrica e luminosa). | Implementou o carregamento automatizado de datasets em lote via Keras e aplica técnicas de Data Augmentation com variações geométricas e luminosas.  |
| **7** | **Arquitetura CNN**  | Não aplicou a estrutura de rede neural convolucional. | Aplicou a estrutura de forma incorreta ou incompleta (ex.: ausência ou mau encadeamento de blocos). | Construiu a arquitetura da CNN intercalando camadas de convolução (Conv2D) e agrupamento (MaxPooling2D) para a extração de características da imagem.  |
| **8** | **Compilação e Decisão** | Não aplicou as camadas de transição e classificação final. | Aplicou incorretamente a transição matricial ou a função de perda inadequada para o problema proposto. | Realizou a transição matricial via Flatten e configura a camada Densa final com função de perda binária para a classificação de duas classes.  |
| **9** | **Auditoria Gráfica**  | Não gerou a visualização gráfica dos dados do modelo. | Gerou apenas parte dos gráficos ou com eixos/curvas incorretos. | Gerou a representação gráfica do histórico do modelo exibindo o comportamento das curvas de Loss e Acurácia (Treino vs. Validação por meio da execução de rotinas em Matplotlib ou Seaborn.  |

