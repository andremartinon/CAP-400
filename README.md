# CAP-400 - Visualização e Análise Computacional de Séries Temporais

Repositório das atividades desenvolvidas na disciplina CAP-400 (2020),
ministrada pelo
[Professor Dr. Reinaldo Rosa](https://github.com/reinaldo-rosa-inpe)
no programa de pós-graduação em
[Computação Aplicada do INPE](http://www.inpe.br/posgraduacao/cap/).

Durante o desenvolvimento destas atividades foram implementadas 
duas bibliotecas:

* https://github.com/andremartinon/pyGPA
* https://github.com/andremartinon/pyCML

## Configuração do ambiente Python

Recomenda-se criar um ambiente python utilizando o anaconda ou o mininconda.

Com as seguintes bibliotecas instaladas:

    conda install tensorflow keras pandas matplotlib scikit-learn scikit-image scipy
    pip install git+https://github.com/andremartinon/pyGPA
    pip install git+https://github.com/andremartinon/pyCML

Para a geração das animações é necessário ter o utilitário ffmpeg instalado.

Todo o desenvolvimento foi realizado para SO Linux (Ubuntu 20.04).
Para executar em outro SO será necessário compatibilizar os programas.

## Breve descrição dos programas

### lstm.py

Contém a classe TimeSeriesLSTM utilizada para treinamento e predição das séries
temporais de métricas extraídas dos conjuntos de dados de CML, onde o objetivo
principal é identificar estados de relaxação nestes conjuntos de dados.

### datasets.py

Este programa gera todos os conjuntos de dados CML
(gráficos, animações e arquivos de saída) necessários para a resolução da lista de
exercícios.