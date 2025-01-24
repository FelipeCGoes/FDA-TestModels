# FDA TestModels
Optimization scripts and testing scripts for the FDA project

#### Detalhamento:

Para todos os scripts, a alteração entre pipelines ocorre descomentando os trechos relevantes que compõem cada pipeline. Vale destacar que no caso dos pipelines que envolvem Imputation, o dataset precisa ser alterado de "basePreProcessedAllAbFinal.csv" para "basePreProcessedAllAbFinal_comNaN.csv" pois o primeiro não inclui valores NaN.

Os scripts de teste realizam o calculo das métricas da forma que tem sido feito para incluir na planilha, por threshold e com uma seleção de métricas derivada dos testes com o Synclass. Uma vez definido os testes finais um novo script será feito com os testes de interesse para o melhor modelo/pipeline de cada classificador.
