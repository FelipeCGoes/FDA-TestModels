# FDA TestModels
Optimization scripts and testing scripts for the FDA project

#### Detalhamento:

Para todos os scripts, a alteração entre pipelines ocorre descomentando os trechos relevantes que compõem cada pipeline. Vale destacar que caso pipeline que envolva Imputation, a base de dados precisa ser alterada de "basePreProcessedAllAbFinal.csv" para "basePreProcessedAllAbFinal_comNaN.csv" pois a primeira não inclui valores NaN.

Os scripts de teste envolvem o calculo das métricas da forma que tem sido feito para incluir na planilha, por threshold e com uma seleção de métricas derivada dos testes com o Synclass. Uma vez definido os testes finais um novo script será feito com os testes de interesse para o melhor modelo/pipeline de cada classificador.
