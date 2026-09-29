# Validação da versão renomeada

A versão `experiments/rlci_gaussian_benchmark.py` foi criada a partir do
arquivo experimental original preservado em
`experiments/eced_gaussian_experiment_original.py`.

As alterações realizadas na versão RLCI são de nomenclatura e nomes de saída.
A lógica de geração dos dados, inicialização, busca de escala, treinamento e
avaliação foi mantida.

## Teste de equivalência

Foi executado um teste determinístico em CPU com:

- 1 seed
- regime `medium`
- 500 amostras
- 1 época de busca
- 3 épocas finais
- alphas `0.5` e `0.6`

Os arquivos `runs.csv` das duas versões foram comparados após substituir apenas
o rótulo de método `RLCI` por `ECED`.

Resultado da comparação:

```text
mesmo número de linhas e colunas: sim
diferenças numéricas: nenhuma
diferenças textuais após normalização do nome do método: nenhuma
```

Esse teste verifica a equivalência da refatoração de nomenclatura nessa
configuração. Para auditoria dos resultados publicados, o arquivo original
continua sendo a referência primária.
