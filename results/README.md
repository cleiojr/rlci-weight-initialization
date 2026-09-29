# Resultados incluídos

Os arquivos desta pasta derivam da execução experimental utilizada na análise
da dissertação.

- `runs_original.csv`: resultados por regime, seed e método, preservando o nome histórico `ECED`.
- `learning_curves_original.csv`: curvas por época, também preservando `ECED`.
- `summary_reconstructed_from_runs.csv`: agregação recalculada diretamente de `runs_original.csv`.
- `runs_rlci_labels.csv`: mesmo conteúdo de `runs_original.csv`, apenas com `ECED` renomeado para `RLCI`.
- `learning_curves_rlci_labels.csv`: mesma regra de renomeação.
- `summary_rlci_labels.csv`: resumo com a nomenclatura final.

## Importante

O arquivo `alpha_search.csv` original não estava disponível no conjunto de
artefatos usado para montar este repositório. Ele não foi reconstruído nem
inventado. O script experimental o gera automaticamente em uma nova execução.
