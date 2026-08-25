# RLCI — Reference-Loss-Calibrated Initialization

Código e resultados associados ao benchmark da **Inicialização Calibrada por
Loss de Referência (RLCI)**, desenvolvido no contexto de uma pesquisa de
Mestrado em Ciência da Computação sobre inicialização de pesos em redes neurais.

## Nota de nomenclatura

O arquivo executado originalmente utiliza o identificador **ECED**, que era o
nome de trabalho do método durante o desenvolvimento experimental.

Na versão final da dissertação, o método passou a ser denominado:

**RLCI — Reference-Loss-Calibrated Initialization**  
**Inicialização Calibrada por Loss de Referência**

Por essa razão, este repositório preserva duas versões:

- `experiments/eced_gaussian_experiment_original.py`: arquivo original, mantido
  para rastreabilidade e reprodução dos resultados;
- `experiments/rlci_gaussian_benchmark.py`: versão com a nomenclatura final RLCI.
  A lógica experimental foi preservada; as alterações são de nomenclatura e
  nomes de saída.

## Ideia central

Para classificação multiclasse com entropia cruzada, a RLCI utiliza como
referência a loss associada à predição uniforme:

```text
L_ref = ln(C)
```

A escala global dos pesos é escolhida de modo que a loss anterior ao treinamento
fique suficientemente próxima de `L_ref`. Entre as escalas que satisfazem essa
restrição, seleciona-se a que apresenta a menor loss de validação após um curto
horizonte de treinamento.

No benchmark da dissertação:

```text
C = 5
L_ref = ln(5) ~= 1.609438
tau = 0.10
E = 3 épocas
```

A especificação completa está em
[`docs/experimental_protocol.md`](docs/experimental_protocol.md).

A equivalência da versão renomeada é documentada em
[`docs/renaming_validation.md`](docs/renaming_validation.md).

## Estrutura

```text
.
├── CITATION.cff
├── LICENSE
├── README.md
├── requirements.txt
├── docs/
│   └── experimental_protocol.md
├── experiments/
│   ├── eced_gaussian_experiment_original.py
│   └── rlci_gaussian_benchmark.py
├── results/
│   ├── README.md
│   ├── runs_original.csv
│   ├── learning_curves_original.csv
│   ├── summary_reconstructed_from_runs.csv
│   ├── runs_rlci_labels.csv
│   ├── learning_curves_rlci_labels.csv
│   └── summary_rlci_labels.csv
└── scripts/
    ├── reproduce_original.bat
    └── reproduce_original.sh
```

## Instalação

Recomenda-se utilizar um ambiente virtual separado.

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```bat
.venv\Scripts\activate
```

Instale as dependências:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> `requirements.txt` contém faixas de versões compatíveis e não deve ser
> interpretado como um `pip freeze` exato do computador no qual o benchmark
> original foi executado.

## Reprodução do benchmark da dissertação

Linux/macOS:

```bash
bash scripts/reproduce_original.sh
```

Windows:

```bat
scripts\reproduce_original.bat
```

Ou diretamente:

```bash
python experiments/eced_gaussian_experiment_original.py \
  --seeds 10 \
  --regimes medium hard \
  --n-samples 6000 \
  --n-features 32 \
  --n-classes 5 \
  --hidden-dims 128 64 32 \
  --activation relu \
  --batch-size 128 \
  --search-epochs 3 \
  --final-epochs 20 \
  --alphas 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 \
  --optimizer sgd \
  --lr 0.03 \
  --momentum 0.9 \
  --weight-decay 0 \
  --device auto
```

O script grava os resultados em `eced_results/`:

- `alpha_search.csv`
- `runs.csv`
- `learning_curves.csv`
- `summary.csv`
- `config.json`

## Resultados de referência

Os seguintes valores são recalculados a partir de `results/runs_original.csv`.
O nome histórico `ECED` é apresentado abaixo como `RLCI`.

| Regime | Método | Escala média | Loss final de teste | Acurácia final |
|---|---|---:|---:|---:|
| medium | RLCI | 0.60 | 0.503 ± 0.080 | 0.888 ± 0.015 |
| medium | He | 1.00 | 0.595 ± 0.067 | 0.878 ± 0.012 |
| medium | Xavier | 1.00 | 0.585 ± 0.065 | 0.881 ± 0.011 |
| medium | Orthogonal | 1.00 | 0.547 ± 0.074 | 0.892 ± 0.011 |
| hard | RLCI | 0.78 | 1.233 ± 0.092 | 0.761 ± 0.016 |
| hard | He | 1.00 | 1.316 ± 0.084 | 0.743 ± 0.016 |
| hard | Xavier | 1.00 | 1.317 ± 0.086 | 0.754 ± 0.008 |
| hard | Orthogonal | 1.00 | 1.253 ± 0.075 | 0.766 ± 0.009 |

Esses valores servem como referência para verificar uma nova execução. Em GPU,
pequenas diferenças podem ocorrer conforme versões do PyTorch/CUDA e operações
do cuBLAS.

## Reprodutibilidade em CUDA

O script habilita algoritmos determinísticos no PyTorch. Em CUDA >= 10.2,
o cuBLAS também pode exigir a variável:

Linux/macOS:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Windows PowerShell:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
```

Defina a variável **antes** de iniciar o processo Python/Jupyter.

## Observação sobre a inicialização da camada de saída

Nas camadas ocultas, cada baseline utiliza sua regra correspondente.
A camada classificadora final é inicializada com Xavier normal e `gain=1`
em todos os métodos. O fator global de escala é então aplicado a todas as
matrizes de pesos.

Essa escolha está preservada no código original.

## Dados sintéticos

Cada seed gera um novo conjunto independente de mistura gaussiana balanceada,
heteroscedástica e anisotrópica.

Regimes:

| Regime | Separação | Escala cov. | Anisotropia | Ruído |
|---|---:|---:|---:|---:|
| medium | 3.0 | 1.0 | 8 | 3% |
| hard | 1.8 | 1.3 | 20 | 8% |

## Arquivos originais e arquivos renomeados

Para auditoria científica, utilize os arquivos que contêm `original` no nome.
Eles preservam a nomenclatura e os resultados históricos.

Os arquivos com `rlci_labels` possuem os mesmos dados numéricos, substituindo
apenas o rótulo `ECED` por `RLCI` para compatibilidade com a terminologia final.

## Citação

Depois de publicar o repositório, edite `CITATION.cff` e substitua:

```text
https://github.com/SEU-USUARIO/rlci-weight-initialization
```

pela URL definitiva.

Recomenda-se criar uma tag/release imutável para a dissertação, por exemplo:

```text
v1.0.0-dissertation
```

e citar essa release ou seu DOI, caso o repositório seja arquivado no Zenodo.

## Licença

MIT. Consulte [`LICENSE`](LICENSE).
