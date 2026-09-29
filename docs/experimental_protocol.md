# Protocolo experimental da RLCI

Este documento registra o benchmark associado aos resultados da RLCI
apresentados na dissertação.

## Problema

Classificação multiclasse em misturas gaussianas sintéticas balanceadas,
heteroscedásticas e anisotrópicas.

- 6.000 amostras por conjunto
- 32 atributos
- 5 classes
- divisão estratificada: 70% treino, 15% validação e 15% teste
- `StandardScaler` ajustado exclusivamente no conjunto de treino

## Regimes

| Regime | Separação | Escala de covariância | Anisotropia | Ruído de rótulo |
|---|---:|---:|---:|---:|
| medium | 3.0 | 1.0 | 8 | 3% |
| hard | 1.8 | 1.3 | 20 | 8% |

Foram utilizadas 10 sementes pareadas por regime.

## Arquitetura

MLP:

`32 -> 128 -> 64 -> 32 -> 5`

- ativação ReLU
- sem Batch Normalization
- vieses zerados

Nas camadas ocultas, a RLCI utiliza Kaiming/He como inicializador-base.
A camada classificadora final utiliza Xavier normal com `gain=1`.
O fator global `alpha` é aplicado a todas as matrizes de pesos.

## Loss de referência

Para `C` classes:

`L_ref = ln(C)`

No benchmark:

`L_ref = ln(5) ~= 1.609438`

O erro relativo é:

`epsilon_ref(alpha) = |L0(alpha) - L_ref| / L_ref`

## Seleção da escala

Grade:

`A = {0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2}`

Tolerância:

`tau = 0.10`

Uma escala é viável quando:

`epsilon_ref(alpha) <= tau`

Entre as escalas viáveis, seleciona-se aquela com menor loss de validação
depois de 3 épocas.

Se nenhuma escala satisfizer a tolerância, aplica-se a ordenação
lexicográfica:

1. menor erro relativo à loss de referência;
2. menor loss após as 3 épocas de busca.

O modelo final é então reinicializado com as mesmas direções aleatórias
e treinado do zero. As épocas usadas na busca não são reaproveitadas.

## Otimização

- SGD
- learning rate: 0.03
- momentum: 0.9
- batch size: 128
- weight decay: 0
- busca: 3 épocas por alpha
- treino final: 20 épocas

## Baselines

- He/Kaiming, escala 1.0
- Xavier/Glorot, escala 1.0
- Orthogonal, escala 1.0

## Nota histórica

O arquivo executado originalmente empregava o nome de trabalho `ECED`.
Durante a redação final, o método foi renomeado para `RLCI`.
A mudança de nomenclatura não altera a lógica experimental.
