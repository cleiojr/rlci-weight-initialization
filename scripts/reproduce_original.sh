#!/usr/bin/env bash
set -euo pipefail

python experiments/eced_gaussian_experiment_original.py   --seeds 10   --regimes medium hard   --n-samples 6000   --n-features 32   --n-classes 5   --hidden-dims 128 64 32   --activation relu   --batch-size 128   --search-epochs 3   --final-epochs 20   --alphas 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2   --optimizer sgd   --lr 0.03   --momentum 0.9   --weight-decay 0   --device auto
