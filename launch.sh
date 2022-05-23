#!/bin/bash
#SBATCH --job-name=compo              # Job name
#SBATCH --partition=gpu_p3
#SBATCH --cpus-per-task=1
#SBATCH --time=1200
#SBATCH --output=test.log            # Standard output and error log

echo "Running job on $(hostname)"

# launch your computation
# python -m egg.zoo.compo_vs_generalization.train --n_epochs=10 --n_attributes=10 --n_values=10 --vocab_size=100 --max_len=40 --batch_size=5120 --sender_cell=lstm --receiver_cell=lstm --random_seed=38401 --data_scaler=1  --train_size=10000 --test_size=100 --validation_size=100 --curriculum
# python -m egg.zoo.compo_vs_generalization.data
python test_cuda.py
