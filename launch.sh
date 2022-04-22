#!/bin/bash
#SBATCH --job-name=compo              # Job name
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G                     # Memory request; MB assumed if unit not specified
#SBATCH --time=01:00:00               # Time limit hrs:min:sec
#SBATCH --output=%x-%j.log            # Standard output and error log

echo "Running job on $(hostname)"

# load conda environment
source /shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate egg36

# launch your computation
python -m egg.zoo.compo_vs_generalization.train --n_epochs=2 --n_attributes=10 --n_values=4 --vocab_size=20 --max_len=20 --batch_size=5120 --sender_cell=lstm --receiver_cell=lstm --random_seed=1 --data_scaler=1  --train_size=500 --test_size=30 --validation_size=30
# python -m egg.zoo.compo_vs_generalization.data
