#!/bin/bash
#SBATCH --job-name supervised_masters
#SBATCH --error supervised_masters-error.e%j
#SBATCH --output supervised_masters-out.o%j
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 20000
#SBATCH --partition shared-gpu
#SBATCH --gpus=1
#SBATCH --time 11:55:00

# Loading required modules
module load GCC/10.2.0 CUDA/11.1.1
module load Python/3.8.6

. /home/users/p/pulfer/cdp_supervised_ad/venv/bin/activate

srun python3 -u /home/users/p/pulfer/cdp_supervised_ad/src/simple.py \
        --t_dir /home/users/p/pulfer/cdp_supervised_ad/dataset/templates \
        --x_dir /home/users/p/pulfer/cdp_supervised_ad/dataset/originals \
        --f_dir /home/users/p/pulfer/cdp_supervised_ad/dataset/fakes \
        --extra_dir /home/users/p/pulfer/cdp_supervised_ad/dataset/synthetic_printed \
        --train_indices /home/users/p/pulfer/cdp_supervised_ad/dataset/train.txt \
        --val_indices /home/users/p/pulfer/cdp_supervised_ad/dataset/val.txt \
        --n_epochs 30 \
        --lr 0.01 \
        --batch_size 8 \
        --result_dir results
        # --concat_ty \
