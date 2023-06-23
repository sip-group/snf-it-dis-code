#!/bin/sh
#SBATCH --job-name cdp_unsupervised_ad
#SBATCH --error error.e%j
#SBATCH --output out.o%j
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 20000
#SBATCH --partition shared-gpu
#SBATCH --gpus=ampere:1
#SBATCH --time 11:59:59

# Loading required modules
module load GCC/10.2.0 CUDA/11.3.1
module load Python/3.8.6

# Installing torchvision in a Python Virtual environment
# virtualenv ~/Projects/cdp_unsupervised_ad/venv
. ~/Projects/cdp_unsupervised_ad/venv/bin/activate
# ~/Projects/cdp_unsupervised_ad/venv/bin/python -m pip install --upgrade pip
# # pip install torch
# pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
# pip install numpy matplotlib opencv-python sklearn
# pip list
# pip install -r ~/Projects/cdp_unsupervised_ad/requirements.txt

# Running Main Program
#~/Projects/cdp_unsupervised_ad/scripts/run_wifs.sh
#~/Projects/cdp_unsupervised_ad/scripts/run_mobile.sh
#~/Projects/cdp_unsupervised_ad/scripts/run_transfer.sh
#~/Projects/cdp_unsupervised_ad/scripts/run_efficiency.sh
~/Projects/cdp_unsupervised_ad/scripts/run_cross.sh