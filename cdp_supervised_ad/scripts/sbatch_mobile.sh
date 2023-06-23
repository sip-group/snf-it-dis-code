#!/bin/sh
#SBATCH --job-name indigo
#SBATCH --error indigo-error.e%j
#SBATCH --output indigo-out.o%j
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 20000
#SBATCH --partition shared-gpu
#SBATCH --gpus=4
#SBATCH --time 11:55:00

# Loading required modules
module load GCC/10.2.0 CUDA/11.1.1
module load Python/3.8.6

# projectd dir
PROJECT_DIR=indigo_supervised_ad

# Installing torchvision in a Python Virtual environment
# virtualenv ~/${PROJECT_DIR}/venv
. ~/${PROJECT_DIR}/venv/bin/activate
# pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# pip install numpy matplotlib pillow tensorboard opencv-python
# pip install pandas seaborn scikit-learn
# pip install setuptools==59.5.0
# pip install -r ~/${PROJECT_DIR}/requirements.txt

# Result directories
mkdir -p ${HOME}/${PROJECT_DIR}/results

# Running multiple configurations (different originals and fakes)
srun ${HOME}/${PROJECT_DIR}/scripts/run_multiple_confs.sh ${HOME}/${PROJECT_DIR}/configurations/mobile/a.json
srun ${HOME}/${PROJECT_DIR}/scripts/run_multiple_confs.sh ${HOME}/${PROJECT_DIR}/configurations/mobile/b.json
