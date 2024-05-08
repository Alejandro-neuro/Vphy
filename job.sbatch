#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6144
#SBATCH --mail-type=END
#SBATCH --gres=gpu:a40:3

module use /opt/insy/modulefiles
module load cuda/11.2 cudnn/11.2-8.1.1.33

echo 'conda activate vphys'


srun python ./Figures/init_exp.py 