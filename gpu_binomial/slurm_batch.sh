#!/bin/bash

#SBATCH -A 2017-45
#SBATCH -J gpu-option-pricing
#SBATCH -t 0:30:00
#SBATCH --nodes=1
#SBATCH --workdir .

#SBATCH -e error_file.e

#reserves a node with a k80 GPU
#SBATCH --gres=gpu:K80:2

# Run the executable file
./a.out > batch_output 2>&1
