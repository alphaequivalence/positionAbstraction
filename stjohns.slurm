#!/bin/bash
#SBATCH --job-name=stjohns
#SBATCH --output=stjohns.out
#SBATCH --error=stjohns.err
#SBATCH --nodes=1
# [@commented] SBATCH --ntasks-per-node=1
#SBATCH --partition=COMPUTE

# export JOBLIB_START_METHOD="forkserver"

# source env/bin/activate

/opt/slurm-19.05.1-2/bin/srun -l python fl.py -vv
#/opt/slurm-19.05.1-2/bin/srun -l python dataset.py -vv
#/opt/slurm-19.05.1-2/bin/srun -l nnictl create --config tehran_1.yml --foreground
#/opt/slurm-17.11.0/bin/srun -l python testing.py -vv
