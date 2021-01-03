#!/bin/bash
#SBATCH --job-name=baghdad
#SBATCH --output=2020-05-25-18:46.out
#SBATCH --error=2020-05-25-18:46.err
#SBATCH --nodes=1
# [@commented] SBATCH --ntasks-per-node=1
#SBATCH --partition=COMPUTE

# export JOBLIB_START_METHOD="forkserver"

# source env/bin/activate

#/opt/slurm-19.05.1-2/bin/srun -l python dataset.py -vv
/opt/slurm-19.05.1-2/bin/srun -l nnictl create --config config.yml --foreground
# /opt/slurm-17.11.0/bin/srun -l python testing.py -vv
