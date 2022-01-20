#!/bin/bash
#SBATCH --account=amt           ### SLURM account to charge for job
#SBATCH --partition=amt         ### queue to submit to
#SBATCH --job-name=0709_0_12    ### job name
#SBATCH --output=0709_0_12.out  ### file in which to store job stdout
#SBATCH --error=0709_0_12.err   ### file in which to store job stderr
#SBATCH --time=20-00:00:00      ### wall-clock time limit in Days-HH:MM:SS
#SBATCH --mem=100000M           ### memory limit per cpu in MB
#SBATCH --nodes=1               ### number of nodes to use
#SBATCH --ntasks-per-node=1     ### number of tasks to launch per node
#SBATCH --cpus-per-task=1       ### number of cores for each task, up to 40 for amt
#SBATCH --nodelist=n237         ### not n234

module load tensorflow
python3 time_miner.py 2018-07-09T00:01:10.0Z 2018-07-09T12:00:00.0Z