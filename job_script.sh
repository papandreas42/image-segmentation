#!/bin/sh 


### General LSF options 
[...] refer to official HPC guide
### -- end of LSF options --

# Load environment variables
source ./.env_var

# Create job_out if it is not present
if [[ ! -d ${REPO}/job_out ]]; then
	mkdir ${REPO}/job_out
fi

date=$(date +%Y%m%d_%H%M)
mkdir ${REPO}/runs/train/${date}

# Activate venv
module load python3/3.10.12
module load cuda/12.1
source ${REPO}/.env/bin/activate

# Exit if previous command failed
if [[ $? -ne 0 ]]; then
	exit 1
fi

# run training
python3 early-stopping.py
