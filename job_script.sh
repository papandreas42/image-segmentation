#!/bin/sh 

### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J testjob
### Avoiding the parallel job message
#BSUB -R "span[hosts=1]"
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 00:20
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
### #BSUB -B
### -- send notification at completion--
### #BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

# Load environment variables
source ./.env_var
export N_TRAIN=7
echo "JOB_SCRIPT_NTRAIN IS $N_TRAIN"


# Create job_out if it is not present
if [[ ! -d ${REPO}/job_out ]]; then
	mkdir ${REPO}/job_out
fi

date=$(date +%Y%m%d_%H%M)
unix_timestamp=$(date +%s)

mkdir ${REPO}/runs/train/${date}_${unix_timestamp}

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

mv *gpu_* ${REPO}/job_out
mv checkpoints ${REPO}/runs/train/${date}_${unix_timestamp}