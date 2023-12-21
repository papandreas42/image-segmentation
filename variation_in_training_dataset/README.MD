# Variation in training dataset size

This folder contains the scripts necessary to run the variation in training dataset experiment on DTU's HPC cluster.

## Prerequisites

- SSH access to your region on DTU hpc
- Clone of the image-segmentation repository

## Steps

1. Connect to your region on DTU hpc using SSH.
2. Clone the image-segmentation repository and change your current working directory to the repository.
3. Copy the dataset zip file to `image-segmentation/` using scp or by installing and using the python gdown module (https://stackoverflow.com/a/50670037)
4. Unzip the dataset `unzip "./training_dataset.zip" -d "./"` 
5. Change the working directory to `image-segmentation/variation_in_training_dataset`
6. Create the python environment
    1. `module load python3/3.10.12`
    2. `python3 -m venv ./.venv`
    3. `source .venv/bin/activate`
    4. `pip install -r requirements.txt`
7. Rename the dataset images by running the `python rename_dataset.py` script.
8. Create a `.env_var` file using the `.env_var_example` template. Update the file with the path to the repository and your wandb API key.
9. Run the baseline experiment for 200 epochs using the command `bsub < job_script.sh`.
10. Modify the for loop in `multiple_jobs_script.sh` (line 8) to submit train jobs with a range of `n_train` values.

- Please contact us for any additional information or troubleshooting.
