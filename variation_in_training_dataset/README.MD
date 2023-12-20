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
5. Rename the dataset images using the `rename_dataset.py` script.
5. Create a `.env_var` file using the `.env_var_example` template. Update the file with the path to the repository and your wandb API key.
6. Create the python environment
    * `python -m venv ./.env`
    * `pip install -r requirements.txt`
6. Run the baseline experiment for 200 epochs using the command `bsub < job_script.sh`.
7. Modify the for loop in `multiple_jobs_script.sh` (line 8) to submit train jobs with a range of `n_train` values.

- Please contact us for any additional information or troubleshooting.
