#!/bin/bash

set_n_train_in_job_script() {
    local n_train=$1
    sed -i "32s/.*/export N_TRAIN=$n_train/" ./job_script.sh
}

for i in {65..105..20}
do
    set_n_train_in_job_script $i
    bsub < job_script.sh
done