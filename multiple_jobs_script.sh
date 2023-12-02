#!/bin/bash

set_n_train_in_var_file() {
    local n_train=$1
    sed -i "3s/.*/export N_TRAIN=$n_train/" ./.env_var
}

for i in {1..4}
do
    set_n_train_in_var_file $i
    source ./.env_var
    echo $N_TRAIN
done