import subprocess

def set_n_train_in_var_file(n_train):
    with open('./env_var', 'rw') as file:
        # read a list of lines into data
        data = file.readlines()

        data[2] = "export N_TRAIN=" + n_train + "\n"
        file.writelines( data )


for i in range(1,5):
    set_n_train_in_var_file(i)
    print(subprocess.run(["source",
                          './env_var'], shell=True))
    print(subprocess.run(["echo",
                          '$N_TRAIN'], shell=True))
