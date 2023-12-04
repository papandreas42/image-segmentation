import numpy as np
import matplotlib.pyplot as plt

validation_dices = []

patience = 8
early_stopping_counter = 0

for epoch in range(1, 31):
    validation_dices.append(np.random.randint(1, 100))
    print("#"*100)
    print("EPOCH: ", epoch)
    if epoch > patience:
        print("epoch larger than patience",max(validation_dices[-(patience+1):-1]), validation_dices[-1])
        to_print = validation_dices.copy()
        to_print[:-(patience+1)] = [0]*len(to_print[:-(patience+1)])
        is_smaller = max(validation_dices[-(patience+1):-1]) > validation_dices[-1]
        if is_smaller:
            early_stopping_counter += 1
        else:
            early_stopping_counter = 0

        if early_stopping_counter >= patience:
            print("early stopping")
            print("best epoch=", validation_dices.index(max(validation_dices))+1)
        print(is_smaller)
        print(early_stopping_counter)
        print(validation_dices)
        print(to_print[:-1])

